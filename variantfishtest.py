import chess.uci
import sys, os
import random, math
import stat_util
import warnings
import argparse

VARIANTS = [
  "chess",
  "giveaway",
  "atomic",
  "crazyhouse",
  "horde",
  "kingofthehill",
  "losers",
  "racingkings",
  "relay",
  "threecheck",
]

RESULTS = [WIN, LOSS, DRAW] = range(3)
SCORES = [1, 0, 0.5]

def elo_stats(scores):
    try:
        elo, elo95, los = stat_util.get_elo(scores)
        return "ELO: %.2f +-%.1f (95%%) LOS: %.1f%%\n" % (elo, elo95, 100*los)
    except (ValueError, ZeroDivisionError):
        return "\n"

def sprt_stats(scores, elo1, elo2):
    s = stat_util.SPRT({'wins': scores[0], 'losses': scores[1], 'draws': scores[2]}, elo1, 0.05, elo2, 0.05, 200)
    return "LLR: %.2f (%.2f,%.2f) [%.2f,%.2f]\n" % (s['llr'], s['lower_bound'], s['upper_bound'], elo1, elo2)

def print_scores(scores):
    return "Total: %d W: %d L: %d D: %d" % (sum(scores), scores[0], scores[1], scores[2])

class EngineMatch:
    """Compare two UCI engines by running an engine match."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("engine1", help="absolute or relative path to first UCI engine", type=str)
        self.parser.add_argument("engine2", help="absolute or relative path to second UCI engine", type=str)
        self.parser.add_argument("-v", "--variant", help="choose a chess variant", type=str, choices=VARIANTS, default=VARIANTS[0])
        self.parser.add_argument("-n", "--max_games", help="maximum number of games", type=int, default=5000)
        self.parser.add_argument("-s", "--sprt", help="perform an SPRT test", action="store_true")
        self.parser.add_argument("--elo0", help="lower bound for SPRT test", type=float, default=0)
        self.parser.add_argument("--elo1", help="upper bound for SPRT test", type=float, default=20)
        self.parser.add_argument("-t", "--time", help="base time in milliseconds", type=int, default=10000)
        self.parser.add_argument("-i", "--inc", help="time increment in milliseconds", type=int, default=100)
        self.parser.add_argument("-b", "--book", help="use EPD opening book", action="store_true")
        self.parser.add_argument("-l", "--log", help="write output to specified file", type=str, default="")
        self.parser.add_argument("--verbosity", help="verbosity level: 0 - only final results, 1 - intermediate results, 2 - moves of games",
                                 type=int, choices=[0,1,2], default=1)
        self.parser.parse_args(namespace=self)

        self.fens = []
        self.engine_paths = [os.path.abspath(self.engine1), os.path.abspath(self.engine2)]
        self.out = open(os.path.abspath(self.log), "a") if self.log else sys.stdout

        self.wt = None
        self.bt = None
        self.scores = [0, 0, 0]
        self.r = []
        self.engines = []
        self.time_losses = []

    def close(self):
        if self.out != sys.stdout:
            self.out.close()

    def run(self):
        """Run a test with previously defined settings."""
        self.print_settings()
        self.init_engines()
        self.init_book()
        while not self.stop():
            pos = "fen "+random.choice(self.fens) if self.fens else "startpos"
            self.init_game()
            self.process_game(0, 1, pos)
            self.init_game()
            self.process_game(1, 0, pos)
        self.print_results()
        self.close()

    def stop(self):
        if (self.max_games and sum(self.scores) >= self.max_games):
            return True
        if self.sprt and self.sprt_finished():
            return True
        return False

    def sprt_finished(self):
        """Check whether SPRT test is finished."""
        return stat_util.SPRT({'wins': self.scores[0], 'losses': self.scores[1], 'draws': self.scores[2]},
                              self.elo0, 0.05, self.elo1, 0.05, 200)["finished"]

    def init_engines(self):
        """Setup engines and info handlers."""
        for path in self.engine_paths:
            if not os.path.exists(path):
                sys.exit(path+" does not exist.")
            self.engines.append(chess.uci.popen_engine(path))
        self.info_handlers = []
        for engine in self.engines:
            engine.uci()
            engine.setoption({"UCI_Variant": self.variant})

            self.info_handlers.append(chess.uci.InfoHandler())
            engine.info_handlers.append(self.info_handlers[-1])
            self.time_losses.append(0)

    def init_book(self):
        """Read opening book file and fill FEN list."""
        if not self.book:
            return
        bookfile = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "books", self.variant+".epd"))
        if os.path.exists(bookfile):
            f = open(bookfile)
            for line in f:
                self.fens.append(line.rstrip(';\n'))
        else:
            warnings.warn(bookfile+" does not exist. Using starting position.")

    def init_game(self):
        """Prepare for next game."""
        self.bestmoves = []
        self.wt = self.time
        self.bt = self.time
        for engine in self.engines:
            engine.ucinewgame()
            engine.setoption({"clear hash": True})

    def play_game(self, white, black, pos = "startpos"):
        """Play a game and return the game result from white's point of view."""
        res = None
        offset = 0
        if pos != "startpos" and " b " in pos:
            offset = 1
        while(True):
            index = white if (len(self.bestmoves) + offset) % 2 == 0 else black
            e = self.engines[index]
            h = self.info_handlers[index]
            e.send_line("position "+pos+" moves "+" ".join(self.bestmoves))
            bestmove, ponder = e.go(wtime=self.wt,btime=self.bt,winc=self.inc,binc=self.inc)

            with h:
                if 1 in h.info["score"]:
                    # check for stalemate, checkmate and variant ending
                    if not h.info["pv"] and bestmove == "(none)":
                        warnings.warn("Reached final position. This might cause undefined behaviour.")
                        if h.info["score"][1].cp == 0:
                            return DRAW
                        elif h.info["score"][1].mate == 0 and self.variant in ["giveaway", "losers"]:
                            return WIN if index == white else LOSS
                        elif h.info["score"][1].mate == 0:
                            return LOSS if index == white else WIN
                        else:
                            raise Exception("Invalid game result.\nMove list: "+" ".join(self.bestmoves))
                    # check for 3fold and 50 moves rule
                    elif h.info["score"][1].cp == 0 and len(h.info["pv"][1]) == 1:
                        return DRAW
                    # check for mate in 1
                    elif h.info["score"][1].mate == 1:
                        return WIN if index == white else LOSS
                    # adjust time remaining on clock
                    if index == white:
                        self.wt += self.inc - h.info["time"]
                        if self.wt < 0:
                            self.time_losses[index] += 1
                            return LOSS
                    else:
                        self.bt += self.inc - h.info["time"]
                        if self.bt < 0:
                            self.time_losses[index] += 1
                            return WIN
                else:
                    raise Exception("Engine does not return a score.\nMove list: "+" ".join(self.bestmoves))
            self.bestmoves.append(bestmove)

    def process_game(self, white, black, pos = "startpos"):
        """Play a game and process the result."""
        res = self.play_game(white, black, pos)
        if self.verbosity > 1:
            self.out.write("Game %d:\n" % (sum(self.scores) + 1)+pos+"\n"+" ".join(self.bestmoves)+"\n")
        self.r.append(SCORES[res] if white == 0 else 1 - SCORES[res])
        if white == 0 or res == DRAW:
            self.scores[res] += 1
        else:
            self.scores[1-res] += 1
        if self.verbosity > 0:
            self.print_stats()

    def print_stats(self):
        """Print intermediate results."""
        self.out.write(print_scores(self.scores) + " ")
        if self.sprt:
            self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
        else:
            self.out.write(elo_stats(self.scores))

    def print_settings(self):
        """Print settings for test."""
        self.out.write("engine1:    %s\n"%self.engine_paths[0])
        self.out.write("engine2:    %s\n"%self.engine_paths[1])
        self.out.write("variant:    %s\n"%self.variant)
        self.out.write("# of games: %d\n"%self.max_games)
        self.out.write("sprt:       %s\n"%self.sprt)
        if self.sprt:
            self.out.write("elo0:       %.2f\n"%self.elo0)
            self.out.write("elo1:       %.2f\n"%self.elo1)
        self.out.write("increment:  %d\n"%self.inc)
        self.out.write("book:       %s\n"%self.book)
        self.out.write("------------------------\n")

    def print_results(self):
        """Print final test result."""
        rm = sum(self.r)/len(self.r)
        drawrate = float(self.scores[2])/sum(self.scores)
        #print(self.r)
        self.out.write("------------------------\n")
        self.out.write("Stats:\n")
        self.out.write("draw rate: %.2f\n" % (drawrate))
        self.out.write("time losses engine1: %d\n" % (self.time_losses[0]))
        self.out.write("time losses engine2: %d\n" % (self.time_losses[1]))
        self.out.write("\n")
        if self.sprt:
            self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
        else:
            self.out.write(elo_stats(self.scores))
        self.out.write(print_scores(self.scores) + "\n")

if __name__ == "__main__": 
    match = EngineMatch()
    match.run()