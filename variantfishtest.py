import sys
import os
import random
import warnings
import argparse
import math  # NEW

import stat_util
import chess.uci

import logging


RESULTS = [WIN, LOSS, DRAW] = range(3)
SCORES = [1, 0, 0.5]


def elo_stats(scores):
    try:
        elo, elo95, los = stat_util.get_elo(scores)
        return "ELO: %.2f +-%.1f (95%%) LOS: %.1f%%\n" % (elo, elo95, 100 * los)
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
        self.parser.add_argument("--e1-options", help="options for first UCI engine", type=lambda kv: kv.split("="), action='append', default=[])
        self.parser.add_argument("--e2-options", help="options for second UCI engine", type=lambda kv: kv.split("="), action='append', default=[])
        self.parser.add_argument("-v", "--variant", help="choose a chess variant", type=str, default="chess")
        self.parser.add_argument("-c", "--config", help="path to variants.ini", type=str)
        self.parser.add_argument("-n", "--max_games", help="maximum number of games", type=int, default=5000)
        self.parser.add_argument("-s", "--sprt", help="perform an SPRT test", action="store_true")
        self.parser.add_argument("--elo0", help="lower bound for SPRT test", type=float, default=0)
        self.parser.add_argument("--elo1", help="upper bound for SPRT test", type=float, default=10)
        self.parser.add_argument("-t", "--time", help="base time in milliseconds", type=int, default=10000)
        self.parser.add_argument("-i", "--inc", help="time increment in milliseconds", type=int, default=100)
        self.parser.add_argument("-b", "--book", help="use EPD opening book", nargs="?", type=str, const=True)
        self.parser.add_argument("-l", "--log", help="write output to specified file", type=str, default="")
        self.parser.add_argument("--verbosity",
                                 help="verbosity level: "
                                      "0 - only final results, 1 - intermediate results, 2 - moves of games, 3 - debug",
                                 type=int, choices=[0, 1, 2, 3], default=1)
        self.parser.parse_args(namespace=self)

        self.variants = self.variant.split(',')
        self.variant = self.variants[0]
        self.fens = []
        self.engine_paths = [os.path.abspath(self.engine1), os.path.abspath(self.engine2)]
        self.engine_options = [dict(self.e1_options), dict(self.e2_options)]
        self.out = open(os.path.abspath(self.log), "a") if self.log else sys.stdout

        self.wt = None
        self.bt = None
        self.scores = [0, 0, 0]
        self.r = []
        self.engines = []
        self.time_losses = []

        # NEW: additional stats
        self.white_wins = 0
        self.black_wins = 0
        self.draw_games = 0
        self.pentanomial = [0] * 5  # [LL, LD, DD/WL, WD, WW]

        if self.verbosity > 2:
            logging.basicConfig()
            chess.uci.LOGGER.setLevel(logging.DEBUG)

    def close(self):
        if self.out != sys.stdout:
            self.out.close()

    def run(self):
        """Run a test with previously defined settings."""
        self.print_settings()
        self.init_engines()
        while not self.stop():
            self.variant = random.choice(self.variants)
            # only (re-)init book if required
            if self.book and (len(self.variants) > 1 or not self.fens):
                self.init_book()
            pos = "fen " + random.choice(self.fens) if self.fens else "startpos"
            self.init_game()
            self.process_game(0, 1, pos)
            self.init_game()
            self.process_game(1, 0, pos)
        self.print_results()
        self.close()

    def stop(self):
        if self.max_games and sum(self.scores) >= self.max_games:
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
                sys.exit(path + " does not exist.")
            self.engines.append(chess.uci.popen_engine(path))
        self.info_handlers = []
        for engine, options in zip(self.engines, self.engine_options):
            engine.uci()
            if self.config:
                engine.setoption({"VariantPath": self.config})
            engine.setoption({"UCI_Variant": self.variant})
            engine.setoption(options)

            self.info_handlers.append(chess.uci.InfoHandler())
            engine.info_handlers.append(self.info_handlers[-1])
            self.time_losses.append(0)

    def init_book(self):
        """Read opening book file and fill FEN list."""
        assert self.book
        if self.book is True:
            bookfile = os.path.abspath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "books", self.variant + ".epd"))
        elif self.book:
            bookfile = os.path.abspath(self.book)
        if os.path.exists(bookfile):
            with open(bookfile) as f:
                self.fens = []
                for line in f:
                    self.fens.append(line.rstrip(';\n'))
        else:
            warnings.warn(bookfile + " does not exist. Using starting position.")

    def init_game(self):
        """Prepare for next game."""
        self.bestmoves = []
        self.wt = self.time
        self.bt = self.time
        for engine in self.engines:
            engine.ucinewgame()
            engine.setoption({"clear hash": True, "UCI_Variant": self.variant})

    def play_game(self, white, black, pos="startpos"):
        """Play a game and return the game result from white's point of view."""
        res = None
        offset = 0
        if pos != "startpos" and " b " in pos:
            offset = 1
        while True:
            index = white if (len(self.bestmoves) + offset) % 2 == 0 else black
            e = self.engines[index]
            h = self.info_handlers[index]
            e.send_line("position " + pos + " moves " + " ".join(self.bestmoves))
            bestmove, ponder = e.go(wtime=self.wt, btime=self.bt, winc=self.inc, binc=self.inc)
            self.bestmoves.append(bestmove)

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
                            raise Exception("Invalid game result.\nMove list: " + " ".join(self.bestmoves))
                    # check for 3fold and 50 moves rule
                    elif h.info["score"][1].cp == 0 and h.info["pv"] and len(h.info["pv"][1]) == 1:
                        return DRAW
                    # check for mate in 1
                    elif h.info["score"][1].mate == 1:
                        return WIN if index == white else LOSS
                    # adjust time remaining on clock
                    if index == white:
                        self.wt += self.inc - h.info.get("time", 0)
                        if self.wt < 0:
                            self.time_losses[index] += 1
                            return LOSS
                    else:
                        self.bt += self.inc - h.info.get("time", 0)
                        if self.bt < 0:
                            self.time_losses[index] += 1
                            return WIN
                else:
                    raise Exception("Engine does not return a score.\nMove list: " + " ".join(self.bestmoves))

    def process_game(self, white, black, pos="startpos"):
        """Play a game and process the result."""
        res = self.play_game(white, black, pos)

        # NEW: per-colour tallies
        if res == WIN:
            self.white_wins += 1
        elif res == LOSS:
            self.black_wins += 1
        else:
            self.draw_games += 1

        if self.verbosity > 1:
            self.out.write(
                "Game %d (%s):\n" % (sum(self.scores) + 1, self.variant) + pos + "\n" + " ".join(self.bestmoves) + "\n")
        self.r.append(SCORES[res] if white == 0 else 1 - SCORES[res])
        if white == 0 or res == DRAW:
            self.scores[res] += 1
        else:
            self.scores[1 - res] += 1

        # NEW: pentanomial update (after each game pair)
        if len(self.r) % 2 == 0:
            pair_score = self.r[-2] + self.r[-1]  # 0,0.5,1,1.5,2
            eps = 1e-9
            if abs(pair_score - 0.0) < eps:
                idx = 0  # LL
            elif abs(pair_score - 0.5) < eps:
                idx = 1  # LD
            elif abs(pair_score - 1.0) < eps:
                idx = 2  # DD or WL
            elif abs(pair_score - 1.5) < eps:
                idx = 3  # WD
            else:
                idx = 4  # WW
            self.pentanomial[idx] += 1

        if self.verbosity > 1:
            self.print_results()
        elif self.verbosity > 0:
            self.print_stats()
        self.out.flush()

    def print_stats(self):
        """Print intermediate results."""
        self.out.write(print_scores(self.scores) + " ")
        if self.sprt:
            self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
        else:
            self.out.write(elo_stats(self.scores))

    def print_settings(self):
        """Print settings for test."""
        self.out.write("engine1:    %s\n" % self.engine_paths[0])
        self.out.write("engine2:    %s\n" % self.engine_paths[1])
        self.out.write("e1-options: %s\n" % self.engine_options[0])
        self.out.write("e2-options: %s\n" % self.engine_options[1])
        self.out.write("variants:   %s\n" % self.variants)
        self.out.write("config:     %s\n" % self.config)
        self.out.write("# of games: %d\n" % self.max_games)
        self.out.write("sprt:       %s\n" % self.sprt)
        if self.sprt:
            self.out.write("elo0:       %.2f\n" % self.elo0)
            self.out.write("elo1:       %.2f\n" % self.elo1)
        self.out.write("time:       %d\n" % self.time)
        self.out.write("increment:  %d\n" % self.inc)
        self.out.write("book:       %s\n" % self.book)
        self.out.write("------------------------\n")

    def print_results(self):
        """Print final test result."""
        drawrate = float(self.scores[2]) / sum(self.scores)
        # print(self.r)
        self.out.write("------------------------\n")
        self.out.write("Stats:\n")
        # NEW: percentage output
        self.out.write("draw rate: %.2f%%\n" % (100.0 * drawrate))
        self.out.write("time losses engine1: %d\n" % (self.time_losses[0]))
        self.out.write("time losses engine2: %d\n" % (self.time_losses[1]))
        # NEW: colour balance and pentanomial
        self.out.write("white wins: %d black wins: %d draws: %d\n" % (self.white_wins, self.black_wins, self.draw_games))
        self.out.write("pentanomial [LL LD DD/WL WD WW]: [%s]\n" % (",".join(str(x) for x in self.pentanomial)))
        self.out.write("\n")
        if self.sprt:
            self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
        else:
            self.out.write(elo_stats(self.scores))
        # NEW: normalized Elo with guard for early/degenerate cases
        try:
            elo, _, _ = stat_util.get_elo(self.scores)
            if sum(self.scores) > 1 and drawrate < 1.0 - 1e-9:
                norm_elo = elo / math.sqrt(1.0 - drawrate)
                self.out.write("Normalised ELO: %.2f\n" % norm_elo)
        except (ValueError, ZeroDivisionError):
            pass
        self.out.write(print_scores(self.scores) + "\n")


if __name__ == "__main__":
    match = EngineMatch()
    match.run()
