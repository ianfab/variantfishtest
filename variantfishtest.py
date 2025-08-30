#!/usr/bin/env python
import sys
import os
import random
import warnings
import argparse
import threading
import logging
import math

import stat_util
import chess.uci


# Global result codes
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
    """Compare two UCI engines by running an engine match concurrently.
    
    This version uses worker threads; each worker plays a match instance
    (a pair of games with colors swapped) independently, creating its own engine
    processes. Global scores are updated under a lock.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("engines", nargs="+", help="absolute or relative paths to UCI engines (2+ engines for tournament mode)", type=str)
        self.parser.add_argument("--engine-options", help="options for engines (format: engine_index:option=value)", 
                                 type=lambda kv: kv.split(":", 1), action='append', default=[])
        # Backward compatibility arguments  
        self.parser.add_argument("--e1-options", help="options for first UCI engine (deprecated, use --engine-options)", 
                                 type=lambda kv: kv.split("="), action='append', default=[])
        self.parser.add_argument("--e2-options", help="options for second UCI engine (deprecated, use --engine-options)", 
                                 type=lambda kv: kv.split("="), action='append', default=[])
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
        self.parser.add_argument("-T", "--threads", help="number of concurrent game threads", type=int, default=1)
        self.parser.parse_args(namespace=self)

        # Split variants and set default variant (workers choose randomly later)
        self.variants = self.variant.split(',')
        self.variant = self.variants[0]
        self.fens = []
        
        # Handle engines and their options
        self.engine_paths = [os.path.abspath(engine) for engine in self.engines]
        self.num_engines = len(self.engine_paths)
        
        # Backward compatibility attributes
        if self.num_engines >= 1:
            self.engine1 = self.engines[0]
        if self.num_engines >= 2:
            self.engine2 = self.engines[1]
        
        # Tournament mode if more than 2 engines
        self.is_tournament = self.num_engines > 2
        
        # Initialize engine options
        engine_opts = [{} for _ in range(self.num_engines)]
        
        # Handle new-style engine options
        for opt_spec in self.engine_options:
            if len(opt_spec) == 2:
                try:
                    engine_idx_str, option_pair = opt_spec
                    engine_idx = int(engine_idx_str) - 1  # Convert to 0-based index
                    if 0 <= engine_idx < self.num_engines:
                        key, value = option_pair.split("=", 1)
                        engine_opts[engine_idx][key] = value
                    else:
                        print(f"Warning: Invalid engine index {engine_idx + 1}, ignoring option")
                except (ValueError, IndexError):
                    print(f"Warning: Invalid engine option format: {':'.join(opt_spec)}")
        
        # Handle backward compatibility options (only for first 2 engines)
        if self.e1_options and self.num_engines >= 1:
            for opt in self.e1_options:
                if len(opt) == 2:
                    engine_opts[0][opt[0]] = opt[1]
        if self.e2_options and self.num_engines >= 2:
            for opt in self.e2_options:
                if len(opt) == 2:
                    engine_opts[1][opt[0]] = opt[1]
                    
        self.engine_options = engine_opts
        
        self.out = open(os.path.abspath(self.log), "a") if self.log else sys.stdout

        # Score tracking for tournament mode
        if self.is_tournament:
            # For tournaments, track scores per engine pair
            self.engine_pairs = [(i, j) for i in range(self.num_engines) for j in range(i + 1, self.num_engines)]
            self.pair_scores = {pair: [0, 0, 0] for pair in self.engine_pairs}  # [wins_first, wins_second, draws]
            self.pair_time_losses = {pair: [0, 0] for pair in self.engine_pairs}
            # Global aggregated scores (for compatibility)
            self.scores = [0, 0, 0]
            self.time_losses = [0] * self.num_engines
        else:
            # Traditional 2-engine mode
            self.scores = [0, 0, 0]  
            self.time_losses = [0, 0]
        # For SPRT and score list if needed
        self.r = []
        # Number of threads to run concurrently
        self.threads = self.threads

        # additional stats
        self.white_wins = 0
        self.black_wins = 0
        self.draw_games = 0
        self.pentanomial = [0] * 5  # [LL, LD, DD/WL, WD, WW]

        if self.verbosity > 2:
            logging.basicConfig()
            chess.uci.LOGGER.setLevel(logging.DEBUG)
        else:
            chess.uci.enable_stderr_deduplication(True)

        # Lock for updating shared counters
        self.lock = threading.Lock()

    def validate_engine_variants(self):
        """Validate that both engines support all required variants.
        
        Sends 'uci' command to each engine and checks the UCI_Variant option
        to ensure all variants in self.variants are supported.
        Exits with error if any variant is unsupported.
        """
        for engine_idx, engine_path in enumerate(self.engine_paths):
            if self.verbosity >= 3:
                self.out.write(f"Validating engine {engine_idx + 1}: {engine_path}\n")
                self.out.flush()
            
            try:
                # Create temporary engine instance for validation
                engine = chess.uci.popen_engine(engine_path)
                engine.uci()
                
                # Load variant configuration if provided
                if self.config:
                    engine.setoption({"VariantPath": self.config})
                    # Re-query engine options after loading variant configuration
                    # as the available variants may have changed
                    engine.uci()
                
                # Check if UCI_Variant option exists
                if "UCI_Variant" not in engine.options:
                    engine.quit()
                    self.out.write(f"Error: Engine {engine_idx + 1} ({engine_path}) does not support variants (no UCI_Variant option)\n")
                    self.out.flush()
                    sys.exit(1)
                
                # Get supported variants from the var field
                uci_variant_option = engine.options["UCI_Variant"]
                supported_variants = uci_variant_option.var if uci_variant_option.var else []
                
                if self.verbosity >= 3:
                    self.out.write(f"Engine {engine_idx + 1} supported variants: {supported_variants}\n")
                    self.out.flush()
                
                # Check if all required variants are supported
                for variant in self.variants:
                    if variant not in supported_variants:
                        engine.quit()
                        self.out.write(f"Error: Engine {engine_idx + 1} ({engine_path}) does not support variant '{variant}'\n")
                        self.out.write(f"Supported variants: {', '.join(supported_variants)}\n")
                        self.out.flush()
                        sys.exit(1)
                
                # Clean up temporary engine
                engine.quit()
                
            except Exception as e:
                self.out.write(f"Error: Failed to validate engine {engine_idx + 1} ({engine_path}): {e}\n")
                self.out.flush()
                sys.exit(1)
        
        if self.verbosity >= 1:
            self.out.write(f"Variant validation passed for all engines: {', '.join(self.variants)}\n")
            self.out.flush()

    def close(self):
        if self.out != sys.stdout:
            self.out.close()

    def stop(self):
        """Check whether testing should stop."""
        if self.max_games and sum(self.scores) >= self.max_games:
            return True
        if self.sprt and self.sprt_finished():
            return True
        return False

    def sprt_finished(self):
        """Check whether SPRT test is finished."""
        return stat_util.SPRT({'wins': self.scores[0], 'losses': self.scores[1], 'draws': self.scores[2]},
                              self.elo0, 0.05, self.elo1, 0.05, 200)["finished"]

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

    def print_settings(self):
        """Print settings for the test."""
        if self.is_tournament:
            self.out.write("Tournament mode with %d engines:\n" % self.num_engines)
            for i, path in enumerate(self.engine_paths):
                self.out.write("engine%d:    %s\n" % (i + 1, path))
                self.out.write("e%d-options: %s\n" % (i + 1, self.engine_options[i]))
            self.out.write("engine pairs: %s\n" % [f"({p[0] + 1},{p[1] + 1})" for p in self.engine_pairs])
        else:
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
        self.out.write("threads:    %d\n" % self.threads)
        self.out.write("------------------------\n")
        self.out.flush()

    def print_stats(self):
        """Print intermediate results."""
        self.out.write(print_scores(self.scores) + " ")
        if self.sprt:
            self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
        else:
            self.out.write(elo_stats(self.scores))
        self.out.flush()

    def print_results(self):
        """Print final test result."""
        drawrate = float(self.scores[2]) / sum(self.scores) if sum(self.scores) > 0 else 0
        self.out.write("------------------------\n")
        
        if self.is_tournament:
            self.out.write("Tournament Results:\n")
            # Print per-pair results
            for pair in self.engine_pairs:
                pair_scores = self.pair_scores[pair]
                total_games = sum(pair_scores)
                if total_games > 0:
                    self.out.write("Engine %d vs Engine %d: " % (pair[0] + 1, pair[1] + 1))
                    self.out.write("Total: %d W: %d L: %d D: %d " % (total_games, pair_scores[0], pair_scores[1], pair_scores[2]))
                    try:
                        elo, elo95, los = stat_util.get_elo(pair_scores)
                        self.out.write("ELO: %.2f +-%.1f LOS: %.1f%%\n" % (elo, elo95, 100 * los))
                    except (ValueError, ZeroDivisionError):
                        self.out.write("\n")
            self.out.write("\nOverall Stats:\n")
        else:
            self.out.write("Stats:\n")
            
        # Global stats (aggregated for tournaments)
        self.out.write("draw rate: %.2f%%\n" % (100.0 * drawrate))
        
        if self.is_tournament:
            # Print time losses per engine
            for i in range(self.num_engines):
                total_tl = sum(self.pair_time_losses[pair][j] for pair in self.engine_pairs
                               for j in range(2) if (pair[0] == i and j == 0) or (pair[1] == i and j == 1))
                self.out.write("time losses engine%d: %d\n" % (i + 1, total_tl))
        else:
            self.out.write("time losses engine1: %d\n" % (self.time_losses[0]))
            self.out.write("time losses engine2: %d\n" % (self.time_losses[1]))
            
        # colour balance and pentanomial
        self.out.write("white wins: %d black wins: %d draws: %d\n" % (self.white_wins, self.black_wins, self.draw_games))
        self.out.write("pentanomial [LL LD DD/WL WD WW]: [%s]\n" % (",".join(str(x) for x in self.pentanomial)))
        self.out.write("\n")
        if self.sprt:
            self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
        else:
            self.out.write(elo_stats(self.scores))
        # normalized Elo with guard for early/degenerate cases
        try:
            elo, _, _ = stat_util.get_elo(self.scores)
            if sum(self.scores) > 1 and drawrate < 1.0 - 1e-9:
                norm_elo = elo / math.sqrt(1.0 - drawrate)
                self.out.write("Normalised ELO: %.2f\n" % norm_elo)
        except (ValueError, ZeroDivisionError):
            pass
        self.out.write(print_scores(self.scores) + "\n")
        self.out.flush()

    def worker(self):
        """Worker thread: play match instances until the global stop condition is met."""
        while True:
            with self.lock:
                if self.stop():
                    break
            try:
                # Play one match instance (two games with color swap)
                if self.is_tournament:
                    res1, res2, tl1, tl2, engine_pair = self.play_match_instance()
                else:
                    res1, res2, tl1, tl2, engine_pair = self.play_match_instance()
            except Exception as e:
                # Log the exception and continue with the next match instance.
                self.out.write("Error in match instance: %s\n" % e)
                self.out.flush()
                continue
            # Update the counters (each match instance is 2 games)
            with self.lock:
                # Check if we can add these games without exceeding the limit
                current_total = sum(self.scores)
                games_to_add = 2  # Each match instance is 2 games
                
                # If adding these games would exceed the limit, only add what we can
                if self.max_games and current_total + games_to_add > self.max_games:
                    remaining_games = self.max_games - current_total
                    if remaining_games <= 0:
                        break  # Already at or over the limit, stop this worker
                    # Only process the first game if we can only fit one more
                    games_to_add = remaining_games
                
                games_added = 0
                white_idx, black_idx = engine_pair
                
                # Game 1: first engine (white_idx) plays white
                if games_added < games_to_add:
                    if self.is_tournament:
                        # Update pair-specific scores
                        if res1 == DRAW:
                            self.pair_scores[engine_pair][DRAW] += 1
                        elif res1 == WIN:
                            self.pair_scores[engine_pair][0] += 1  # First engine wins
                        else:
                            self.pair_scores[engine_pair][1] += 1  # Second engine wins
                        # Update time losses
                        self.pair_time_losses[engine_pair][0] += tl1
                    
                    # Update global scores for compatibility
                    if res1 == DRAW:
                        self.scores[DRAW] += 1
                        self.draw_games += 1
                    else:
                        self.scores[res1] += 1
                        if res1 == WIN:
                            self.white_wins += 1
                        else:
                            self.black_wins += 1
                    self.r.append(SCORES[res1])
                    games_added += 1
                
                # Game 2: colors swapped 
                if games_added < games_to_add:
                    if self.is_tournament:
                        # Update pair-specific scores (remember colors are swapped)
                        if res2 == DRAW:
                            self.pair_scores[engine_pair][DRAW] += 1
                        elif res2 == WIN:
                            self.pair_scores[engine_pair][1] += 1  # Second engine wins (was white)
                        else:
                            self.pair_scores[engine_pair][0] += 1  # First engine wins (was black)
                        # Update time losses
                        self.pair_time_losses[engine_pair][1] += tl2
                    
                    # Update global scores for compatibility
                    if res2 == DRAW:
                        self.scores[DRAW] += 1
                        self.draw_games += 1
                    else:
                        self.scores[1 - res2] += 1
                        if res2 == WIN:
                            self.white_wins += 1  # white won (res2 is from white's perspective)
                        else:
                            self.black_wins += 1  # black won
                    self.r.append(1 - SCORES[res2])
                    games_added += 1
                
                # Update time loss counts for non-tournament mode
                if not self.is_tournament:
                    self.time_losses[0] += tl1
                    self.time_losses[1] += tl2
                
                # Update pentanomial if we have complete pairs
                if len(self.r) >= 2:
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
                
                # Print intermediate stats if verbosity > 0
                if self.verbosity > 1:
                    self.print_results()
                elif self.verbosity > 0:
                    self.print_stats()

    def play_match_instance(self):
        """
        Play a pair of games (swapping colors between the two engines)
        and return a tuple: (result_game1, result_game2, time_loss_game1, time_loss_game2, engine_pair)
        Each result is from white's perspective (WIN, LOSS, or DRAW).
        For tournaments, randomly selects an engine pair.
        """
        # Choose a variant randomly if multiple are provided.
        variant = random.choice(self.variants)
        # Pick a position from the opening book (if available), else use "startpos"
        pos = "fen " + random.choice(self.fens) if self.fens else "startpos"

        if self.is_tournament:
            # For tournaments, randomly select an engine pair
            engine_pair = random.choice(self.engine_pairs)
            white_idx, black_idx = engine_pair
        else:
            # Traditional 2-engine mode
            white_idx, black_idx = 0, 1
            engine_pair = (0, 1)

        # Game 1: first engine plays white, second engine plays black.
        res1, tl1, _ = self.play_game_instance(variant, pos, white=white_idx, black=black_idx)
        # Game 2: swap colors.
        res2, tl2, _ = self.play_game_instance(variant, pos, white=black_idx, black=white_idx)
        return res1, res2, tl1, tl2, engine_pair

    def play_game_instance(self, variant, pos, white, black):
        """
        Play a single game between two engines.
        
        Parameters:
          variant : the chess variant to use.
          pos     : starting position (either "startpos" or a FEN string prefixed by "fen ").
          white   : index indicating which engine plays white.
          black   : index for black.
        
        Returns a tuple: (result, time_loss, move_list)
          - result: from white's perspective (WIN, LOSS, DRAW)
          - time_loss: 1 if a time loss occurred, else 0.
          - move_list: the list of moves played (for logging/debug purposes).
        """
        bestmoves = []
        wt = self.time
        bt = self.time
        # Create engine processes for the two engines playing this game.
        engines = []
        engine_indices = [white, black]
        for idx in engine_indices:
            path = self.engine_paths[idx]
            opts = self.engine_options[idx]
            engine = chess.uci.popen_engine(path)
            engine.uci()
            if self.config:
                engine.setoption({"VariantPath": self.config})
            engine.setoption({"UCI_Variant": variant})
            engine.setoption(opts)
            engines.append(engine)

        # Create custom info handler that forwards error strings to stderr
        class ErrorForwardingInfoHandler(chess.uci.InfoHandler):
            def __init__(self, engine_name):
                super(ErrorForwardingInfoHandler, self).__init__()
                self.engine_name = engine_name
            
            def string(self, string):
                # Check for error strings and forward to stderr
                if string.startswith("ERROR:"):
                    sys.stderr.write("[%s] %s\n" % (self.engine_name, string))
                    sys.stderr.flush()
                # Call parent handler to process the string normally
                super(ErrorForwardingInfoHandler, self).string(string)
        
        # Set up info handlers for the two engines with error forwarding
        info_handlers = []
        for i, engine in enumerate(engines):
            engine_idx = engine_indices[i]
            engine_name = "Engine%d(%s)" % (engine_idx + 1, os.path.basename(self.engine_paths[engine_idx]))
            handler = ErrorForwardingInfoHandler(engine_name)
            engine.info_handlers.append(handler)
            info_handlers.append(handler)
        # Start a new game for each engine.
        for engine in engines:
            engine.ucinewgame()
            engine.setoption({"clear hash": True, "UCI_Variant": variant})

        # Some variants may require a positional offset.
        offset = 0
        if pos != "startpos" and " b " in pos:
            offset = 1

        # Main game loop.
        while True:
            # Determine which engine should move based on the move number.
            engine_idx = white if (len(bestmoves) + offset) % 2 == 0 else black
            # Map engine index to the engines array (which only has 2 engines for this game)
            array_idx = 0 if engine_idx == white else 1
            engine = engines[array_idx]
            handler = info_handlers[array_idx]
            # Send the current position and moves to the engine.
            cmd = "position " + pos + " moves " + " ".join(bestmoves)
            engine.send_line(cmd)
            # Request a move with the remaining time and increment.
            bestmove, ponder = engine.go(wtime=wt, btime=bt, winc=self.inc, binc=self.inc)
            bestmoves.append(bestmove)
            with handler:
                # Check that the engine returned a score.
                if 1 in handler.info.get("score", {}):
                    # Check for end-of-game conditions.
                    if not handler.info.get("pv") and bestmove == "(none)":
                        warnings.warn("Reached final position. This might cause undefined behaviour.")
                        if handler.info["score"][1].cp == 0:
                            result = DRAW
                            break
                        elif handler.info["score"][1].mate == 0 and variant in ["giveaway", "losers"]:
                            result = WIN if engine_idx == white else LOSS
                            break
                        elif handler.info["score"][1].mate == 0:
                            result = LOSS if engine_idx == white else WIN
                            break
                        else:
                            raise Exception("Invalid game result.\nMove list: " + " ".join(bestmoves))
                    elif handler.info["score"][1].cp == 0 and handler.info.get("pv") and len(handler.info["pv"][1]) == 1:
                        result = DRAW
                        break
                    elif handler.info["score"][1].mate == 1:
                        result = WIN if engine_idx == white else LOSS
                        break
                    # Adjust the clock for the engine that just moved.
                    if engine_idx == white:
                        wt += self.inc - handler.info.get("time", 0)
                        if wt < 0:
                            tl = 1
                            result = LOSS
                            break
                    else:
                        bt += self.inc - handler.info.get("time", 0)
                        if bt < 0:
                            tl = 1
                            result = WIN
                            break
                else:
                    raise Exception("Engine does not return a score.\nMove list: " + " ".join(bestmoves))
        # Close engine processes.
        for engine in engines:
            engine.quit()
        # If no time loss was recorded in the loop, set tl to 0.
        if 'tl' not in locals():
            tl = 0
        if self.verbosity > 1:
            self.out.write(
                "Game (%s):\n" % (self.variant,) + pos + "\n" + " ".join(bestmoves) + "\n")
        return result, tl, bestmoves

    def run(self):
        """Main routine: print settings, optionally load the opening book,
        and start worker threads until the stop condition is met."""
        self.print_settings()
        # Validate that engines support all required variants
        self.validate_engine_variants()
        # If using an opening book, initialize the FEN list.
        if self.book and (len(self.variants) > 1 or not self.fens):
            self.init_book()

        # Start the worker threads.
        workers = []
        for i in range(self.threads):
            t = threading.Thread(target=self.worker)
            t.start()
            workers.append(t)
        # Wait for all workers to finish.
        for t in workers:
            t.join()

        self.print_results()
        self.close()


if __name__ == "__main__":
    match = EngineMatch()
    match.run()
