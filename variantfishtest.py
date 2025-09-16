#!/usr/bin/env python
import sys
import os
import random
import warnings
import argparse
import threading
import logging
import math
from collections import defaultdict

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
        self.parser.add_argument("--scheduler", help="pairing scheduler for tournaments", 
                                 choices=["roundrobin", "random", "copeland_ucb", "borda_ucb"], 
                                 default="random")
        # Engine aliases for clearer tournament output
        self.parser.add_argument("--alias", help="alias for an engine (format: engine_index:alias). Repeatable.",
                                 type=lambda kv: kv.split(":", 1), action='append', default=[])
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
        # Initialize aliases (None by default → falls back to "Engine N")
        engine_aliases = [None for _ in range(self.num_engines)]
        
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
        
        # Parse aliases (format: index:alias) — indices are 1-based like engine numbering
        for alias_spec in getattr(self, 'alias', []) or []:
            if len(alias_spec) == 2:
                try:
                    idx_str, alias_value = alias_spec
                    idx = int(idx_str) - 1
                    if 0 <= idx < self.num_engines:
                        # Normalize empty strings to None
                        alias_value = alias_value.strip()
                        engine_aliases[idx] = alias_value if alias_value else None
                    else:
                        print(f"Warning: Invalid alias engine index {idx + 1}, ignoring alias")
                except (ValueError, IndexError):
                    print(f"Warning: Invalid alias format: {':'.join(alias_spec)}")

        self.engine_options = engine_opts
        self.engine_aliases = engine_aliases
        
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
            
            # New data structures for adaptive scheduling
            K = self.num_engines
            self.in_flight = defaultdict(int)  # key: (i,j) with i<j, value: ongoing matches count
            self.wins = [[0] * K for _ in range(K)]   # wins[i][j] = wins of i vs j (directed)
            self.draws = [[0] * K for _ in range(K)]  # draws[i][j] = draws in games where i faced j
            self.games = [[0] * K for _ in range(K)]  # games[i][j] = total games where i faced j
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
                name = self._engine_label(engine_idx)
                self.out.write(f"Validating {name}: {engine_path}\n")
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
                alias = self.engine_aliases[i]
                if alias:
                    self.out.write("engine%d:    %s (alias: %s)\n" % (i + 1, path, alias))
                else:
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
        if self.is_tournament:
            # For tournaments, show current leader and total games instead of aggregate ELO
            total_games = sum(self.scores)
            self.out.write("Total games: %d " % total_games)
            self._print_current_leader_brief()
        else:
            # For 1v1 matches, show traditional stats
            self.out.write(print_scores(self.scores) + " ")
            if self.sprt:
                self.out.write(sprt_stats(self.scores, self.elo0, self.elo1))
            else:
                self.out.write(elo_stats(self.scores))
        self.out.flush()

    def _print_current_leader_brief(self):
        """Print just the current leader for intermediate tournament updates."""
        if not self.is_tournament:
            return
        
        win_rates, total_games = self._calculate_engine_win_rates()
        
        # Find leader(s)
        if win_rates:
            max_rate = max(win_rates.values())
            leaders = [i for i, rate in win_rates.items() if abs(rate - max_rate) < 1e-9]
            
            if len(leaders) == 1:
                leader = leaders[0]
                leader_name = self._engine_label(leader)
                self.out.write("- Current Leader: %s (%.3f = %.1f%%, %d games)\n" %
                               (leader_name, win_rates[leader], win_rates[leader] * 100, total_games[leader]))
            else:
                leader_names = [self._engine_label(i) for i in leaders]
                self.out.write("- Tied Leaders: %s (%.3f = %.1f%%)\n" %
                               (', '.join(leader_names), max_rate, max_rate * 100))

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
                    name_a = self._engine_label(pair[0])
                    name_b = self._engine_label(pair[1])
                    self.out.write("%s vs %s: " % (name_a, name_b))
                    self.out.write("Total: %d W: %d L: %d D: %d " % (total_games, pair_scores[0], pair_scores[1], pair_scores[2]))
                    try:
                        elo, elo95, los = stat_util.get_elo(pair_scores)
                        self.out.write("ELO: %.2f +-%.1f LOS: %.1f%%\n" % (elo, elo95, 100 * los))
                    except (ValueError, ZeroDivisionError):
                        self.out.write("\n")
            
            # Calculate and display tournament leader
            self._print_tournament_leader()
            
            # Compute and display relative Elo ratings per engine (Cutechess-style)
            ratings = self._compute_tournament_ratings()
            if ratings:
                self.out.write("\nRatings (Elo, Error, Games, Score):\n")
                ratings_sorted = sorted(
                    ratings,
                    key=lambda r: ((r['elo'] if r['elo'] is not None else float('-inf')), (r['score'] if r['score'] is not None else -1.0)),
                    reverse=True,
                )
                for rank, r in enumerate(ratings_sorted, start=1):
                    name = r['name']
                    games = r['games']
                    score_pct = 100.0 * r['score'] if r['score'] is not None else 0.0
                    if r['elo'] is not None and r['error'] is not None:
                        self.out.write("%2d. %-25s Elo: %7.0f  Error: %4.0f  Games: %4d  Score: %5.1f%%\n" %
                                       (rank, name, r['elo'], r['error'], games, score_pct))
                    else:
                        self.out.write("%2d. %-25s Elo:    n/a  Error:  n/a  Games: %4d  Score: %5.1f%%\n" %
                                       (rank, name, games, score_pct))
            
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
                name_i = self._engine_label(i)
                self.out.write("time losses %s: %d\n" % (name_i, total_tl))
        else:
            self.out.write("time losses engine1: %d\n" % (self.time_losses[0]))
            self.out.write("time losses engine2: %d\n" % (self.time_losses[1]))
            
        # colour balance and pentanomial
        self.out.write("white wins: %d black wins: %d draws: %d\n" % (self.white_wins, self.black_wins, self.draw_games))
        self.out.write("pentanomial [LL LD DD/WL WD WW]: [%s]\n" % (",".join(str(x) for x in self.pentanomial)))
        self.out.write("\n")
        
        # For tournaments, don't print aggregate ELO stats as they don't make sense
        if not self.is_tournament:
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

    def _print_tournament_leader(self):
        """Calculate and print current tournament leader with confidence statistics."""
        if not self.is_tournament:
            return
        
        win_rates, total_games = self._calculate_engine_win_rates()
        
        # Find leader(s)
        if win_rates:
            max_rate = max(win_rates.values())
            leaders = [i for i, rate in win_rates.items() if abs(rate - max_rate) < 1e-9]
            
            self.out.write(f"\nTotal games played: {sum(total_games.values())}\n")
            if len(leaders) == 1:
                leader = leaders[0]
                leader_name = self._engine_label(leader)
                self.out.write(f"Current Leader: {leader_name} ({win_rates[leader]:.3f} = {win_rates[leader]*100:.1f}%, {total_games[leader]} games)\n")
            else:
                leader_names = [self._engine_label(i) for i in leaders]
                self.out.write(f"Tied Leaders: {', '.join(leader_names)} ({max_rate:.3f} = {max_rate*100:.1f}%)\n")
            
            # Display confidence intervals for all engines
            self.out.write("Confidence (95% Wilson intervals):\n")
            for i in sorted(win_rates.keys()):
                games = total_games[i]
                if games > 0:
                    rate = win_rates[i]
                    # Wilson score interval
                    z = 1.96
                    denom = 1 + z * z / games
                    center = (rate + z * z / (2 * games)) / denom
                    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denom
                    lower = max(0, center - margin)
                    upper = min(1, center + margin)
                    name_i = self._engine_label(i)
                    self.out.write(f"  {name_i}: {rate:.3f} [{lower:.3f}, {upper:.3f}] ({games} games)\n")
                else:
                    name_i = self._engine_label(i)
                    self.out.write(f"  {name_i}: no games yet\n")

    def _calculate_engine_win_rates(self):
        """Calculate win rates and total games for each engine. Returns (win_rates, total_games) dicts."""
        win_rates = {}
        total_games = {}
        
        for i in range(self.num_engines):
            wins = 0
            games = 0
            for pair in self.engine_pairs:
                if pair[0] == i:
                    # Engine i is first in pair
                    pair_scores = self.pair_scores[pair]
                    wins += pair_scores[0]  # wins for first engine
                    wins += pair_scores[2] * 0.5  # half points for draws
                    games += sum(pair_scores)
                elif pair[1] == i:
                    # Engine i is second in pair
                    pair_scores = self.pair_scores[pair]
                    wins += pair_scores[1]  # wins for second engine
                    wins += pair_scores[2] * 0.5  # half points for draws
                    games += sum(pair_scores)
            
            if games > 0:
                win_rates[i] = wins / games
                total_games[i] = games
            else:
                win_rates[i] = 0.5
                total_games[i] = 0
                
        return win_rates, total_games

    def _compute_tournament_ratings(self):
        """Compute Cutechess-style Elo and error per engine from aggregated W/L/D.
        Elo is computed from each engine's global score fraction (wins + 0.5*draws)/games,
        with error as the 95% half-interval. Returns a list of dicts per engine.
        """
        if not self.is_tournament:
            return []
        wins = [0] * self.num_engines
        losses = [0] * self.num_engines
        draws = [0] * self.num_engines
        for (i, j), ps in self.pair_scores.items():
            w_i, w_j, d = ps[0], ps[1], ps[2]
            wins[i] += w_i
            losses[i] += w_j
            draws[i] += d
            wins[j] += w_j
            losses[j] += w_i
            draws[j] += d
        ratings = []
        for i in range(self.num_engines):
            g = wins[i] + losses[i] + draws[i]
            name = self._engine_label(i)
            if g > 0:
                try:
                    elo, elo95, _ = stat_util.get_elo([wins[i], losses[i], draws[i]])
                    score = (wins[i] + 0.5 * draws[i]) / g
                    ratings.append({
                        'index': i,
                        'name': name,
                        'elo': elo,
                        'error': elo95,
                        'games': g,
                        'score': score,
                        'wins': wins[i],
                        'losses': losses[i],
                        'draws': draws[i],
                    })
                except (ValueError, ZeroDivisionError):
                    ratings.append({
                        'index': i,
                        'name': name,
                        'elo': None,
                        'error': None,
                        'games': g,
                        'score': None,
                        'wins': wins[i],
                        'losses': losses[i],
                        'draws': draws[i],
                    })
            else:
                ratings.append({
                    'index': i,
                    'name': name,
                    'elo': None,
                    'error': None,
                    'games': 0,
                    'score': None,
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                })
        return ratings

    def select_pair(self):
        """
        Select an engine pair (i, j) with i < j based on the configured scheduler.
        Returns tuple (i, j) representing the engine indices to play against each other.
        """
        if not self.is_tournament or self.scheduler == "random":
            # Default behavior: random selection
            return random.choice(self.engine_pairs)
        
        K = self.num_engines
        pairs = self.engine_pairs
        
        # Take a snapshot of current state (no lock needed for reading)
        wins_snapshot = [row[:] for row in self.wins]
        draws_snapshot = [row[:] for row in self.draws] 
        games_snapshot = [row[:] for row in self.games]
        in_flight_snapshot = dict(self.in_flight)
        
        if self.scheduler == "roundrobin":
            # Simple round-robin: pick pair with fewest games
            min_games = float('inf')
            best_pairs = []
            for i, j in pairs:
                total_games = games_snapshot[i][j] + games_snapshot[j][i]
                if total_games < min_games:
                    min_games = total_games
                    best_pairs = [(i, j)]
                elif total_games == min_games:
                    best_pairs.append((i, j))
            return random.choice(best_pairs)
        
        # For UCB algorithms, we need to compute probabilities and confidence intervals
        def beta_confidence_interval(wins, total, alpha=0.01):
            """Compute Beta posterior confidence interval using Jeffreys prior Beta(0.5, 0.5)"""
            if total == 0:
                return 0.5, 0.0, 1.0  # No data: p_hat=0.5, wide interval
            
            # Jeffreys prior: Beta(0.5, 0.5)
            alpha_post = wins + 0.5
            beta_post = (total - wins) + 0.5
            
            p_hat = alpha_post / (alpha_post + beta_post)
            
            # For confidence interval, we'll use a simple approximation
            # since we don't have scipy.stats.beta available
            # Wilson score interval approximation
            z = 1.96  # 95% confidence (alpha=0.05)
            if total < 5:
                # Wide interval for small sample sizes
                margin = 0.5
            else:
                p = wins / total
                margin = z * math.sqrt(p * (1 - p) / total)
                margin = min(margin, 0.5)  # Cap the margin
            
            lower = max(0.0, p_hat - margin)
            upper = min(1.0, p_hat + margin)
            
            return p_hat, lower, upper
        
        # Compute probability estimates with virtual visits
        lambda_vv = 0.5  # Virtual visits weight (reduced from 1.0 based on UCB literature)
        prob_matrix = {}  # prob_matrix[(i,j)] = (p_hat, lower, upper) for i beating j
        
        for i in range(K):
            for j in range(K):
                if i != j:
                    # Include virtual visits for ongoing matches of this unordered pair
                    pair_key = (min(i, j), max(i, j))
                    vv_count = lambda_vv * in_flight_snapshot.get(pair_key, 0)
                    
                    w_ij = wins_snapshot[i][j] + 0.5 * draws_snapshot[i][j]  # Draws count as 0.5 wins
                    n_ij = games_snapshot[i][j]
                    n_eff = n_ij + vv_count
                    
                    prob_matrix[(i, j)] = beta_confidence_interval(w_ij, n_eff)
        
        if self.scheduler == "copeland_ucb":
            return self._select_copeland_ucb(prob_matrix, K, pairs)
        elif self.scheduler == "borda_ucb":
            return self._select_borda_ucb(prob_matrix, K, pairs)
        else:
            # Fallback to random
            return random.choice(pairs)
    
    def _select_copeland_ucb(self, prob_matrix, K, pairs):
        """Copeland-UCB: Find optimistic leader and most ambiguous opponent"""
        
        # ε-greedy exploration: 10% pure uncertainty sampling (Auer et al. 2002)
        if random.random() < 0.1:
            # Pure uncertainty sampling: select pair with largest confidence interval
            best_width = -1.0
            best_pairs = []
            for pair_key in pairs:
                i, j = pair_key
                _, lower, upper = prob_matrix.get((i, j), (0.5, 0.0, 1.0))
                width = upper - lower
                if width > best_width:
                    best_width = width
                    best_pairs = [pair_key]
                elif abs(width - best_width) < 1e-9:
                    best_pairs.append(pair_key)
            if best_pairs:
                return random.choice(best_pairs)
        
        # Compute optimistic Copeland scores
        copeland_upper = [0] * K
        for i in range(K):
            for j in range(K):
                if i != j:
                    _, _, upper = prob_matrix.get((i, j), (0.5, 0.0, 1.0))
                    if upper >= 0.5:
                        copeland_upper[i] += 1
        
        # Find optimistic leader
        max_copeland = max(copeland_upper)
        leaders = [i for i in range(K) if copeland_upper[i] == max_copeland]
        
        # Small probability of exploring near-leaders to prevent tunnel vision
        if random.random() < 0.05 and max_copeland > 0:
            near_leaders = [i for i in range(K) if copeland_upper[i] >= max_copeland - 1]
            if near_leaders:
                leaders = near_leaders
        
        leader = random.choice(leaders)
        
        # Find most ambiguous opponent for the leader
        best_score = float('inf')
        best_pairs = []
        
        for j in range(K):
            if j != leader:
                pair_key = (min(leader, j), max(leader, j))
                if pair_key in pairs:
                    p_hat, lower, upper = prob_matrix.get((leader, j), (0.5, 0.0, 1.0))
                    
                    # Ambiguity score: small gap (close to 0.5) and large width
                    gap = abs(0.5 - p_hat)
                    width = upper - lower
                    score = gap - 0.5 * width  # k=0.5 from literature, smaller is better
                    
                    if score < best_score:
                        best_score = score
                        best_pairs = [pair_key]
                    elif abs(score - best_score) < 1e-9:
                        best_pairs.append(pair_key)
        
        return random.choice(best_pairs) if best_pairs else random.choice(pairs)
    
    def _select_borda_ucb(self, prob_matrix, K, pairs):
        """Borda-UCB: Use average win probability across all opponents"""
        
        # Compute optimistic Borda scores
        borda_upper = [0.0] * K
        for i in range(K):
            total_upper = 0.0
            for j in range(K):
                if i != j:
                    _, _, upper = prob_matrix.get((i, j), (0.5, 0.0, 1.0))
                    total_upper += upper
            borda_upper[i] = total_upper / (K - 1) if K > 1 else 0.0
        
        # Find optimistic leader
        max_borda = max(borda_upper)
        leaders = [i for i in range(K) if abs(borda_upper[i] - max_borda) < 1e-9]
        
        # Small probability of exploring near-leaders
        if random.random() < 0.05:
            near_leaders = [i for i in range(K) if borda_upper[i] >= max_borda - 0.05]
            if near_leaders:
                leaders = near_leaders
        
        leader = random.choice(leaders)
        
        # Find opponent that most reduces uncertainty in leader's Borda score
        # Choose pair with largest confidence interval width
        best_width = -1.0
        best_pairs = []
        
        for j in range(K):
            if j != leader:
                pair_key = (min(leader, j), max(leader, j))
                if pair_key in pairs:
                    _, lower, upper = prob_matrix.get((leader, j), (0.5, 0.0, 1.0))
                    width = upper - lower
                    
                    if width > best_width:
                        best_width = width
                        best_pairs = [pair_key]
                    elif abs(width - best_width) < 1e-9:
                        best_pairs.append(pair_key)
        
        return random.choice(best_pairs) if best_pairs else random.choice(pairs)

    def worker(self):
        """Worker thread: play match instances until the global stop condition is met."""
        while True:
            with self.lock:
                if self.stop():
                    break
            
            # Select pair intelligently and manage in_flight counter
            if self.is_tournament:
                selected_pair = self.select_pair()
                with self.lock:
                    self.in_flight[selected_pair] += 1
            else:
                selected_pair = None
                
            try:
                # Play one match instance (two games with color swap)
                if self.is_tournament:
                    res1, res2, tl1, tl2, engine_pair = self.play_match_instance(forced_pair=selected_pair)
                else:
                    res1, res2, tl1, tl2, engine_pair = self.play_match_instance()
            except Exception as e:
                # Decrement in_flight counter on error
                if self.is_tournament:
                    with self.lock:
                        self.in_flight[selected_pair] -= 1
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
                        
                        # Update directed statistics for adaptive scheduling
                        i, j = white_idx, black_idx  # Game 1: white=i, black=j
                        if res1 == DRAW:
                            self.draws[i][j] += 1
                            self.draws[j][i] += 1
                        elif res1 == WIN:
                            self.wins[i][j] += 1  # white (i) won
                        else:  # res1 == LOSS
                            self.wins[j][i] += 1  # black (j) won
                        self.games[i][j] += 1
                        self.games[j][i] += 1
                    
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
                        
                        # Update directed statistics for adaptive scheduling
                        # Game 2: colors swapped, but engine indices remain consistent
                        white_engine = black_idx  # black_idx from Game 1 is now white in Game 2
                        black_engine = white_idx  # white_idx from Game 1 is now black in Game 2
                        
                        if res2 == DRAW:
                            self.draws[white_engine][black_engine] += 1
                            self.draws[black_engine][white_engine] += 1
                        elif res2 == WIN:
                            self.wins[white_engine][black_engine] += 1  # white engine won
                        else:  # res2 == LOSS
                            self.wins[black_engine][white_engine] += 1  # black engine won
                        self.games[white_engine][black_engine] += 1
                        self.games[black_engine][white_engine] += 1
                    
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
                
                # Decrement in_flight counter for tournament mode
                if self.is_tournament and selected_pair:
                    self.in_flight[selected_pair] -= 1
                
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

    def play_match_instance(self, forced_pair=None):
        """
        Play a pair of games (swapping colors between the two engines)
        and return a tuple: (result_game1, result_game2, time_loss_game1, time_loss_game2, engine_pair)
        Each result is from white's perspective (WIN, LOSS, or DRAW).
        For tournaments, uses forced_pair if provided, otherwise randomly selects an engine pair.
        """
        # Choose a variant randomly if multiple are provided.
        variant = random.choice(self.variants)
        # Pick a position from the opening book (if available), else use "startpos"
        pos = "fen " + random.choice(self.fens) if self.fens else "startpos"

        if self.is_tournament:
            # For tournaments, use forced pair if provided, otherwise random selection
            if forced_pair:
                engine_pair = forced_pair
            else:
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
            alias = self._engine_label(engine_idx)
            engine_name = "%s(%s)" % (alias, os.path.basename(self.engine_paths[engine_idx]))
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
            if self.is_tournament:
                name_w = self._engine_label(white)
                name_b = self._engine_label(black)
                self.out.write(
                    "Game (%s, %s vs %s):\n" % (variant, name_w, name_b) + pos + "\n" + " ".join(bestmoves) + "\n")
            else:
                self.out.write(
                    "Game (%s):\n" % (variant,) + pos + "\n" + " ".join(bestmoves) + "\n")
        return result, tl, bestmoves

    def _engine_label(self, idx):
        """Return display label for engine index respecting aliases.
        If alias not provided, fall back to 'Engine N'.
        """
        if 0 <= idx < len(self.engine_aliases) and self.engine_aliases[idx]:
            return self.engine_aliases[idx]
        return f"Engine {idx + 1}"

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
