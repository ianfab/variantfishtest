# variantfishtest

variantfishtest.py is a python script to run matches between UCI chess variant engines. It supports both traditional head-to-head matches between two engines and tournament mode with multiple engines. It is mainly used for testing of [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish) for variants not supported by [cutechess](https://github.com/cutechess/cutechess).

The script is variant-agnostic and therefore supports arbitrary variants, and relies on the correctness and consistency of the engines' rule implementation. On startup, the script validates that all engines support all specified variants by checking their UCI_Variant option. A similar script with rule-aware game adjudication is [fairyfishtest](https://github.com/ianfab/fairyfishtest), which uses the CECP/xboard protocol to run matches.

Run `python variantfishtest.py -h` for instructions on usage.

## Modes

### Two-Engine Mode (Traditional)
When exactly 2 engines are specified, the script runs in traditional mode with head-to-head matches between the two engines.

### Tournament Mode
When 3 or more engines are specified, the script automatically enters tournament mode and runs a round-robin tournament where each engine plays against every other engine. Games are still played in pairs with color swapping to ensure fairness.

## Examples

### Traditional Two-Engine Match
```
python variantfishtest.py stockfish1 stockfish2 --e1-options EvalFile=xiangqi-28d6221e2440.nnue --e2-options EvalFile=xiangqi-83f16c17fe26.nnue -t 10000 -i 100 -v xiangqi -b
```

### Tournament with Multiple Engines
```
python variantfishtest.py engine1 engine2 engine3 engine4 --engine-options 1:Hash=128 --engine-options 2:Threads=2 --engine-options 3:Hash=256 --engine-options 4:Threads=4 -t 5000 -i 50 -v chess -n 100
```

### Engine Aliases (Tournament or 1v1)
To make results more readable, you can assign aliases to engines. These aliases are used in intermediate stats and final tournament results instead of generic labels like "Engine 1".

```
python variantfishtest.py /path/to/stockfish /path/to/devbuild /path/to/nnue \
  --alias 1:SF15 --alias 2:Dev --alias 3:NNUE \
  --scheduler roundrobin -n 60 -v chess
```

You can repeat `--alias` for each engine using 1-based indices.

### Engine Options
- **New format**: `--engine-options N:option=value` where N is the engine number (1-based)
- **Legacy format**: `--e1-options option=value` and `--e2-options option=value` (still supported for backward compatibility)

## Output

### Two-Engine Mode Output
A typical output looks like:
```
ELO: 103.73 +-71.1 (95%) LOS: 99.9%
Total: 100 W: 63 L: 34 D: 3
```
This means that:
* Engine 1 is 103.73 Elo stronger than engine 2 with a statistical uncertainty of 71.1 Elo at a 95% [confidence level](https://en.wikipedia.org/wiki/Confidence_interval).
* Its [likelihood of superiority (LOS)](https://www.chessprogramming.org/Match_Statistics#Likelihood_of_superiority) is 99.9%.
* It played 100 games, with 63 wins, 34 losses, and 3 draws.

### Tournament Mode Output
In tournament mode, the output shows results for each engine pair. When aliases are provided, they are shown in place of default engine numbers. You can also choose different pairing schedulers:

```
Tournament Results:
SF15 vs Dev: Total: 20 W: 12 L: 6 D: 2 ELO: 85.3 +-45.2 LOS: 95.8%
SF15 vs NNUE: Total: 20 W: 8 L: 10 D: 2 ELO: -15.7 +-42.1 LOS: 35.2%
Dev vs NNUE: Total: 20 W: 5 L: 13 D: 2 ELO: -67.4 +-43.8 LOS: 12.1%

Total games played: 60
Current Leader: SF15 (0.633 = 63.3%, 60 games)
Confidence (95% Wilson intervals):
  SF15: 0.633 [0.509, 0.743] (60 games)
  Dev:  0.450 [0.329, 0.576] (60 games)
  NNUE: 0.417 [0.298, 0.548] (60 games)

Ratings (Elo, Error, Games, Score):
 1. SF15                      Elo:     87  Error:   41  Games:   60  Score:  63.3%
 2. Dev                       Elo:    -35  Error:   45  Games:   60  Score:  45.0%
 3. NNUE                      Elo:    -52  Error:   46  Games:   60  Score:  41.7%

Overall Stats:
draw rate: 10.00%
time losses SF15: 0
time losses Dev: 1
time losses NNUE: 0
white wins: 25 black wins: 29 draws: 6
```

#### Ratings Table (Cutechess‑style Relative Elo)
- The ratings table summarizes each engine’s relative Elo and 95% error, computed from its overall W/L/D across the tournament.
- This mirrors cutechess’s “Elo” column: it treats a player’s results as if versus a single opponent and derives Elo from the overall score fraction (wins + 0.5*draws)/games.
- It is a quick, comparable indicator for ranking; it is not a full multi‑player rating system.

### Pairing Schedulers

You can control how engine pairs are selected during tournaments using the `--scheduler` option:

- `random` (default): Random pair selection, maintains backward compatibility
- `roundrobin`: Ensures equal number of games between all pairs
- `copeland_ucb`: Adaptive scheduler using Copeland scoring with Upper Confidence Bounds
- `borda_ucb`: Adaptive scheduler using Borda scoring with Upper Confidence Bounds

Example:
```bash
python variantfishtest.py engine1 engine2 engine3 --scheduler copeland_ucb
```

The UCB schedulers are designed to:
- Find the overall best engine efficiently, even with cycles (A > B > C > A)
- Soft-exclude weak engines (selected rarely, never completely dropped)
- Focus games on decisive matches that reduce ranking uncertainty
- Work seamlessly with multithreading using virtual visits
