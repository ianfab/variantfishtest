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
In tournament mode, the output shows results for each engine pair. You can now choose different pairing schedulers:

```
Tournament Results:
Engine 1 vs Engine 2: Total: 20 W: 12 L: 6 D: 2 ELO: 85.3 +-45.2 LOS: 95.8%
Engine 1 vs Engine 3: Total: 20 W: 8 L: 10 D: 2 ELO: -15.7 +-42.1 LOS: 35.2%
Engine 2 vs Engine 3: Total: 20 W: 5 L: 13 D: 2 ELO: -67.4 +-43.8 LOS: 12.1%

Overall Stats:
draw rate: 10.00%
time losses engine1: 0
time losses engine2: 1
time losses engine3: 0
white wins: 25 black wins: 29 draws: 6
```

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
