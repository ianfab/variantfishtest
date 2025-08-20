# variantfishtest

variantfishtest.py is a python script to run matches between two given UCI chess variant engines. It is mainly used for testing of [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish) for variants not supported by [cutechess](https://github.com/cutechess/cutechess).

The script is variant-agnostic and therefore supports arbitrary variants, and relies on the correctness and consistency of the engines' rule implementation. On startup, the script validates that both engines support all specified variants by checking their UCI_Variant option. A similar script with rule-aware game adjudication is [fairyfishtest](https://github.com/ianfab/fairyfishtest), which uses the CECP/xboard protocol to run matches.

Run `python variantfishtest.py -h` for instructions on usage.

### Example
```
python variantfishtest.py stockfish1 --e1-options EvalFile=xiangqi-28d6221e2440.nnue stockfish2 --e2-options EvalFile=xiangqi-83f16c17fe26.nnue -t 10000 -i 100 -v xiangqi -b
```

### Output
A typical output looks like
```
ELO: 103.73 +-71.1 (95%) LOS: 99.9%
Total: 100 W: 63 L: 34 D: 3
```
This means that 
* Engine 1 is 103.73 Elo stronger than engine 2 with a statistical uncertainty of 71.1 Elo at a 95% [confidence level](https://en.wikipedia.org/wiki/Confidence_interval).
* Its [likelihood of superiority (LOS)](https://www.chessprogramming.org/Match_Statistics#Likelihood_of_superiority) is 99.9%.
* It played 100 games, with 63 wins, 34 losses, and 3 draws.
