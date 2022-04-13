# variantfishtest

variantfishtest.py is a python script to run matches between two given UCI chess variant engines. It is mainly used for testing of [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish) for variants not supported by [cutechess](https://github.com/cutechess/cutechess).

The script is variant-agnostic and therefore supports arbitrary variants, and relies on the correctness and consistency of the engines' rule implementation. A similar script with rule-aware game adjudication is [fairyfishtest](https://github.com/ianfab/fairyfishtest), which uses the CECP/xboard protocol to run matches.

Run `python variantfishtest.py -h` for instructions on usage.

### Example
```
python variantfishtest.py stockfish1 --e1-options EvalFile=xiangqi-28d6221e2440.nnue stockfish2 --e2-options EvalFile=xiangqi-83f16c17fe26.nnue -t 10000 -i 100 -v xiangqi -b
```
