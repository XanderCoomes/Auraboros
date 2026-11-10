"""Compatibility entry point; the implementation lives in approx/snake.py."""
if __package__ in (None, ""):
    from snake import eval_main as main
else:
    from .snake import eval_main as main

# to eval on 100 games, no visual: python3 eval.py --games 100
# to eval on one game, visually: python3 eval.py --visual
# to watch game 6 of the same batch: python3 eval.py --start-game 6 --visual
if __name__ == "__main__":
    main()
