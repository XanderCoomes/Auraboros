"""Compatibility entry point; the implementation lives in approx/snake.py."""
if __package__ in (None, ""):
    from snake import train_main as main
else:
    from .snake import train_main as main


if __name__ == "__main__":
    main()
