"""Watch one game in the terminal.

    python watch.py runs/default/best.pt               # trained agent
    python watch.py runs/default/best.pt --show-q      # also print Q-values each step
    python watch.py --random                           # random safe-masked policy (no checkpoint)
    python watch.py runs/default/best.pt --seed 10003 --delay 0.1

Board legend: the head is an arrow pointing the way the snake is moving (▲ ▶ ▼ ◀), the body
is drawn as one connected line from head to tail (which ends in a thin stub), * is the apple.
With colour on, the body fades from bright (head) to dark (tail); --no-color turns it off.
With --show-q, disallowed actions are marked "x" and the chosen action with "<".
"""
import argparse
import time

import numpy as np
import torch

from evaluate import env_kwargs_from, load, pick_device
from snake_env import SnakeEnv

ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")

# Box-drawing glyph for a body cell, keyed by the directions of its neighbours in the snake
# (U/D/L/R). A cell with one neighbour (the tail) gets a half-line stub pointing at it.
SEGMENT = {
    frozenset("LR"): "─", frozenset("UD"): "│",
    frozenset("DR"): "┌", frozenset("DL"): "┐",
    frozenset("UR"): "└", frozenset("UL"): "┘",
    frozenset("R"): "╶", frozenset("L"): "╴", frozenset("U"): "╵", frozenset("D"): "╷",
}
HEAD = {"U": "▲", "R": "▶", "D": "▼", "L": "◀"}
GREENS = (46, 40, 34, 28, 22)  # 256-colour greens, head (bright) -> tail (dark)


def _dir(a, b):
    """Direction letter of the step from cell a to the adjacent cell b."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    return {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}[(dr, dc)]


def render_directional(env, color=True):
    """Draw the board so you can see which way the snake is going.

    Consecutive body cells are joined with line segments (including the gap between two
    horizontally adjacent cells), so a body folded against itself still reads as one path.
    """
    n, body = env.n, list(env.body)
    L = len(body)
    paint = (lambda s, code: f"\033[{code}m{s}\033[0m") if color else (lambda s, code: s)

    def shade(i):  # colour for body index i (0 = head)
        k = round(i * (len(GREENS) - 1) / max(L - 1, 1))
        return f"38;5;{GREENS[k]}"

    cells = [[paint(".", "2") for _ in range(n)] for _ in range(n)]
    gaps = [[" "] * (n - 1) for _ in range(n)]  # gaps[r][c] sits between (r, c) and (r, c+1)

    for i, cell in enumerate(body):
        r, c = cell
        nbrs = [body[j] for j in (i - 1, i + 1) if 0 <= j < L]
        if i == 0:
            glyph = HEAD[_dir(body[1], cell)] if L > 1 else "@"
            cells[r][c] = paint(glyph, "1;" + shade(0))
        else:
            glyph = SEGMENT[frozenset(_dir(cell, nb) for nb in nbrs)]
            cells[r][c] = paint(glyph, shade(i))
        if i + 1 < L:  # connector to the next segment if it sits beside us on the same row
            nr, nc = body[i + 1]
            if nr == r and abs(nc - c) == 1:
                gaps[r][min(c, nc)] = paint("─", shade(i))

    if env.apple is not None:
        ar, ac = env.apple
        cells[ar][ac] = paint("*", "1;31")

    rows = []
    for r in range(n):
        line = cells[r][0]
        for c in range(1, n):
            line += gaps[r][c - 1] + cells[r][c]
        rows.append(line)
    return "\n".join(rows) + f"\nlen={L} steps={env.steps} hunger={env.hunger}"


def q_table(q, mask, chosen):
    """A few lines showing Q(s, a) per action, which are masked, and which was taken."""
    lines = []
    for a, name in enumerate(ACTION_NAMES):
        val = f"{q[a]:8.3f}" if q is not None else "       -"
        flag = "  x" if not mask[a] else ("  <" if a == chosen else "")
        lines.append(f"  {name:5s} {val}{flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--random", action="store_true", help="random allowed moves instead of an agent")
    ap.add_argument("--seed", type=int, default=10_000, help="game seed (sets the apple sequence)")
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between frames")
    ap.add_argument("--show-q", action="store_true")
    ap.add_argument("--no-color", action="store_true", help="plain text board (no ANSI colours)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    if args.random:
        net, env = None, SnakeEnv(seed=args.seed)
    else:
        assert args.checkpoint, "give a checkpoint path, or use --random"
        net, a = load(args.checkpoint, device)
        net.eval()
        env = SnakeEnv(seed=args.seed, **env_kwargs_from(a))
    rng = np.random.default_rng(args.seed)

    while not env.done:
        mask = env.mask
        q = None
        if net is None:
            action = int(rng.choice(np.flatnonzero(mask)))
        else:
            with torch.no_grad():
                q = net(torch.from_numpy(env.obs()[None]).to(device))[0].cpu().numpy()
            action = int(np.where(mask, q, -np.inf).argmax())
        frame = render_directional(env, color=not args.no_color)
        if args.show_q:
            frame += "\n" + q_table(q, mask, action)
        print("\033[H\033[J" + frame)  # clear the terminal, then draw
        env.step(action)
        time.sleep(args.delay)

    print("\033[H\033[J" + render_directional(env, color=not args.no_color))
    print(f"result: {env.reason} | apples {env.apples} | steps {env.steps}")
