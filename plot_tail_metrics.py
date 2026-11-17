"""Two-panel comparison (apples / 100 steps, mean apples over 100 greedy games) from compare_tail.py's summary.json.

Usage: python plot_tail_metrics.py [test_result/tail_compare_<timestamp>]   (default: latest). Writes compare_metrics.png there.
"""
import glob, json, os, sys
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4c78a8", "#f58518", "#54a24b"]
d = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("test_result/tail_compare_*"))[-1]
s = json.load(open(os.path.join(d, "summary.json")))
sets, episodes = list(s["results"]), list(next(iter(s["results"].values())))
short = {"danger,apple_dir,heading": "default", "danger,apple_dir,heading,length_bucket": "length_bucket",
         "danger,apple_dir,heading,tail": "tail"}
fig, axes = plt.subplots(2, 1, figsize=(10, 9))
rng = np.random.default_rng(0)
off = (np.arange(len(sets)) - (len(sets) - 1) / 2) * 0.25
for ax, key, yl in [(axes[0], "apples_per_100_steps_per_seed", "apples / 100 steps"),
                    (axes[1], "per_seed", "mean apples over 100 greedy games")]:
    for i, name in enumerate(sets):
        for x, e in enumerate(episodes):
            r = s["results"][name][e]
            v = np.array(r[key])
            ax.scatter(x + off[i] + rng.uniform(-0.05, 0.05, len(v)), v, s=14, alpha=0.45, color=COLORS[i])
            ax.errorbar(x + off[i], v.mean(), yerr=v.std(), fmt="o", color=COLORS[i], capsize=4, ms=7,
                        label=f"{short.get(name, name)} ({r['n_states']})" if x == 0 else None)
    ax.set_xticks(range(len(episodes)))
    ax.set_xticklabels([f"{int(e)//1000}k episodes" for e in episodes])
    ax.set_ylabel(yl)
    ax.set_title(f"{s['seeds']} seeds: dots = seeds, marker = mean, bars = std")
    ax.legend(loc="best", fontsize=9)
fig.tight_layout()
out = os.path.join(d, "compare_metrics.png")
fig.savefig(out, dpi=130)
print("saved", out)
for name in sets:
    for e in episodes:
        r = s["results"][name][e]
        a = np.array(r["apples_per_100_steps_per_seed"])
        print(f"{short.get(name,name):14s} {e:>6s} apples/100steps {a.mean():6.2f}+-{a.std():5.2f} | mean apples {r['mean']:5.2f}+-{r['std']:5.2f}")
