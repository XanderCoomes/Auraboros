# Auraboros
Reinforcement Learning for Snake

The root-level scripts implement the tabular baseline. The separate
[approx/ implementation](approx/README.md) provides linear approximate Q-learning
with continuous action features, configurable rewards, and saved-weight evaluation.
Its settings, rewards, features and agent are together in [approx/snake.py](approx/snake.py).

## Setup
```
pip install numpy matplotlib pillow
```
For the optional live viewer, also install Pygame in the same Python environment:
```
python -m pip install pygame
```

## Train
```
python train.py --size 8 --episodes 20000
```
Useful options: `--alpha`, `--gamma`, `--eps-start`, `--eps-end`, `--eps-decay-frac`, `--seed`, `--features`, `--log`, `--out`.

Outputs (in the current folder):
- `log.csv` - per-episode epsilon, score, steps, return, max_length, final_bucket (bucket where the game ended)
- `q_table.pkl` - the learned Q-table
- `q_table.json` - the training config used (including the feature set)

### Watch training and evaluation live
```
# Watch training moves and its periodic/final greedy evaluation games
python train.py --size 8 --episodes 20000 --visual --fps 30

# Train at full speed between evaluations; watch only the evaluation games
python train.py --size 8 --episodes 20000 --visual-eval --eval-every 2000 --eval-games 10

# Sample training episodes rather than animating all 20,000
python train.py --size 8 --episodes 20000 --visual --visual-every 100 --fps 60
```
The Pygame window shows the board, episode/game, score, length, steps, epsilon, rewards,
features, and recent scores. **Space** pauses/resumes; **Up/Down** changes playback speed;
**Escape** or closing the window stops the run. Stopping training saves the current Q-table,
logs completed episodes, and marks the config as interrupted (a partial episode may have learned
before you stopped). Ctrl+C also saves progress.

`--visual` slows displayed games to `--fps` steps per second (default 30, range 1..240).
`--visual-eval` keeps a periodically refreshed training preview between animated evaluations.
Without either flag, no Pygame dependency or window is needed. `--eval-every 0` disables periodic
evaluations; a final evaluation still runs. Evaluation uses a separate environment and preserves
the agent's random-generator state, so evaluation checks do not alter subsequent training.

## Evaluate a saved agent (no training)
```
# Watch the saved agent play greedily, with learning disabled
python eval.py --q q_table.pkl --games 10 --visual --fps 15

# Get evaluation statistics without graphics
python eval.py --q q_bucket.pkl --games 100 --seed 1000
```
`eval.py` reads the board size and feature set from the matching `.json`, loads the `.pkl`,
and reports mean/max apples over the requested games. It does not write models or training logs.
If the JSON is missing, supply both `--size 8` and `--features danger,apple_dir,heading` explicitly.
Overrides that disagree with saved size/features are rejected.

## Running with different features
The state the agent sees is made of pluggable discrete features. Pick them with `--features`
(comma-separated feature names; the order you type does not matter).

Available features:

| name | values | meaning |
|---|---|---|
| `danger` | 8 | wall/body one cell straight / left / right (3 bits) |
| `apple_dir` | 9 | apple row sign x column sign relative to the head |
| `heading` | 4 | current heading |
| `length_bucket` | 4 | snake length bucket (`SHORT/MEDIUM/LONG/XLONG`) |

The number of states is the product of the values, so every feature you add multiplies it
(default `danger,apple_dir,heading` = 288 states; with `length_bucket` = 1152) and learning gets slower.

### Train with a chosen feature set
```
# default features (danger, apple_dir, heading)
python train.py --size 8 --episodes 20000

# add a feature
python train.py --size 8 --episodes 20000 --features danger,apple_dir,heading,length_bucket --out q_bucket.pkl --log log_bucket.csv

# remove a feature
python train.py --size 8 --episodes 20000 --features danger,apple_dir --out q_noheading.pkl --log log_noheading.csv
```
- Give each feature set its own `--out` and `--log`, otherwise the default `q_table.pkl` / `log.csv` are overwritten.
- A Q-table only works with the feature set it was trained with. The feature list is saved in the matching `.json`,
  so `plot.py` picks the right features automatically (no `--features` needed there).
- An unknown name prints `unknown feature(s) [...]; available: [...]`.

### Compare several feature sets at once
```
python compare_features.py --size 8 --episodes 20000 --seeds 5 \
    --sets "danger,apple_dir,heading" "danger,apple_dir,heading,length_bucket" "danger,apple_dir"
```
Trains every set with the same hyperparameters and seeds, then plays greedy games. Run time grows with
(number of sets x number of seeds). Creates `test_result/compare_<timestamp>/`:
- `curves.png` - training score curves (mean over seeds, band = min..max)
- `buckets.png` - per length bucket: games reaching it, death rate, steps spent, and how games end
- `summary.json` - all numbers behind the plots

Bucket statistics are computed for every set (even without `length_bucket`), since buckets derive from snake length.
Other options: `--eval-games`, `--window`, `--alpha`, `--gamma`, `--eps-*`, `--outdir`.

### Using features in code
```python
from features import FeatureSet
from feature_library import make_feature_set
from feature_library.danger import DangerFeature
from feature_library.length_bucket import LengthBucketFeature

fs = FeatureSet([DangerFeature(), LengthBucketFeature()])   # import the classes directly
fs = make_feature_set("danger,heading")                      # or by name (what --features uses)
env = SnakeEnv(size=8, features=fs)
```
`make_feature_set` orders features as in `REGISTRY`; with `FeatureSet([...])` the order you pass is the Q-table layout.

## Adding a new feature
Each feature is a subclass of the `FeatureBase` interface (`features.py`) in its own file under `feature_library/`.

**1. Create `feature_library/<name>.py`** (example: `tail_dir.py`):
```python
from features import FeatureBase


class TailDirFeature(FeatureBase):
    name = "tail_dir"                  # the name used in --features
    cardinality = 9                    # extract() must return an int in [0, cardinality)
    description = "tail position relative to head: rowSign*3 + colSign"

    def extract(self, env):            # env is the SnakeEnv: env.snake, env.apple, env.heading, env.size, env.is_danger(h) ...
        ...
        return value

    def format_value(self, value):     # readable form for the GIF overlay
        return str(value)
```
All five members are required; a missing one raises `Can't instantiate abstract class ...` when the feature is created.

**2. Register it in `feature_library/__init__.py`:**
```python
from .tail_dir import TailDirFeature
REGISTRY = {cls.name: cls for cls in (DangerFeature, AppleDirFeature, HeadingFeature, LengthBucketFeature, TailDirFeature)}
```
Add it at the **end** of the tuple. The order fixes the Q-table layout, so inserting in the middle changes how existing Q-tables are read.

Then use it right away: `python train.py --features danger,apple_dir,heading,tail_dir`.
No other file needs to change (Q-table size, json, plots and GIF overlay follow automatically). If `extract` needs information
that `SnakeEnv` does not expose yet, add a small helper method to `snake_env.py`.

Length buckets are fractions of the board area: <10% / <25% / <50% / rest (8x8: 3-6 / 7-15 / 16-31 / 32+);
edit `BUCKET_THRESHOLDS` in `feature_library/length_bucket.py` to change them.

## Plot
```
python plot.py --log log.csv --q q_table.pkl --size 8
```
- `--size` must match the board size used in training.
- Omit `--q` to plot only the learning curves (no GIF).
- `--window` sets the moving-average window (default 200).

## Check the results
Each plot run creates a new folder `test_result/train_<timestamp>/` containing:
- `curves.png` - learning curves (score and episode steps vs. episode)
- `game.gif` - one greedy game played by the trained agent, with state/Q-values/reward overlay
- `game.json` - method, state definition, training config, greedy evaluation over 100 games, per-step log
- copies of `log.csv`, `q_table.pkl`, `q_table.json` (the next training run overwrites the originals)

Open `curves.png` and `game.gif` with any image viewer or browser.
