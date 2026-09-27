"""Record the fixed chronological 70/20/10 split as (Date, home, away, split) keys.

The split was originally exported as train/test/validation CSVs of dataset_2012-26. Only the
game keys are kept so the split can be applied to any feature set without shipping the CSVs.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.Pregame.paths import GAME_KEY, SPLIT_KEYS

SPLITS = ["train", "test", "validation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Directory containing train.csv, test.csv, validation.csv.")
    args = parser.parse_args()

    frames = []
    for split in SPLITS:
        keys = pd.read_csv(args.split_dir / f"{split}.csv", usecols=GAME_KEY)
        frames.append(keys.assign(split=split))
    keys = pd.concat(frames, ignore_index=True)
    if keys.duplicated(GAME_KEY).any():
        raise ValueError("Duplicate game keys across split files.")

    SPLIT_KEYS.parent.mkdir(parents=True, exist_ok=True)
    keys.to_csv(SPLIT_KEYS, index=False)
    print(keys.split.value_counts().to_string())
    print(f"Wrote {SPLIT_KEYS}")


if __name__ == "__main__":
    main()
