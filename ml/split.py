import pandas as pd


def split_real_instances(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    random_state: int = 42,
):
    real = metadata[
        metadata["source"] == "real"
    ].copy()

    well_counts = (
        real.groupby("well_id")
        .size()
        .sort_values(ascending=False)
    )

    # Three buckets of well IDs
    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    targets = {
        "train": len(real) * train_ratio,
        "validation": len(real) * validation_ratio,
        "test": len(real) * (
            1 - train_ratio - validation_ratio
        ),
    }

    # Deterministic ordering
    wells = list(well_counts.items())

    for well_id, count in wells:

        deficits = {
            split: targets[split] - counts[split]
            for split in splits
        }

        selected_split = max(
            deficits,
            key=deficits.get,
        )

        splits[selected_split].append(well_id)
        counts[selected_split] += count

    train = real[
        real["well_id"].isin(splits["train"])
    ]

    validation = real[
        real["well_id"].isin(splits["validation"])
    ]

    test = real[
        real["well_id"].isin(splits["test"])
    ]

    return train, validation, test