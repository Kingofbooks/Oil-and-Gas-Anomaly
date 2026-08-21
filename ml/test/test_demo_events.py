from pathlib import Path

import pandas as pd


DATASET = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
    r"\datasets\demo\oil_gas_demo_22features.csv"
)

GROUND_TRUTH_COLUMN = "class"


def main():

    print("=" * 80)
    print("DEMO GROUND-TRUTH EVENTS")
    print("=" * 80)

    df = pd.read_csv(DATASET)

    print(f"Rows: {len(df)}")
    print(f"Ground-truth column: {GROUND_TRUTH_COLUMN}")

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    if GROUND_TRUTH_COLUMN not in df.columns:

        print()
        print("ERROR: Ground-truth column not found.")
        print("Available columns:")
        print(df.columns.tolist())

        return

    # ---------------------------------------------------------
    # GROUND TRUTH
    # ---------------------------------------------------------

    ground_truth = (
        pd.to_numeric(
            df[GROUND_TRUTH_COLUMN],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 80)
    print(ground_truth.value_counts().sort_index())

    anomaly = ground_truth.eq(1)

    # ---------------------------------------------------------
    # FIND CONTIGUOUS ANOMALY EVENTS
    # ---------------------------------------------------------

    groups = anomaly.ne(anomaly.shift()).cumsum()

    events = []

    for _, group in df.groupby(groups):

        group_truth = ground_truth.loc[group.index]

        if int(group_truth.iloc[0]) != 1:
            continue

        events.append(
            {
                "start_row": group.index.min(),
                "end_row": group.index.max(),
                "rows": len(group),
                "start_time": group["timestamp"].iloc[0],
                "end_time": group["timestamp"].iloc[-1],
            }
        )

    # ---------------------------------------------------------
    # PRINT EVENTS
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("GROUND-TRUTH ANOMALY EVENTS")
    print("=" * 80)

    if not events:

        print("No anomaly events found.")

    else:

        for i, event in enumerate(events, 1):

            print(
                f"Event {i:2d}: "
                f"rows {event['start_row']:5d} → "
                f"{event['end_row']:5d} | "
                f"rows={event['rows']:4d} | "
                f"{event['start_time']} → "
                f"{event['end_time']}"
            )

    print()
    print("=" * 80)
    print(f"Total ground-truth events: {len(events)}")
    print("=" * 80)


if __name__ == "__main__":
    main()