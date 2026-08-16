from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class InstanceMetadata:
    path: Path
    well_id: str | None
    folder_type: int
    source: str
    num_rows: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class ThreeWDataset:

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_instances(self) -> list[Path]:
        return sorted(self.root.glob("*/*.parquet"))

    def get_metadata(self, path: Path) -> InstanceMetadata:
        folder_type = int(path.parent.name)

        if path.name.startswith("SIMULATED_"):
            source = "simulated"
            well_id = None
        else:
            source = "real"
            well_id = path.stem.split("_")[0]

        df = pd.read_parquet(path)

        return InstanceMetadata(
            path=path,
            well_id=well_id,
            folder_type=folder_type,
            source=source,
            num_rows=len(df),
            start_time=df.index[0],
            end_time=df.index[-1],
        )
    
    def get_well_id(self, path: Path) -> str:
        if path.name.startswith("SIMULATED_"):
            return path.stem

        return path.stem.split("_")[0]
    
    def build_index(self) -> list[InstanceMetadata]:
        instances = self.list_instances()

        metadata = []

        for path in instances:
            metadata.append(self.get_metadata(path))

        return metadata

    def get_metadata(self, path: Path) -> InstanceMetadata:
        event_class = int(path.parent.name)

        if path.name.startswith("SIMULATED_"):
            source = "simulated"
        else:
            source = "real"

        well_id = self.get_well_id(path)

        df = pd.read_parquet(path)

        return InstanceMetadata(
            path=path,
            well_id=well_id,
            folder_type=event_class,
            source=source,
            num_rows=len(df),
            start_time=df.index[0],
            end_time=df.index[-1],
        )