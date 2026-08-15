from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)
class SensorDataGenerator:
    
    def __init__(self, n_rows=500, random_state=42):
        self.n_rows = n_rows
        self.rng = np.random.default_rng(random_state)
        
        self.phase_lengths = {
            "NORMAL": int(n_rows * 0.60),  
            "ANOMALY_BUILDUP": int(n_rows * 0.10),  
            "ANOMALY": int(n_rows * 0.10),   
            "RECOVERY": int(n_rows * 0.20),  
        }
        
    def generate_data(self)-> pd.DataFrame:
        timestamps=pd.date_range(
            start="2026-03-31 15:15:00", periods=self.n_rows,freq="10s"
        )
        
        n_norm = self.phase_lengths["NORMAL"]
        n_build = self.phase_lengths["ANOMALY_BUILDUP"]
        n_anom = self.phase_lengths["ANOMALY"]
        n_rec = self.phase_lengths["RECOVERY"]
        
        phases = (
            ["NORMAL"] * n_norm
            + ["ANOMALY_BUILDUP"] * n_build
            + ["ANOMALY"] * n_anom
            + ["RECOVERY"] * n_rec
        )
        
        ground_truth = (
            [0] * n_norm + [1] * (n_build + n_anom) + [0] * n_rec
        )
        
        dhsv = np.ones(self.n_rows, dtype=int)
        production_valve = np.ones(self.n_rows, dtype=int)
        gas_lift_choke = 45.0 + self.rng.normal(0, 0.3, self.n_rows)
        
        prod_choke_norm = 80.0 + self.rng.normal(0, 0.5, n_norm)
        prod_choke_build = 80.0 + self.rng.normal(0, 0.5, n_build)
        prod_choke_anom = 80.0 + self.rng.normal(0, 0.5, n_anom)
        prod_choke_rec = np.linspace(50.0, 80.0, n_rec) + self.rng.normal(
            0, 0.5, n_rec
        )
        
        production_choke = np.concatenate(
            [
                prod_choke_norm,
                prod_choke_build,
                prod_choke_anom,
                prod_choke_rec,
            ]
        )
        
        
        p_norm = 100.0 + self.rng.normal(0, 0.4, n_norm)
        p_build = np.linspace(100.0, 135.0, n_build) + self.rng.normal(
            0, 0.4, n_build
        )
        p_anom = 135.0 + self.rng.normal(0, 1.2, n_anom)
        p_rec = np.linspace(135.0, 100.0, n_rec) + self.rng.normal(
            0, 0.4, n_rec
        )

        pressure = np.concatenate([p_norm, p_build, p_anom, p_rec])
        
        flow_rate = 250.0 - (pressure - 100.0) * 2.2 + self.rng.normal(
            0, 1.5, self.n_rows
        )
        
        
        t_norm = 65.0 + self.rng.normal(0, 0.2, n_norm)
        t_build = np.linspace(65.0, 78.0, n_build) + self.rng.normal(
            0, 0.2, n_build
        )
        t_anom = 78.0 + self.rng.normal(0, 0.5, n_anom)
        t_rec = np.linspace(78.0, 65.0, n_rec) + self.rng.normal(
            0, 0.2, n_rec
        )

        temperature = np.concatenate([t_norm, t_build, t_anom, t_rec])
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "operating_phase": phases,
                "ground_truth_anomaly": ground_truth,
                "pressure": np.round(pressure, 2),
                "temperature": np.round(temperature, 2),
                "flow_rate": np.round(flow_rate, 2),
                "production_choke": np.round(production_choke, 2),
                "gas_lift_choke": np.round(gas_lift_choke, 2),
                "production_valve": production_valve,
                "dhsv": dhsv,
            }
        )

        return df

def main():
    gen=SensorDataGenerator()
    df= gen.generate_data()
    path=Path(r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\oil gas anomaly\data\sensor_500.csv")
    df.to_csv(path, index=False)
    print(f"Validation passed! Successfully generated and saved dataset to {path}")
    # print(df.head(10))
    # print(df.tail(10))
    # print(df.shape)
    # print("\nPhase Summary:")
    # print(df.groupby("operating_phase")["pressure"].agg(["count", "min", "mean", "max"]))
    
if __name__ == "__main__":
    main()