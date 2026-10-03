"""Read-only verification of saved predictions and prepared split assignments."""
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parent
checks = 0
for split in ("primary", "robust"):
    base = ROOT / "full" / split
    results = json.loads((base / "evaluation/results.json").read_text())
    for task in ("source", "hour48", "trio"):
        clusters = []
        for part in ("fit", "val", "test"):
            with np.load(base / "data" / f"{task}_{part}.npz") as z:
                clusters.append(set(z["cluster"].tolist()))
        assert all(not clusters[i] & clusters[j] for i in range(3) for j in range(i + 1, 3))
        checks += 1
        for part in ("val", "test"):
            with np.load(base / "evaluation" / f"{task}_{part}_predictions.npz") as z:
                for model, expected in results[part][task].items():
                    pred = z[model]
                    observed = np.mean((pred - z["y"]) ** 2)
                    np.testing.assert_allclose(observed, expected["mse"], rtol=2e-6, atol=2e-7)
                    checks += 1
metrics = json.loads((ROOT / "fractions/evaluation/metrics.json").read_text())
for record in metrics:
    with np.load(ROOT / "fractions/evaluation" / (record["subset"] + "_predictions.npz")) as z:
        observed = np.mean((z[record["model"]] - z["y"]) ** 2)
        np.testing.assert_allclose(observed, record["mse"], rtol=2e-6, atol=2e-7)
        checks += 1
print(f"Verified {checks} saved-prediction and split checks; no models trained.")
