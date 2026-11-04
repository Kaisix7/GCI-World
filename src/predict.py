"""
Inference & Prediction Generation Module.
Combines saved model predictions or performs inference on new test batches.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def to_power_rank(preds: np.ndarray, p: float = 3.5) -> np.ndarray:
    """Apply power rank transformation."""
    ranked = rankdata(preds) / len(preds)
    return ranked ** p

def blend_predictions(
    sub_paths: list[Path],
    weights: list[float],
    output_path: Path,
    power: float = 3.5
):
    print(f"Blending {len(sub_paths)} predictions with power={power}...")
    dfs = [pd.read_csv(p) for p in sub_paths]
    sk_ids = dfs[0]["SK_ID_CURR"].values
    
    ranked_preds = []
    for df in dfs:
        p = df["TARGET"].values
        ranked_preds.append(to_power_rank(p, p=power))
        
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    
    final_pred = np.zeros(len(sk_ids))
    for w, rp in zip(weights, ranked_preds):
        final_pred += w * rp
        
    final_pred = (final_pred - final_pred.min()) / (final_pred.max() - final_pred.min())
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"SK_ID_CURR": sk_ids, "TARGET": final_pred})
    out_df.to_csv(output_path, index=False)
    print(f"Generated submission at {output_path} with {len(out_df)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output-file", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--power", type=float, default=3.5)
    args = parser.parse_args()
    
    blend_predictions(args.outputs, args.weights, args.output_file, args.power)
