"""
Generates a small, synthetic "fleet" sample CSV purely for DEMO purposes —
to upload through the Upload & Predict tab and show the pipeline working on
new/unseen data.

This is NOT used for training. The model is trained only on the real AI4I 2020
dataset (see train_models.py) — that separation matters: training on real,
imbalanced, unmanipulated data is what makes the model's evaluation credible.
This script exists purely so you have a clean, realistic file to demo with,
without having to strip columns out of the real dataset live.

Usage:
    python generate_demo_data.py
Output:
    data/demo_fleet_sample.csv
"""
import numpy as np
import pandas as pd
import os

np.random.seed(7)

N = 25
types = np.random.choice(["L", "M", "H"], size=N, p=[0.6, 0.3, 0.1])

rows = []
for i in range(N):
    t = types[i]
    # Roughly a fifth of the sample is deliberately drawn from a "stressed"
    # operating regime (high torque + high tool wear) so the demo shows a
    # mix of low- and high-risk predictions rather than all-healthy machines.
    stressed = np.random.rand() < 0.25

    air_temp = np.random.normal(300, 2)
    process_temp = air_temp + np.random.normal(10, 1)
    rot_speed = np.random.normal(1500, 100) if not stressed else np.random.normal(1350, 80)
    torque = np.random.normal(40, 8) if not stressed else np.random.normal(62, 6)
    tool_wear = np.random.uniform(0, 100) if not stressed else np.random.uniform(180, 250)

    rows.append({
        "Type": t,
        "Air temperature": round(air_temp, 1),
        "Process temperature": round(process_temp, 1),
        "Rotational speed": round(rot_speed, 0),
        "Torque": round(max(torque, 3), 1),
        "Tool wear": round(max(tool_wear, 0), 0),
    })

df = pd.DataFrame(rows)

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "demo_fleet_sample.csv")
df.to_csv(out_path, index=False)
print(f"Demo sample saved to {out_path} ({N} synthetic machines, ~25% in a stressed operating regime).")
