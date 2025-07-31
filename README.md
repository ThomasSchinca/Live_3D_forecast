# 3D Live Forecasting 

## 1. The model

This script uses the ShapeFinder3D to make live forecasting of spatio-temporal patterns using historical data at the PRIO-Grid month level.

- It creates **3D arrays** where conflict is detected, called **trajectories**
- For each trajectory, it **matches similar patterns** from the past based on spatial and temporal similarity.
- Matched regions are **aligned, normalized, and averaged** to produce a predicted future activity pattern.
- These predictions are **clustered**, denoised, and spatially projected onto a reference grid.
- The entire operation is **parallelized**, supports **intermediate checkpointing**, and automatically saves results at key stages.
- It should run in around **15 hours**.
---

## 2. Outputs of the Model

### ✅ `df_output.csv` — **Main Forecast Table**
- **Type:** CSV.
- **Dimensions:** `h × N` (where `h = 6` months, and `N` is the number of PRIO-GRID cells).
- **Content:** Predicted activity level for each grid cell at each future time step.
- **Index:** Future time steps (0 to `h-1`).
- **Columns:** PRIO-GRID cell IDs.
- **Purpose:** Main deliverable; this file represents the spatio-temporal forecast.

---

### ✅ `gr_b_tot.pkl` — **Active Grid Cells**
- **Type:** Pickle (list of lists).
- **Content:** For each input trajectory, the list of PRIO-GRID cells predicted to become active.
- **Use Case:** Allows quick identification of future areas of interest.

---

### 🧩 `matches.pkl` — **Matched Subpatterns**
- **Type:** Pickle (dictionary).
- **Content:** For each trajectory index, a DataFrame of best-matching historical subregions (coordinates, distance scores, transformations).
- **Purpose:** Diagnostic and interpretability — reveals which historical patterns contributed to each forecast.

---

### 🧩 `input.pkl` — **Original Source Patterns**
- **Type:** Pickle (dictionary).
- **Content:** The input 3D subarrays corresponding to each original trajectory used in forecasting.
- **Purpose:** Allows reproduction, validation, or further experimentation from raw input structures.

---

These four outputs collectively provide:
- The **forecast results** (`df_output.csv`, `gr_b_tot.pkl`)
- The **underlying justifications and diagnostics** (`matches.pkl`, `input.pkl`)
