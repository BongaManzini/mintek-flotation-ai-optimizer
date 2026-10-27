import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "MiningProcess_Flotation_Plant_Database.csv"
OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

print("⏳ Loading dataset...")
start_t = time.time()

# Parse columns with comma as decimal
df = pd.read_csv(DATA_PATH, decimal=",")
print(f"✅ Loaded {len(df):,} rows in {time.time() - start_t:.2f}s")

# Ensure date is parsed
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Feature engineering
print("🛠️ Engineering process features...")
# Specific Reagent Dosages (relative to ore feed)
df['amina_per_silica_feed'] = df['Amina Flow'] / (df['% Silica Feed'] + 1e-5)
df['starch_per_iron_feed'] = df['Starch Flow'] / (df['% Iron Feed'] + 1e-5)
df['pulp_flow_density_prod'] = df['Ore Pulp Flow'] * df['Ore Pulp Density']

# Aeration groupings: Roughers (1-3), Scavengers (4-5), Cleaners (6-7)
df['rougher_air_avg'] = df[['Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow']].mean(axis=1)
df['scavenger_air_avg'] = df[['Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow']].mean(axis=1)
df['cleaner_air_avg'] = df[['Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow']].mean(axis=1)

# Level groupings
df['rougher_level_avg'] = df[['Flotation Column 01 Level', 'Flotation Column 02 Level', 'Flotation Column 03 Level']].mean(axis=1)
df['scavenger_cleaner_level_avg'] = df[['Flotation Column 04 Level', 'Flotation Column 05 Level', 'Flotation Column 06 Level', 'Flotation Column 07 Level']].mean(axis=1)

# Rolling features (hydraulic residence time ~ 30 minutes = 90 rows at 20s interval)
# Downsample to 1-minute intervals for clean training and time-series benchmark
df_sampled = df.iloc[::3].copy().reset_index(drop=True) # every 1 min
print(f"📊 Resampled to 1-min records: {len(df_sampled):,} rows")

features = [
    '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 
    'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
    'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow',
    'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow',
    'Flotation Column 01 Level', 'Flotation Column 02 Level', 'Flotation Column 03 Level',
    'Flotation Column 04 Level', 'Flotation Column 05 Level', 'Flotation Column 06 Level', 'Flotation Column 07 Level',
    'amina_per_silica_feed', 'starch_per_iron_feed', 'pulp_flow_density_prod',
    'rougher_air_avg', 'scavenger_air_avg', 'cleaner_air_avg',
    'rougher_level_avg', 'scavenger_cleaner_level_avg'
]

target_silica = '% Silica Concentrate'
target_iron = '% Iron Concentrate'

X = df_sampled[features]
y_silica = df_sampled[target_silica]
y_iron = df_sampled[target_iron]

# Chronological split: 80% train, 20% test (preserving industrial time sequence)
split_idx = int(len(df_sampled) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_silica_train, y_silica_test = y_silica.iloc[:split_idx], y_silica.iloc[split_idx:]
y_iron_train, y_iron_test = y_iron.iloc[:split_idx], y_iron.iloc[split_idx:]

print(f"🚂 Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

# 1. Baseline Heuristic Rule
def heuristic_predict(df_feat):
    # Industrial rule-of-thumb: silica concentrate increases with feed silica, decreased by amina flow efficiency
    eff = (df_feat['Amina Flow'] / 500.0) * (1.0 - (df_feat['Ore Pulp pH'] - 10.0).abs() * 0.1)
    eff = eff.clip(lower=0.3)
    pred = (df_feat['% Silica Feed'] * 0.16) / eff
    return pred.clip(lower=0.5, upper=6.0)

y_heur_pred = heuristic_predict(X_test)
mae_heur = mean_absolute_error(y_silica_test, y_heur_pred)
rmse_heur = np.sqrt(mean_squared_error(y_silica_test, y_heur_pred))
r2_heur = r2_score(y_silica_test, y_heur_pred)

print(f"--- Baseline Heuristic --- MAE: {mae_heur:.4f}, RMSE: {rmse_heur:.4f}, R2: {r2_heur:.4f}")

# 2. Linear Regression (Ridge)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_silica_train)
y_ridge_pred = ridge.predict(X_test)
mae_ridge = mean_absolute_error(y_silica_test, y_ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_silica_test, y_ridge_pred))
r2_ridge = r2_score(y_silica_test, y_ridge_pred)
print(f"--- Ridge Regression --- MAE: {mae_ridge:.4f}, RMSE: {rmse_ridge:.4f}, R2: {r2_ridge:.4f}")

# 3. Random Forest (on 30k sample for fast benchmark)
rf_sample_idx = np.random.RandomState(42).choice(len(X_train), size=35000, replace=False)
rf = RandomForestRegressor(n_estimators=60, max_depth=16, n_jobs=-1, random_state=42)
rf.fit(X_train.iloc[rf_sample_idx], y_silica_train.iloc[rf_sample_idx])
y_rf_pred = rf.predict(X_test)
mae_rf = mean_absolute_error(y_silica_test, y_rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_silica_test, y_rf_pred))
r2_rf = r2_score(y_silica_test, y_rf_pred)
print(f"--- Random Forest --- MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")

# 4. HistGradientBoostingRegressor (Gradient Boosted Trees - Industry SOTA for Tabular)
hgbr_silica = HistGradientBoostingRegressor(max_iter=150, max_depth=12, learning_rate=0.08, random_state=42)
t0 = time.time()
hgbr_silica.fit(X_train, y_silica_train)
fit_time = time.time() - t0
y_hgbr_pred = hgbr_silica.predict(X_test)
mae_hgbr = mean_absolute_error(y_silica_test, y_hgbr_pred)
rmse_hgbr = np.sqrt(mean_squared_error(y_silica_test, y_hgbr_pred))
r2_hgbr = r2_score(y_silica_test, y_hgbr_pred)
print(f"--- HistGradientBoosting --- MAE: {mae_hgbr:.4f}, RMSE: {rmse_hgbr:.4f}, R2: {r2_hgbr:.4f} (Fit: {fit_time:.2f}s)")

# Train Iron Concentrate Predictor as well
hgbr_iron = HistGradientBoostingRegressor(max_iter=120, max_depth=10, learning_rate=0.08, random_state=42)
hgbr_iron.fit(X_train, y_iron_train)
y_iron_pred = hgbr_iron.predict(X_test)
mae_iron = mean_absolute_error(y_iron_test, y_iron_pred)
r2_iron = r2_score(y_iron_test, y_iron_pred)
print(f"--- Iron Model --- MAE: {mae_iron:.4f}, R2: {r2_iron:.4f}")

# Feature importance using permutation on test sample
from sklearn.inspection import permutation_importance
test_sample_idx = np.random.RandomState(42).choice(len(X_test), size=5000, replace=False)
perm_res = permutation_importance(hgbr_silica, X_test.iloc[test_sample_idx], y_silica_test.iloc[test_sample_idx], n_repeats=3, random_state=42)
feat_imp = sorted(zip(features, perm_res.importances_mean), key=lambda x: x[1], reverse=True)

print("\n🏆 Top 10 Most Influential Flotation Features:")
for f, imp in feat_imp[:10]:
    print(f"  {f:32s} : {imp:.5f}")

# Benchmark summary
benchmark_results = {
    "heuristic": {"name": "Industrial Rule-of-Thumb", "mae": float(mae_heur), "rmse": float(rmse_heur), "r2": float(r2_heur)},
    "ridge": {"name": "Ridge Linear Model", "mae": float(mae_ridge), "rmse": float(rmse_ridge), "r2": float(r2_ridge)},
    "rf": {"name": "Random Forest Regressor", "mae": float(mae_rf), "rmse": float(rmse_rf), "r2": float(r2_rf)},
    "hgbr": {"name": "Gradient Boosted Trees (Mintek AI Soft Sensor)", "mae": float(mae_hgbr), "rmse": float(rmse_hgbr), "r2": float(r2_hgbr), "iron_mae": float(mae_iron), "iron_r2": float(r2_iron)}
}

# Export small sample of test telemetry (actual vs predicted) for website interactive chart
sub_idx = np.linspace(0, len(y_silica_test)-1, 120, dtype=int)
test_timeline = {
    "timestamps": [str(t) for t in df_sampled['date'].iloc[split_idx:].iloc[sub_idx]],
    "actual_silica": [round(float(v), 2) for v in y_silica_test.iloc[sub_idx]],
    "predicted_silica": [round(float(v), 2) for v in y_hgbr_pred[sub_idx]],
    "baseline_silica": [round(float(v), 2) for v in y_heur_pred.iloc[sub_idx]],
    "feed_silica": [round(float(v), 2) for v in X_test['% Silica Feed'].iloc[sub_idx]],
    "amina_flow": [round(float(v), 1) for v in X_test['Amina Flow'].iloc[sub_idx]],
    "actual_iron": [round(float(v), 2) for v in y_iron_test.iloc[sub_idx]],
    "predicted_iron": [round(float(v), 2) for v in y_iron_pred[sub_idx]]
}

# Linear surrogate weights for browser-based interactive simulation
# We can also fit a clean polynomial/linear surrogate on the inputs for the client-side calculator
sim_feats = ['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density', 'rougher_air_avg']
surrogate = Ridge(alpha=10.0)
surrogate.fit(X_train[sim_feats], y_silica_train)
surrogate_iron = Ridge(alpha=10.0)
surrogate_iron.fit(X_train[sim_feats], y_iron_train)

surrogate_data = {
    "features": sim_feats,
    "silica_intercept": float(surrogate.intercept_),
    "silica_coefs": {f: float(c) for f, c in zip(sim_feats, surrogate.coef_)},
    "iron_intercept": float(surrogate_iron.intercept_),
    "iron_coefs": {f: float(c) for f, c in zip(sim_feats, surrogate_iron.coef_)}
}

# Save metadata
with open(os.path.join(OUT_DIR, "benchmark_metrics.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

with open(os.path.join(OUT_DIR, "feature_importance.json"), "w") as f:
    json.dump([{"feature": f, "importance": float(imp)} for f, imp in feat_imp], f, indent=2)

with open(os.path.join(OUT_DIR, "test_timeline.json"), "w") as f:
    json.dump(test_timeline, f, indent=2)

with open(os.path.join(OUT_DIR, "surrogate_model.json"), "w") as f:
    json.dump(surrogate_data, f, indent=2)

print("\n🎉 Benchmarks and model telemetry successfully generated in model_artifacts/!")
