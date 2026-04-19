import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# 1. Load training data
#    Train on: 20, 80, 200, 800, 1000, 5000
# =========================================================
df = pd.read_csv("quadratic_yieldstress_combined.csv")

# =========================================================
# 2. Normalize axis to match paper
#    x/x0 and h/h0
# =========================================================
x0 = 305.0
h0 = 30.5

df["x_ratio"] = df["x"] / x0
df["h_ratio"] = df["flow_depth"] / h0

print("Training data preview:")
print(df.head())
print("\nTraining yield stress values:")
print(sorted(df["yield_stress"].unique()))
print("\nTraining data shape:", df.shape)

# =========================================================
# 3. Define training input/output
# =========================================================
X_train = df[["x_ratio", "yield_stress"]]
y_train = df["h_ratio"]

# =========================================================
# 4. Build and train Polynomial model
# =========================================================
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=4, include_bias=False),
    Ridge(alpha=1.0)
)

poly_model.fit(X_train, y_train)

# =========================================================
# 5. Build and train Random Forest model
# =========================================================
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# =========================================================
# 6. Predict unseen case: yield_stress = 50
# =========================================================
x_ratio_vals = np.linspace(df["x_ratio"].min(), df["x_ratio"].max(), 500)

X_new = pd.DataFrame({
    "x_ratio": x_ratio_vals,
    "yield_stress": 50
})

# Polynomial prediction
h_poly = poly_model.predict(X_new)
h_poly = np.clip(h_poly, 0, None)

# Random Forest prediction
h_rf = rf_model.predict(X_new)
h_rf = np.clip(h_rf, 0, None)

# =========================================================
# 7. Load real CFD data for yield_stress = 50
# =========================================================
df_50 = pd.read_csv("quadratic_50_clean.csv")

# If x_ratio / h_ratio not yet inside file, create them
if "x_ratio" not in df_50.columns:
    df_50["x_ratio"] = df_50["x"] / x0

if "h_ratio" not in df_50.columns:
    if "flow_depth" in df_50.columns:
        df_50["h_ratio"] = df_50["flow_depth"] / h0
    else:
        raise ValueError("df_50 must contain either 'h_ratio' or 'flow_depth'.")

# Sort by x_ratio for safer interpolation / plotting
df_50 = df_50.sort_values("x_ratio").reset_index(drop=True)

print("\nCFD 50-case preview:")
print(df_50.head())
print("\nCFD 50-case shape:", df_50.shape)

# =========================================================
# 8. Check interpolation range
# =========================================================
print("\nInterpolation range check:")
print(f"Training x_ratio range = {df['x_ratio'].min():.3f} to {df['x_ratio'].max():.3f}")
print(f"CFD x_ratio range      = {df_50['x_ratio'].min():.3f} to {df_50['x_ratio'].max():.3f}")

outside_mask = (
    (df_50["x_ratio"] < x_ratio_vals.min()) |
    (df_50["x_ratio"] > x_ratio_vals.max())
)
print("Points outside interpolation range:", outside_mask.sum())

# =========================================================
# 9. Interpolate predictions onto CFD x-grid
# =========================================================
h_poly_interp = np.interp(df_50["x_ratio"], x_ratio_vals, h_poly)
h_rf_interp = np.interp(df_50["x_ratio"], x_ratio_vals, h_rf)

# =========================================================
# 10. Basic model performance
# =========================================================
r2_poly = r2_score(df_50["h_ratio"], h_poly_interp)
r2_rf = r2_score(df_50["h_ratio"], h_rf_interp)

rmse_poly = np.sqrt(mean_squared_error(df_50["h_ratio"], h_poly_interp))
rmse_rf = np.sqrt(mean_squared_error(df_50["h_ratio"], h_rf_interp))

mae_poly = np.mean(np.abs(h_poly_interp - df_50["h_ratio"]))
mae_rf = np.mean(np.abs(h_rf_interp - df_50["h_ratio"]))

print("\nOverall Model Performance:")
print("Polynomial:")
print(f"  R²   = {r2_poly:.4f}")
print(f"  RMSE = {rmse_poly:.4f}")
print(f"  MAE  = {mae_poly:.4f}")

print("Random Forest:")
print(f"  R²   = {r2_rf:.4f}")
print(f"  RMSE = {rmse_rf:.4f}")
print(f"  MAE  = {mae_rf:.4f}")

# =========================================================
# 11. Plot: CFD vs Polynomial vs Random Forest
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(df_50["x_ratio"], df_50["h_ratio"], label="Tan et al., 2025", linewidth=2)
plt.plot(x_ratio_vals, h_poly, "--", label="Polynomial", linewidth=2)
plt.plot(x_ratio_vals, h_rf, "-.", label="Random Forest", linewidth=2)

plt.xlabel("Distance ratio, x/x₀")
plt.ylabel("Flow depth ratio, h/h₀")
plt.title("Model Comparison: CFD vs Polynomial vs Random Forest")
plt.xlim(0, 12.5)
plt.ylim(0, 0.2)
plt.grid(True)
plt.legend()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 12. Signed error
# =========================================================
error_poly = h_poly_interp - df_50["h_ratio"]
error_rf = h_rf_interp - df_50["h_ratio"]

plt.figure(figsize=(8, 5))
plt.plot(df_50["x_ratio"], error_poly, "--", color="orange", label="Polynomial Error", linewidth=2)
plt.plot(df_50["x_ratio"], error_rf, "-.", color="green", label="Random Forest Error", linewidth=2)
plt.axhline(0, color="black", linestyle="-", linewidth=1)

plt.xlabel("Distance ratio, x/x₀")
plt.ylabel("Signed Error (Predicted - CFD)")
plt.title("Signed Error Distribution")
plt.xlim(0, 12.5)
plt.grid(True)
plt.legend()
plt.savefig("error_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 13. Save comparison CSV
# =========================================================
comparison_df = pd.DataFrame({
    "x": df_50["x"],
    "x_ratio": df_50["x_ratio"],
    "yield_stress": 50,
    "true_h_ratio_CFD": df_50["h_ratio"],
    "predicted_h_ratio_polynomial": h_poly_interp,
    "predicted_h_ratio_random_forest": h_rf_interp,
    "signed_error_polynomial": error_poly,
    "signed_error_random_forest": error_rf,
})

comparison_df.to_csv("comparison_yieldstress_50_all_models.csv", index=False)
print("\nSaved comparison file: comparison_yieldstress_50_all_models.csv")
