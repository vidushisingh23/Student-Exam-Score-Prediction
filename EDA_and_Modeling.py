import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# Ensure folders exist
os.makedirs("model", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)

# Load datasets and label with subject
math_df = pd.read_csv("data/math_dataset.csv")
math_df["subject"] = "Math"

por_df = pd.read_csv("data/portugese_dataset.csv")
por_df["subject"] = "Portuguese"

# Combine both datasets
df = pd.concat([math_df, por_df], ignore_index=True)

# Data Summary
print("Initial Data Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Data Types:\n", df.dtypes)

# Feature Engineering
df_encoded = pd.get_dummies(df, drop_first=True)

# Top Correlated Features with G3
correlations = df_encoded.corr()['G3'].drop('G3').sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=correlations.values[:10], y=correlations.index[:10], palette="viridis")
plt.title("Top 10 Features Correlated with G3")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("data/plots/top10_corr_g3.png")
plt.close()

# Data Visualization (EDA)
# Heatmap of top 20 correlated features with G3
top_features = correlations.head(20).index.tolist() + ['G3']
plt.figure(figsize=(14, 10))
sns.heatmap(
    df_encoded[top_features].corr(),
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.6},
    xticklabels=True,
    yticklabels=True
)
plt.title("Correlation Heatmap of Top Features", fontsize=16)
plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("data/plots/eda_heatmap.png", dpi=300)
plt.close()

# Study time vs G3
sns.boxplot(x='studytime', y='G3', data=df)
plt.title("Study Time vs Final Grade")
plt.savefig("data/plots/studytime_vs_g3.png")
plt.close()

# Absences vs G3
sns.scatterplot(x='absences', y='G3', data=df)
plt.title("Absences vs Final Grade")
plt.savefig("data/plots/absences_vs_g3.png")
plt.close()

# G1 vs G3
sns.regplot(x='G1', y='G3', data=df)
plt.title("G1 vs Final Grade")
plt.savefig("data/plots/g1_vs_g3.png")
plt.close()

# G2 vs G3
sns.regplot(x='G2', y='G3', data=df)
plt.title("G2 vs Final Grade")
plt.savefig("data/plots/g2_vs_g3.png")
plt.close()

# Feature Selection
selected_features = ['G1', 'G2', 'studytime', 'absences']
X = df[selected_features]
y = df['G3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Model Training & Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    results[name] = {'R2': r2, 'MAE': mae, 'MSE': mse}
    print(f"{name} - R2: {r2:.3f}, MAE: {mae:.2f}, MSE: {mse:.2f}")
# Save metrics for UI display
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

with open("model/metrics.pkl", "wb") as f:
    pickle.dump({
        "r2": round(final_r2, 3),
        "mae": round(final_mae, 2),
        "rmse": round(final_rmse, 2)
    }, f)


# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
print("Best RF Parameters:", grid_search.best_params_)

# Save best model
best_model = grid_search.best_estimator_
with open("model/regressor.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Prediction Visualization
y_pred = best_model.predict(X_test_scaled)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, c='green')
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Scores")
plt.grid(True)
plt.savefig("data/plots/actual_vs_pred.png")
plt.close()

print("\nProject pipeline complete. Model, scaler, and visualizations saved.")
