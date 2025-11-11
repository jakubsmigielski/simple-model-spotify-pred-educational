import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os 
import warnings


import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


FILE_PATH = 'data/spotify_data.csv' 
TARGET_COLUMNS = 'track_popularity'
CHARTS_DIR = 'charts' 

if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)
    print(f"Created directory: {CHARTS_DIR}")

print(f"Loading data from: {FILE_PATH}")
df = pd.read_csv(FILE_PATH)
df_clean = df.copy() 



columns_to_drop_now = [
    'track_id', 'track_name', 'album_id', 'album_name'
]
df_clean = df_clean.drop(columns=columns_to_drop_now, errors='ignore')

df_clean['explicit'] = df_clean['explicit'].replace({True: 1, False: 0}).astype(float) 

def convert_duration_to_seconds(duration_str):
    if pd.isna(duration_str) or not isinstance(duration_str, (str, float)):
        return np.nan
    try:
        parts = str(duration_str).split('.')
        minutes = int(parts[0])
        seconds = int(parts[1][:2].ljust(2, '0')) if len(parts) > 1 else 0
        return minutes * 60 + seconds
    except:
        return np.nan

df_clean['track_duration_sec'] = df_clean['track_duration_min'].apply(convert_duration_to_seconds)
df_clean = df_clean.drop('track_duration_min', axis=1)

df_clean['album_release_date'] = pd.to_datetime(df_clean['album_release_date'], errors='coerce')
current_year = datetime.now().year
df_clean['track_age_years'] = current_year - df_clean['album_release_date'].dt.year
df_clean = df_clean.drop('album_release_date', axis=1)

df_clean = pd.get_dummies(df_clean, columns=['album_type'], prefix='type', drop_first=True)

df_clean['artist_genre_count'] = df_clean['artist_genres'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' and x != 'nan' else 0
)

df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)
print("Data Cleaning and Feature Engineering completed.")



corr_cols = [
    'track_popularity', 'artist_popularity', 'artist_followers', 
    'track_duration_sec', 'track_age_years', 'explicit', 'artist_genre_count'
]
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean[corr_cols].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Key Features and Popularity')
plt.savefig(os.path.join(CHARTS_DIR, 'eda_1_correlation_heatmap.png'))
print("Saved chart 1: charts/eda_1_correlation_heatmap.png")

plt.figure(figsize=(12, 6))
df_plot = df_clean[df_clean['track_age_years'] >= 0].groupby('track_age_years')[TARGET_COLUMNS].mean().reset_index()
sns.lineplot(data=df_plot, x='track_age_years', y=TARGET_COLUMNS)
plt.title('Average Popularity by Track Age (Years)')
plt.xlabel('Track Age (Years)')
plt.ylabel('Average Popularity')
plt.grid(True)
plt.savefig(os.path.join(CHARTS_DIR, 'eda_2_age_vs_popularity.png'))
print("Saved chart 2: charts/eda_2_age_vs_popularity.png")

df_plot_explicit = df_clean.copy()
df_plot_explicit['Explicit Status'] = df_plot_explicit['explicit'].apply(
    lambda x: 'Explicit (1)' if x == 1 else 'Not Explicit (0)'
)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Explicit Status', y=TARGET_COLUMNS, data=df_plot_explicit)
plt.title('Popularity vs. Explicit Tag')
plt.xlabel('Explicit Tag')
plt.ylabel('Track Popularity')
plt.savefig(os.path.join(CHARTS_DIR, 'eda_3_explicit_vs_popularity.png'))
print("Saved chart 3: charts/eda_3_explicit_vs_popularity.png")



columns_to_drop_model = ['artist_name', 'artist_genres'] 
df_model = df_clean.drop(columns=columns_to_drop_model, errors='ignore').select_dtypes(include=np.number)

Y = df_model[TARGET_COLUMNS]
X = df_model.drop(TARGET_COLUMNS, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2, 
    random_state=42 
)
print(f"\nModel data prepared. Training on {X_train.shape[0]} records.")


results = {}

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, name):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse) 
    r2 = r2_score(Y_test, Y_pred)
    results[name] = {'RMSE': rmse, 'R2': r2, 'Y_pred': Y_pred, 'Model': model}
    return Y_pred

model_rf = RandomForestRegressor(n_estimators=100, random_state=42) 
Y_pred_rf = train_and_evaluate(model_rf, X_train, Y_train, X_test, Y_test, 'Random Forest')

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, objective='reg:squarederror')
Y_pred_xgb = train_and_evaluate(model_xgb, X_train, Y_train, X_test, Y_test, 'XGBoost')



results_df = pd.DataFrame([
    {'Model': name, 'RMSE': res['RMSE'], 'R2': res['R2']} 
    for name, res in results.items()
])

print("\n--- METRIC RESULTS COMPARISON ---")
print(results_df.to_markdown(index=False))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=Y_test, y=results['Random Forest']['Y_pred'], alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.title(f'Random Forest: Prediction (R2: {results["Random Forest"]["R2"]:.4f})')
plt.xlabel('Actual Popularity'); plt.ylabel('Predicted Popularity')
plt.xlim(0, 100); plt.ylim(0, 100)

plt.subplot(1, 2, 2)
sns.scatterplot(x=Y_test, y=results['XGBoost']['Y_pred'], alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.title(f'XGBoost: Prediction (R2: {results["XGBoost"]["R2"]:.4f})')
plt.xlabel('Actual Popularity'); plt.ylabel('Predicted Popularity')
plt.xlim(0, 100); plt.ylim(0, 100)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'model_comparison_scatter_final.png'))
print("\nSaved chart 4: charts/model_comparison_scatter_final.png")

model_xgb_final = results['XGBoost']['Model']
feature_importances = pd.Series(model_xgb_final.feature_importances_, index=X_train.columns)
top_10_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_features.values, y=top_10_features.index, palette='viridis')
plt.title('Top 10 Feature Importances (XGBoost)')
plt.xlabel('Feature Importance Score'); plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'feature_importance_final.png'))
print("Saved chart 5: charts/feature_importance_final.png")