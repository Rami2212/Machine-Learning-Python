import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# 1. Load Dataset
DATA_PATH = "./melb_data.csv"

data = pd.read_csv(DATA_PATH)

y = data["Price"]
X = data.drop("Price", axis=1)

# 2. Train/Validation Split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# 3. Select numerical + categorical columns
categorical_cols = [
    cname for cname in X_train_full.columns 
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

numerical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].dtype in ["int64", "float64"]
]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# 4. Preprocessing: Missing Values + Categorical Encoding
numerical_transformer = SimpleImputer(strategy="constant")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 5. Model: RandomForest
rf_model = RandomForestRegressor(n_estimators=120, random_state=0)

# Pipeline = Preprocessing + Model
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

# Train
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_valid)
rf_mae = mean_absolute_error(y_valid, rf_preds)

print("\n🔹 Random Forest MAE:", rf_mae)

# 6. Cross-Validation
cv_scores = -1 * cross_val_score(
    rf_pipeline, X_train, y_train,
    cv=5, scoring="neg_mean_absolute_error"
)

print("\n🔹 Cross-Validation MAE Scores:", list(cv_scores))
print("🔹 Average CV MAE:", cv_scores.mean())

# 7. XGBoost Model (optional)
try:
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=4,
        eval_metric="mae"
    )

    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ])

    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_valid)
    xgb_mae = mean_absolute_error(y_valid, xgb_preds)

    print("\n🔹 XGBoost MAE:", xgb_mae)

except Exception as e:
    print("\nXGBoost not installed. Install via:")
    print("    pip install xgboost")
    print("Skipping XGBoost step.\n")

# 8. Data Leakage Check
leakage_cols = [
    col for col in X_train.columns
    if col not in X_valid.columns
]

if leakage_cols:
    print("POSSIBLE DATA LEAKAGE in columns:", leakage_cols)
else:
    print("No column leakage detected")

print("\nFinished! All ML steps completed successfully.\n")
