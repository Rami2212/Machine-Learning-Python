import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load training data
TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        "train.csv or test.csv not found"
    )

home_data = pd.read_csv(TRAIN_PATH)

# Target variable
y = home_data["SalePrice"]

# Features
features = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
]

# Select features
X = home_data[features]

# 2. Train/Validation check
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

val_preds = rf_model.predict(val_X)
mae = mean_absolute_error(val_y, val_preds)

print(f"Validation MAE: {mae:,.0f}")

# 3. Train on full dataset
rf_full_model = RandomForestRegressor(random_state=1)
rf_full_model.fit(X, y)

# 4. Load test data & predict
test_data = pd.read_csv(TEST_PATH)
test_X = test_data[features]
test_predictions = rf_full_model.predict(test_X)

# 5. Create submission
output = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_predictions
})

output.to_csv("submission.csv", index=False)
print("ubmission.csv generated successfully!")
