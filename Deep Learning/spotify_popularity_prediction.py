import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Load your dataset
spotify = pd.read_csv("spotify.csv")

# Drop missing rows, split features + target
X = spotify.copy().dropna()
y = X.pop("track_popularity")
artists = X["track_artist"]

# Feature groups
features_num = [
    "danceability","energy","key","loudness","mode",
    "speechiness","acousticness","instrumentalness",
    "liveness","valence","tempo","duration_ms"
]
features_cat = ["playlist_genre"]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat)
)

# Grouped train/validation split (by artist)
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size, n_splits=1)
    train_idx, valid_idx = next(splitter.split(X, y, groups=group))
    return (X.iloc[train_idx], X.iloc[valid_idx],
            y.iloc[train_idx], y.iloc[valid_idx])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

# Transform data
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

# Scale output to 0–1
y_train = y_train / 100
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape:", input_shape)

# Model with Early Stopping

early_stopping = callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=input_shape),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mae"
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

# Plot learning curves
history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot(title="Training vs Validation Loss")
plt.show()

print("Minimum Validation Loss:", history_df["val_loss"].min())
