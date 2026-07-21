"""Configuration variables for the project."""
DATA_PATH = "data/dataset.csv"
CONTINUOUS_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
DISCRETE_FEATURES = ["key", "time_signature"]  # Discrete features with limited values (OHE)
BINARY_FEATURES = ["explicit", "mode"]  # True or False, passtrhough to model as 0/1
HIGH_CARD_CATEGORICAL = ["track_genre"]  # 114 categories, managed differently (target/OHE)

ID_FEATURES = ["track_id", "artists", "album_name", "track_name"]
ARTIST_FEATURE = ["artists"]

TARGET_REG = "popularity"
TARGET_CLASS = "track_genre"