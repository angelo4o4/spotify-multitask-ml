import pandas as pd
from src import config


def load_and_clean_data(file_path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load raw data, deduplicate by popularity and drop rows with missing data."""
    df = pd.read_csv(file_path)

    # Deduplication by popularity
    df_clean = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates(subset=["track_name", "artists"], keep="first")
        .copy()
    )
    # Conversion of binary features to 0/1
    for col in config.BINARY_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)

    # Drop rows with missing data in id_features
    metadata = [col for col in ["track_name", "artists", "album_name"] if col in config.ID_FEATURES]
    if metadata:
        df_clean = df_clean.dropna(subset=metadata, how="any")
    
    # Removing index column: Unnamed: 0
    if "Unnamed: 0" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Unnamed: 0"])
    
    return df_clean