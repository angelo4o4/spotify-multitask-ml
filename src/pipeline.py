from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, PolynomialFeatures
from src import config


def build_preprocessor(model_family: str, task: str = "regression", target_type: str = "continuous", poly_degree=None, use_artist=False) -> ColumnTransformer:
    """
    Build a preprocessor for the given model family and task.
    Args:
        model_family: 'tree' (RF, XGBoost, LightGBM maybe) -> no scaling, TargetEcoder for track_genre
                      'linear' (Linear/Ridge/Lasso/ElasticNet/SVR/MLP) -> scaling + OHE
        task: 'regression' -> track_genre is a feature
              'classification' -> track_genre is the target, must be excluded from the features
    """
    continuous_cols = list(config.CONTINUOUS_FEATURES)
    discrete_cols = list(config.DISCRETE_FEATURES)
    binary_cols = list(config.BINARY_FEATURES)
    genre_cols = list(config.HIGH_CARD_CATEGORICAL) if task == "regression" else []
    artist_cols = list(config.ARTIST_FEATURE) if use_artist else []

    cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)

    if model_family == "tree":
        transformers = [
            ("passthrough", "passthrough", continuous_cols + discrete_cols + binary_cols),
        ]
        if genre_cols:
            transformers.append((
                "genre", 
                TargetEncoder(target_type=target_type, smooth="auto",
                              cv=cv_inner), genre_cols))
        if artist_cols:
            transformers.append(("artists", TargetEncoder(target_type=target_type, smooth="auto", cv=cv_inner), artist_cols))

    elif model_family == "linear":
        cont_steps = [("scaler", StandardScaler())]
        if poly_degree:
            cont_steps += [
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ("poly_scaler", StandardScaler()),  # scale again
            ]
        transformers = [
            ("continuous", Pipeline(cont_steps), continuous_cols),
            ("discrete_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), discrete_cols),
            ("bin_passthrough", "passthrough", binary_cols)
        ]
        if genre_cols:
            transformers.append(("genre_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), genre_cols))
        if artist_cols:
            # target enccoding + scaling
            artist_pipeline = Pipeline([
                ("target_enc", TargetEncoder(target_type=target_type, smooth="auto", cv=cv_inner)),
                ("scaler", StandardScaler())
            ])
            transformers.append(("artist_encoded", artist_pipeline, artist_cols))
    else:
        raise ValueError(f"Invalid model_family: {model_family}. Must be 'tree' or 'linear'.")
    
    return ColumnTransformer(transformers=transformers, remainder="drop")  # verbose_feature_names_out=False