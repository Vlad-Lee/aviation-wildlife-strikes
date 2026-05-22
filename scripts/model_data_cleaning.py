import pandas as pd
import numpy as np
import zipfile
import io
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
}

_PHASE_MAP = {
    "TAKE-OFF RUN": "TAKEOFF", "DEPARTURE": "TAKEOFF", "CLIMB": "CLIMB",
    "EN ROUTE": "CRUISE",  "DESCENT": "DESCENT",  "APPROACH": "APPROACH",
    "ARRIVAL": "APPROACH", "LANDING ROLL": "LANDING", "TAXI": "GROUND",
    "PARKED": "GROUND",    "LOCAL": "UNKNOWN",
}

_SIZE_MASS_MAP = {"SMALL": 1, "MEDIUM": 5, "LARGE": 15}

_NO_PRECIP_VALS = {"NONE", "NO", "N", "NAN", ""}

_DAM_COLS = [
    "DAM_RAD", "DAM_WINDSHLD", "DAM_NOSE",
    "DAM_ENG1", "DAM_ENG2", "DAM_ENG3", "DAM_ENG4",
    "DAM_PROP", "DAM_WING_ROT", "DAM_FUSE", "DAM_LG",
    "DAM_TAIL", "DAM_LGHTS", "DAM_OTHER",
]

_ING_COLS = ["ING_ENG1", "ING_ENG2", "ING_ENG3", "ING_ENG4", "INGESTED_OTHER"]

# Final feature columns entering each stage — no raw intermediates.
# Stage 2 extra columns are post-event and excluded from Stage 1 to prevent leakage.
_STAGE1_FEATURES = [
    "MONTH", "SPEED", "LOG_KINETIC_ENERGY", "HEIGHT_BIN",
    "LATITUDE", "LONGITUDE", "AC_MASS", "ENGINE_TYPE", "NUM_ENGS",
    "PHASE_GROUP", "OPERATOR_CAT", "FAAREGION", "WARNED_FLAG",
    "TIME_BUCKET", "BIRD_SEASON", "SKY", "HAS_PRECIP",
    "NUM_STRUCK_CAT", "NUM_SEEN_CAT", "IS_MULTI_STRIKE",
]

_STAGE2_EXTRA_FEATURES = [
    "DAM_RAD", "DAM_WINDSHLD", "DAM_NOSE",
    "DAM_ENG1", "DAM_ENG2", "DAM_ENG3", "DAM_ENG4",
    "DAM_PROP", "DAM_WING_ROT", "DAM_FUSE", "DAM_LG",
    "DAM_TAIL", "DAM_LGHTS", "DAM_OTHER",
    "ING_ENG1", "ING_ENG2", "ING_ENG3", "ING_ENG4", "INGESTED_OTHER",
    "ANY_ENGINE_INGESTION", "ENGINES_DAMAGED",
    "LOG_AOS",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_raw_data(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, dtype=str, engine="openpyxl")
    else:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    return df


def normalize_count_bucket(val):
    """Normalize a NUM_STRUCK / NUM_SEEN cell to one of four clean bucket labels."""
    val = str(val).strip().upper()
    if val in ["NAN", "NONE", "", "NA", "N/A", "NOT REPORTED"]:
        return "UNKNOWN"
    if val == "1":       return "1"
    if val == "2-10":    return "2-10"
    if val == "11-100":  return "11-100"
    if any(x in val for x in ["MORE", "100+", ">100", "OVER 100"]):
        return "100+"
    return "UNKNOWN"


def _parse_binary_flag(val):
    """Convert Y/N/blank damage columns to 1.0/0.0. Blank means not struck — not unknown."""
    if pd.isna(val):
        return 0.0
    return 0.0 if str(val).strip().upper() in ("", "N", "NO", "NAN", "0", "FALSE") else 1.0


def _parse_precip(val):
    """Return 1.0 if any precipitation is present, 0.0 if all parts are no-precip, NaN if unrecorded."""
    if pd.isna(val):
        return np.nan
    parts = [p.strip().upper() for p in str(val).split(",")]
    return 0.0 if all(p in _NO_PRECIP_VALS for p in parts) else 1.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _rename_columns(df):
    return df.rename(columns={
        "TYPE_ENG":       "ENGINE_TYPE",
        "INCIDENT_MONTH": "MONTH",
        "PRECIPITATION":  "PRECIP",
        "BIRDS_SEEN":     "NUM_SEEN",    # legacy name guard
        "BIRDS_STRUCK":   "NUM_STRUCK",  # legacy name guard
    })


def _apply_filters(df):
    df = df[df["FAAREGION"].notna() & (df["FAAREGION"].str.strip().str.upper() != "FGN")].copy()
    df = df[df["STATE"].isin(_US_STATES)].copy()
    return df


def _engineer_flight_context(df):
    if "AC_MASS" in df.columns:
        df["AC_MASS"] = df["AC_MASS"].astype(str)

    def _map_operator(name):
        name = str(name).upper()
        if any(w in name for w in ["MILITARY", "NAVY", "AIR FORCE", "ARMY", "USAF", "USCG", "COAST GUARD"]):
            return "MIL"
        if any(w in name for w in ["GOV", "FED", "STATE", "COUNTY", "CITY"]):
            return "GOV"
        if any(w in name for w in ["PRIVATE", "PRIVATELY", "PERSONAL", "PVT", "CHARTER"]):
            return "PVT"
        return "BUS"

    if "OPERATOR" in df.columns:
        df["OPERATOR_CAT"] = df["OPERATOR"].apply(_map_operator)

    if "PHASE_OF_FLIGHT" in df.columns:
        df["PHASE_GROUP"] = (
            df["PHASE_OF_FLIGHT"].str.strip().str.upper()
            .map(_PHASE_MAP).fillna("UNKNOWN")
        )

    def _bin_height(h):
        h = pd.to_numeric(h, errors="coerce")
        if pd.isna(h) or h < 0: return "UNKNOWN"
        if h == 0:               return "GROUND"
        if h < 500:              return "LOW"
        if h < 3000:             return "APPROACH_CLIMB"
        return "CRUISE"

    if "HEIGHT" in df.columns:
        df["HEIGHT_BIN"] = df["HEIGHT"].apply(_bin_height)

    return df


def _engineer_strike_counts(df):
    for raw_col, cat_col in [("NUM_STRUCK", "NUM_STRUCK_CAT"), ("NUM_SEEN", "NUM_SEEN_CAT")]:
        if raw_col in df.columns:
            df[cat_col] = df[raw_col].apply(normalize_count_bucket)

    if "NUM_STRUCK_CAT" in df.columns:
        df["IS_MULTI_STRIKE"] = df["NUM_STRUCK_CAT"].isin(["2-10", "11-100", "100+"]).astype(float)

    return df


def _engineer_temporal(df):
    if "TIME" in df.columns:
        hour = pd.to_datetime(df["TIME"], format="%H:%M", errors="coerce").dt.hour
        df["TIME_BUCKET"] = (
            pd.cut(hour, bins=[-1, 5, 11, 17, 23],
                   labels=["NIGHT", "MORNING", "AFTERNOON", "EVENING"])
            .astype(str).replace("nan", "UNKNOWN")
        )

    def _map_bird_season(month):
        month = pd.to_numeric(month, errors="coerce")
        if pd.isna(month):       return "UNKNOWN"
        if month in [3, 4, 5]:  return "SPRING"
        if month in [6, 7]:     return "SUMMER"
        if month in [8, 9, 10]: return "FALL"
        return "WINTER"

    if "MONTH" in df.columns:
        df["BIRD_SEASON"] = df["MONTH"].apply(_map_bird_season)

    return df


def _engineer_environmental(df):
    if "SKY" in df.columns:
        df["SKY"] = (
            df["SKY"].astype(str).str.strip().str.upper()
            .replace({"NAN": "UNKNOWN", "": "UNKNOWN"})
        )
    if "PRECIP" in df.columns:
        df["HAS_PRECIP"] = df["PRECIP"].apply(_parse_precip)

    return df


def _engineer_damage_features(df):
    for col in _DAM_COLS + _ING_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_binary_flag)

    eng_ing = [c for c in ["ING_ENG1", "ING_ENG2", "ING_ENG3", "ING_ENG4"] if c in df.columns]
    if eng_ing:
        df["ANY_ENGINE_INGESTION"] = (df[eng_ing].sum(axis=1) > 0).astype(float)

    eng_dam = [c for c in ["DAM_ENG1", "DAM_ENG2", "DAM_ENG3", "DAM_ENG4"] if c in df.columns]
    if eng_dam:
        df["ENGINES_DAMAGED"] = df[eng_dam].sum(axis=1)

    if "AOS" in df.columns:
        df["LOG_AOS"] = np.log1p(pd.to_numeric(df["AOS"], errors="coerce").fillna(0))

    return df


def _engineer_physics_and_targets(df):
    if "SIZE" in df.columns and "SPEED" in df.columns:
        size_clean = df["SIZE"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        ke = size_clean.map(_SIZE_MASS_MAP).fillna(1) * pd.to_numeric(df["SPEED"], errors="coerce").fillna(0) ** 2
        df["LOG_KINETIC_ENERGY"] = np.log1p(ke)

    df["TOTAL_COST"] = (
        pd.to_numeric(df.get("COST_REPAIRS_INFL_ADJ", 0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("COST_OTHER_INFL_ADJ",   0), errors="coerce").fillna(0)
    )
    df["HAS_DAMAGE"] = np.where(df["TOTAL_COST"] > 0, 1, 0)

    if "WARNED" in df.columns:
        df["WARNED_FLAG"] = np.where(
            df["WARNED"].astype(str).str.strip().str.upper() == "YES", 1, 0
        )

    return df


def _select_features(df):
    keep = _STAGE1_FEATURES + _STAGE2_EXTRA_FEATURES + ["HAS_DAMAGE", "TOTAL_COST"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def clean_and_engineer(file_path):
    print("🚀 Loading raw data...")
    df = load_raw_data(file_path)
    df = _rename_columns(df)
    df = _apply_filters(df)
    df = _engineer_flight_context(df)
    df = _engineer_strike_counts(df)
    df = _engineer_temporal(df)
    df = _engineer_environmental(df)
    df = _engineer_damage_features(df)
    print("⚙️ Engineering physics and target features...")
    df = _engineer_physics_and_targets(df)
    df = _select_features(df)
    return df


def split_and_impute(df):
    print("✂️ Splitting data to prevent information leakage...")
    target_stage1 = df["HAS_DAMAGE"]
    target_stage2 = df["TOTAL_COST"]
    features = df.drop(columns=["HAS_DAMAGE", "TOTAL_COST"])

    # Separate feature spaces before splitting so indices stay aligned.
    stage2_extra_present = [c for c in _STAGE2_EXTRA_FEATURES if c in features.columns]
    features_s1 = features.drop(columns=stage2_extra_present)
    features_s2 = features.copy()

    # Single split — same indices used for both feature sets.
    X_train_s1, X_test_s1, y_train_s1, y_test_s1, y_train_s2, y_test_s2 = train_test_split(
        features_s1, target_stage1, target_stage2,
        test_size=0.20, random_state=42, stratify=target_stage1,
    )
    X_train_s2_all = features_s2.loc[X_train_s1.index].copy()
    X_test_s2_all  = features_s2.loc[X_test_s1.index].copy()

    # -------------------------------------------------------------------------
    # Missingness indicators (before imputation overwrites UNKNOWN)
    # -------------------------------------------------------------------------
    MISSINGNESS_FLAG_THRESHOLD = 5.0
    missingness_cols = ["HEIGHT_BIN", "PHASE_GROUP", "TIME_BUCKET", "SKY", "NUM_STRUCK_CAT", "NUM_SEEN_CAT"]

    print("🚩 Creating missingness indicators before imputation...")
    print(f"\n{'Column':<20} {'Missing (n)':>12} {'Missing (%)':>12} {'Flag':>8}")
    print("-" * 56)
    active_missingness_cols = []
    for col in missingness_cols:
        if col in X_train_s1.columns:
            is_unknown  = X_train_s1[col].fillna("UNKNOWN").astype(str).str.upper() == "UNKNOWN"
            n_missing   = is_unknown.sum()
            pct_missing = 100 * n_missing / len(X_train_s1)
            keep        = pct_missing >= MISSINGNESS_FLAG_THRESHOLD
            print(f"{col:<20} {n_missing:>12,} {pct_missing:>11.1f}%  {'✓' if keep else '✗ dropped'}")
            if keep:
                active_missingness_cols.append(col)
    print()

    for col in active_missingness_cols:
        flag = f"{col}_MISSING"
        for X_train, X_test in [(X_train_s1, X_test_s1), (X_train_s2_all, X_test_s2_all)]:
            X_train[flag] = (X_train[col].fillna("UNKNOWN").astype(str).str.upper() == "UNKNOWN").astype(int)
            X_test[flag]  = (X_test[col].fillna("UNKNOWN").astype(str).str.upper() == "UNKNOWN").astype(int)

    # -------------------------------------------------------------------------
    # Categorical imputation: training-set mode
    # -------------------------------------------------------------------------
    print("🩹 Imputing 'UNKNOWN' categories and missing values...")
    cat_cols = [
        "AC_MASS", "HEIGHT_BIN", "PHASE_GROUP", "OPERATOR_CAT", "ENGINE_TYPE", "FAAREGION",
        "TIME_BUCKET", "SKY", "BIRD_SEASON", "NUM_STRUCK_CAT", "NUM_SEEN_CAT",
    ]
    for col in cat_cols:
        for X_train, X_test in [(X_train_s1, X_test_s1), (X_train_s2_all, X_test_s2_all)]:
            if col in X_train.columns:
                X_train[col] = X_train[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
                X_test[col]  = X_test[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
                valid = X_train[X_train[col] != "UNKNOWN"]
                if not valid.empty:
                    mode = valid[col].mode().iloc[0]
                    X_train[col] = X_train[col].replace("UNKNOWN", mode)
                    X_test[col]  = X_test[col].replace("UNKNOWN", mode)

    # -------------------------------------------------------------------------
    # Numerical imputation: training-set median
    # -------------------------------------------------------------------------
    num_cols_s1 = [
        "SPEED", "LATITUDE", "LONGITUDE", "NUM_ENGS", "MONTH",
        "LOG_KINETIC_ENERGY", "HAS_PRECIP", "IS_MULTI_STRIKE",
    ]
    for col in num_cols_s1:
        for X_train, X_test in [(X_train_s1, X_test_s1), (X_train_s2_all, X_test_s2_all)]:
            if col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")
                median = X_train[col].median()
                X_train[col] = X_train[col].fillna(median)
                X_test[col]  = X_test[col].fillna(median)

    # Stage 2 binary/numeric cols are already 0-filled; included defensively.
    for col in stage2_extra_present:
        if col in X_train_s2_all.columns:
            X_train_s2_all[col] = pd.to_numeric(X_train_s2_all[col], errors="coerce").fillna(0)
            X_test_s2_all[col]  = pd.to_numeric(X_test_s2_all[col],  errors="coerce").fillna(0)

    # -------------------------------------------------------------------------
    # One-hot encoding
    # -------------------------------------------------------------------------
    print("🔀 One-Hot Encoding categorical variables...")
    existing_cat_s1 = [c for c in cat_cols if c in X_train_s1.columns]
    existing_cat_s2 = [c for c in cat_cols if c in X_train_s2_all.columns]

    X_train_s1_enc     = pd.get_dummies(X_train_s1,     columns=existing_cat_s1, drop_first=True)
    X_test_s1_enc      = pd.get_dummies(X_test_s1,      columns=existing_cat_s1, drop_first=True)
    X_train_s2_all_enc = pd.get_dummies(X_train_s2_all, columns=existing_cat_s2, drop_first=True)
    X_test_s2_all_enc  = pd.get_dummies(X_test_s2_all,  columns=existing_cat_s2, drop_first=True)

    X_train_s1_enc,     X_test_s1_enc     = X_train_s1_enc.align(X_test_s1_enc,         join="left", axis=1, fill_value=0)
    X_train_s2_all_enc, X_test_s2_all_enc = X_train_s2_all_enc.align(X_test_s2_all_enc, join="left", axis=1, fill_value=0)

    # -------------------------------------------------------------------------
    # Stage 2 matrix: conditional on damage having occurred
    # -------------------------------------------------------------------------
    train_mask         = y_train_s2 > 0
    X_train_stage2     = X_train_s2_all_enc[train_mask]
    y_train_stage2_log = np.log1p(y_train_s2[train_mask])

    return (
        X_train_s1_enc,    # Stage 1 train
        X_test_s1_enc,     # Stage 1 test  (also used for unconditional hurdle eval)
        y_train_s1,
        y_test_s1,
        X_train_stage2,    # Stage 2 train (damage-confirmed rows, expanded feature set)
        y_train_stage2_log,
        y_test_s2,
        X_test_s2_all_enc, # Stage 2 test  (expanded feature set, all rows)
    )


def package_matrices(matrices, out_dir):
    print("📦 Packaging matrices into ZIP...")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "modeling_matrices_package.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in matrices.items():
            buf = io.StringIO()
            if isinstance(df, pd.Series):
                df = df.to_frame(name)
            df.to_csv(buf, index=False)
            z.writestr(f"{name}.csv", buf.getvalue())

    print(f"✅ Data cleaning complete! Modeling matrices saved to {zip_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir  = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "Public.xlsx"
    out_dir   = base_dir / "data" / "processed" / "modeling"

    df_clean = clean_and_engineer(data_path)
    (
        X_train_s1, X_test_unf,
        y_train_s1, y_test_s1,
        X_train_s2, y_train_s2_log,
        y_test_s2_raw, X_test_s2,
    ) = split_and_impute(df_clean)

    matrices = {
        "X_train_stage1":     X_train_s1,
        "X_test_unfiltered":  X_test_unf,
        "y_train_stage1":     y_train_s1,
        "y_test_stage1":      y_test_s1,
        "X_train_stage2":     X_train_s2,
        "X_test_stage2":      X_test_s2,
        "y_train_stage2_log": y_train_s2_log,
        "y_test_stage2_raw":  y_test_s2_raw,
    }

    package_matrices(matrices, out_dir)