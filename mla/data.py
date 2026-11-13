from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "output"

N_SPLITS = 5
SEED = 42

TARGET_MAP = {
    "dataset_1000_hypothyroid": "binaryClass",
    "dataset_1002_ipums_la_98-small": "binaryClass",
    "dataset_1004_synthetic_control": "binaryClass",
    "dataset_1013_analcatdata_challenger": "Damaged",
    "dataset_1014_analcatdata_dmft": "binaryClass",
    "dataset_1016_vowel": "binaryClass",
    "dataset_1018_ipums_la_99-small": "binaryClass",
    "dataset_1020_mfeat-karhunen": "binaryClass",
    "dataset_1021_page-blocks": "binaryClass",
    "dataset_1022_mfeat-pixel": "binaryClass",
    "dataset_1023_soybean": "binaryClass",
    "dataset_1039_hiva_agnostic": "label",
    "dataset_1045_kc1-top5": "DL",
    "dataset_1049_pc4": "c",
    "dataset_1050_pc3": "c",
    "dataset_1056_mc1": "c",
    "dataset_1059_ar1": "defects",
    "dataset_1061_ar4": "defects",
    "dataset_1064_ar6": "defects",
    "dataset_1065_kc3": "c",
    "dataset_311_oil_spill": "class",
    "dataset_312_scene": "Urban",
    "dataset_316_yeast_ml8": "class14",
    "dataset_38_sick": "Class",
    "dataset_450_analcatdata_lawsuit": "Laid.off",
    "dataset_463_backache": "col_33",
    "dataset_757_meta": "binaryClass",
    "dataset_764_analcatdata_apnea3": "binaryClass",
    "dataset_765_analcatdata_apnea2": "binaryClass",
    "dataset_767_analcatdata_apnea1": "binaryClass",
    "dataset_865_analcatdata_neavote": "binaryClass",
    "dataset_867_visualizing_livestock": "binaryClass",
    "dataset_875_analcatdata_chlamydia": "binaryClass",
    "dataset_940_water-treatment": "binaryClass",
    "dataset_947_arsenic-male-bladder": "binaryClass",
    "dataset_949_arsenic-female-bladder": "binaryClass",
    "dataset_950_arsenic-female-lung": "binaryClass",
    "dataset_951_arsenic-male-lung": "binaryClass",
    "dataset_954_spectrometer": "binaryClass",
    "dataset_958_segment": "binaryClass",
    "dataset_962_mfeat-morphological": "binaryClass",
    "dataset_966_analcatdata_halloffame": "binaryClass",
    "dataset_968_analcatdata_birthday": "binaryClass",
    "dataset_971_mfeat-fourier": "binaryClass",
    "dataset_976_JapaneseVowels": "binaryClass",
    "dataset_978_mfeat-factors": "binaryClass",
    "dataset_980_optdigits": "binaryClass",
    "dataset_984_analcatdata_draft": "binaryClass",
    "dataset_987_collins": "binaryClass",
    "dataset_995_mfeat-zernike": "binaryClass",
}


def find_target(df, csv_path):
    target = TARGET_MAP.get(csv_path.stem)
    if target and target in df.columns:
        return target
    return df.columns[-1]


def encode_binary(series):
    vals = series.dropna().unique()
    if set(vals) <= {0, 1, 0.0, 1.0, True, False}:
        return series.astype(float)
    str_vals = set(str(v) for v in vals)
    if str_vals == {"-1", "1"}:
        return series.map({-1: 0, 1: 1, "-1": 0, "1": 1}).astype(float)
    mapping = {v: float(i) for i, v in enumerate(sorted(vals, key=str))}
    return series.map(mapping)


def imbalance_ratio(y):
    counts = np.bincount(y.astype(int))
    if len(counts) < 2 or counts.min() == 0:
        return float("inf")
    return round(counts.max() / counts.min(), 2)


def load_dataset(csv_path):
    """Return X, y, target column, and imbalance ratio for one CSV.

    Median imputation uses the whole table, matching the original notebooks.
    """
    df = pd.read_csv(csv_path)
    target_col = find_target(df, csv_path)
    if target_col not in df.columns:
        raise ValueError(f"missing target column {target_col!r}")

    df = df.dropna(subset=[target_col])
    for col in df.columns:
        if col == target_col:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
            df[col] = df[col].replace(-1, np.nan)

    y = encode_binary(df[target_col]).values.astype(float)
    features = df.drop(columns=[target_col])
    X = features.fillna(features.median(numeric_only=True)).values.astype(float)
    if len(np.unique(y)) < 2:
        raise ValueError("target has a single class")
    return X, y, target_col, imbalance_ratio(y)
