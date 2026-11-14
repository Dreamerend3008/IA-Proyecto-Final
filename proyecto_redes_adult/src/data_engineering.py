"""
Ingeniería de datos - Dataset Adult (Census Income), UCI.

Pipeline:
  1. load_raw()          -> une adult.data + adult.test (limpia el formato del .test)
  2. clean()             -> espacios, '?' -> NaN, duplicados, etiqueta binaria
  3. build_features()    -> imputación, dummies, normalización (SIN data leakage)
  4. make_splits()       -> particiones train/test con distintas proporciones

Uso rápido:
    from src.data_engineering import load_raw, clean, build_features
    df = clean(load_raw())
    X_train, X_test, y_train, y_test, info = build_features(df, test_size=0.2)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "education", "marital_status", "occupation",
               "relationship", "race", "sex", "native_country"]
TARGET = "target"  # 1 = ">50K", 0 = "<=50K"
RANDOM_STATE = 42


# --------------------------------------------------------------------------- 1
def load_raw() -> pd.DataFrame:
    """Lee adult.data y adult.test y los unifica (48.842 filas esperadas)."""
    train = pd.read_csv(RAW / "adult.data", names=COLUMNS, header=None,
                        skipinitialspace=True, na_values="?")
    # adult.test trae una primera línea basura ('|1x3 Cross validator') -> skiprows=1
    # y las etiquetas terminan en punto ('<=50K.', '>50K.').
    test = pd.read_csv(RAW / "adult.test", names=COLUMNS, header=None,
                       skiprows=1, skipinitialspace=True, na_values="?")
    train["origen"] = "train_original"
    test["origen"] = "test_original"
    return pd.concat([train, test], ignore_index=True)


# --------------------------------------------------------------------------- 2
def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Limpieza básica: strings, etiqueta binaria, duplicados."""
    df = df.copy()

    # Quitar espacios y el '.' final de la etiqueta del .test
    for col in CATEGORICAL + ["income"]:
        df[col] = df[col].astype("string").str.strip()
    df["income"] = df["income"].str.rstrip(".")

    # Variable objetivo binaria
    df[TARGET] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns="income")

    # education y education_num son la misma información (redundante)
    # -> se conserva education_num (ordinal) y se elimina education.
    df = df.drop(columns="education")

    if drop_duplicates:
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "origen"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- 3
def build_features(df: pd.DataFrame, test_size: float = 0.2,
                   random_state: int = RANDOM_STATE, clip_outliers: bool = True,
                   drop_fnlwgt: bool = True):
    """
    Devuelve X_train, X_test, y_train, y_test (DataFrames/Series) e `info` (dict).

    IMPORTANTE (data leakage): primero se parte train/test y DESPUÉS se calculan
    modas, límites de outliers y media/desviación SOLO con train.
    """
    df = df.drop(columns="origen", errors="ignore").copy()
    if drop_fnlwgt:
        # fnlwgt es un peso muestral del censo, no una característica de la persona.
        df = df.drop(columns="fnlwgt")

    num = [c for c in NUMERIC if c in df.columns and c != "education_num"] + ["education_num"]
    num = list(dict.fromkeys(num))
    cat = [c for c in CATEGORICAL if c != "education" and c in df.columns]

    y = df[TARGET]
    X = df.drop(columns=TARGET)

    # Partición ESTRATIFICADA (mantiene la proporción ~76/24 en ambos conjuntos)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_tr, X_te = X_tr.copy(), X_te.copy()

    # --- Nulos: moda de TRAIN para workclass, occupation, native_country
    modas = {c: X_tr[c].mode()[0] for c in cat}
    for c in cat:
        X_tr[c] = X_tr[c].fillna(modas[c])
        X_te[c] = X_te[c].fillna(modas[c])

    # --- Outliers: winsorización con percentiles 1 y 99 de TRAIN (no se borran filas)
    limites = {}
    if clip_outliers:
        for c in ["capital_gain", "capital_loss", "hours_per_week", "age"]:
            lo, hi = X_tr[c].quantile([0.01, 0.99])
            limites[c] = (float(lo), float(hi))
            X_tr[c] = X_tr[c].clip(lo, hi)
            X_te[c] = X_te[c].clip(lo, hi)
        # capital_gain/loss son casi todo ceros: el p99 puede ser muy bajo, se
        # transforman con log1p en vez de recortar demasiado.
    for c in ["capital_gain", "capital_loss"]:
        X_tr[c] = np.log1p(X_tr[c])
        X_te[c] = np.log1p(X_te[c])

    # --- Dummies (one-hot) con las mismas columnas en train y test
    X_tr = pd.get_dummies(X_tr, columns=cat, dtype=float)
    X_te = pd.get_dummies(X_te, columns=cat, dtype=float)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

    # --- Normalización (z-score) SOLO de columnas numéricas, ajustada en TRAIN
    scaler = StandardScaler().fit(X_tr[num])
    X_tr[num] = scaler.transform(X_tr[num])
    X_te[num] = scaler.transform(X_te[num])

    info = {"numericas": num, "categoricas": cat, "modas": modas,
            "limites_outliers": limites, "scaler": scaler,
            "n_features": X_tr.shape[1], "test_size": test_size}
    return X_tr, X_te, y_tr, y_te, info


# --------------------------------------------------------------------------- 4
def make_splits(df: pd.DataFrame, test_sizes=(0.1, 0.2, 0.3, 0.4), save: bool = True):
    """Genera varias particiones (para el punto 4 del enunciado) y las guarda."""
    out = {}
    for ts in test_sizes:
        X_tr, X_te, y_tr, y_te, info = build_features(df, test_size=ts)
        out[ts] = (X_tr, X_te, y_tr, y_te, info)
        if save:
            tag = f"{int(round((1 - ts) * 100))}_{int(round(ts * 100))}"
            PROCESSED.mkdir(parents=True, exist_ok=True)
            pd.concat([X_tr, y_tr], axis=1).to_csv(PROCESSED / f"train_{tag}.csv", index=False)
            pd.concat([X_te, y_te], axis=1).to_csv(PROCESSED / f"test_{tag}.csv", index=False)
    return out


if __name__ == "__main__":
    raw = load_raw()
    print("Filas unificadas:", raw.shape)
    df = clean(raw)
    print("Tras limpieza  :", df.shape)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "adult_clean.csv", index=False)  # dataset limpio (sin escalar)
    splits = make_splits(df)
    for ts, (X_tr, X_te, y_tr, y_te, info) in splits.items():
        print(f"test={ts:.0%} -> train {X_tr.shape}, test {X_te.shape}, "
              f"%>50K train={y_tr.mean():.3f} test={y_te.mean():.3f}")
