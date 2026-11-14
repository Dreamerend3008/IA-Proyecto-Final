# Proyecto 3 – Redes neuronales feed-forward con Adult (Census Income)

Introducción a la Inteligencia Artificial – Pontificia Universidad Javeriana.

## Estructura
```
data/raw/          adult.data, adult.test, adult.names (originales UCI, sin modificar)
data/processed/    adult_clean.csv (dataset unificado y limpio)
src/               data_engineering.py (pipeline), eda.py (gráficas)
reports/figures/   gráficas del análisis exploratorio
notebooks/         notebooks de modelado (Persona de modelos)
```

## Cómo reproducir
```bash
pip install -r requirements.txt
python -m src.data_engineering   # genera adult_clean.csv y particiones train/test (90/10, 80/20, 70/30, 60/40)
python -m src.eda                # genera las figuras
```

## Uso en los modelos
```python
from src.data_engineering import load_raw, clean, build_features
df = clean(load_raw())
X_train, X_test, y_train, y_test, info = build_features(df, test_size=0.2)
# X_*: 88 columnas, listas para la red | y_*: 0 = <=50K, 1 = >50K
```

## Decisiones de ingeniería de datos
- Unión de adult.data + adult.test (48.842 filas); se limpia la 1.ª línea basura del .test y el punto final de las etiquetas.
- '?' -> NaN en workclass, occupation, native_country; se imputan con la moda **de train**.
- Se eliminan 52 duplicados exactos; se elimina `education` (redundante con `education_num`) y `fnlwgt` (peso muestral).
- Outliers: winsorización p1–p99 (límites de train) + log1p en capital_gain/loss. No se borran filas.
- One-hot de categóricas; z-score de numéricas ajustado **solo con train** (sin data leakage).
- Partición **estratificada** (semilla 42): ~23,9 % de >50K en train y test. Clases desbalanceadas -> usar precision, recall y F1, no solo accuracy.
