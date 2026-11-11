"""Análisis exploratorio: genera las gráficas en reports/figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_engineering import ROOT, CATEGORICAL, NUMERIC, TARGET, clean, load_raw

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    df = clean(load_raw())
    nums = [c for c in NUMERIC if c in df.columns]

    # 1. Balance de clases
    plt.figure(figsize=(4, 4))
    ax = sns.countplot(x=TARGET, data=df)
    ax.set_xticks([0, 1], ["<=50K", ">50K"])
    for p in ax.patches:
        ax.annotate(f"{p.get_height() / len(df):.1%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Distribución de la variable objetivo")
    plt.tight_layout(); plt.savefig(FIG / "01_balance_clases.png", dpi=150); plt.close()

    # 2. Histogramas numéricas
    df[nums].hist(bins=30, figsize=(12, 7))
    plt.suptitle("Distribución de variables numéricas")
    plt.tight_layout(); plt.savefig(FIG / "02_hist_numericas.png", dpi=150); plt.close()

    # 3. Boxplots (outliers)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), nums):
        sns.boxplot(y=df[c], ax=ax); ax.set_title(c)
    plt.tight_layout(); plt.savefig(FIG / "03_boxplots_outliers.png", dpi=150); plt.close()

    # 4. Nulos
    nulos = load_raw()[["workclass", "occupation", "native_country"]].isna().mean() * 100
    plt.figure(figsize=(5, 3.5))
    nulos.plot(kind="bar"); plt.ylabel("% de nulos"); plt.title("Valores nulos ('?')")
    plt.tight_layout(); plt.savefig(FIG / "04_nulos.png", dpi=150); plt.close()

    # 5. Categóricas vs objetivo (tasa de >50K)
    cats = [c for c in CATEGORICAL if c in df.columns and c != "native_country"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, c in zip(axes.ravel(), cats):
        (df.groupby(c)[TARGET].mean().sort_values().plot(kind="barh", ax=ax))
        ax.set_title(f"% >50K por {c}"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(FIG / "05_categoricas_vs_objetivo.png", dpi=150); plt.close()

    # 6. Correlación (numéricas + objetivo)
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[nums + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de correlación")
    plt.tight_layout(); plt.savefig(FIG / "06_correlacion.png", dpi=150); plt.close()
    print("Figuras guardadas en", FIG)


if __name__ == "__main__":
    main()
