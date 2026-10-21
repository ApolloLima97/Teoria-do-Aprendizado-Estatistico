import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from svm import carregar_dados, preparar_dados


def preparar_matriz(df: pd.DataFrame, variaveis=None):
    """Usa as mesmas variáveis numéricas do SVM e padroniza (média 0, desvio 1),
    porque o K-means trabalha com distâncias."""
    X, _ = preparar_dados(df)
    if variaveis is not None:
        X = X[list(variaveis)]
    return StandardScaler().fit_transform(X), X


def metodo_cotovelo(X, k_max: int = 10, seed: int = 42) -> pd.DataFrame:
    """Roda o K-means para k = 1..k_max e guarda a inércia (WCSS):
    soma das distâncias ao quadrado de cada ponto até o centróide do seu grupo."""
    inercias = []
    for k in range(1, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        inercias.append(modelo.inertia_)
    return pd.DataFrame({"k": range(1, k_max + 1), "inercia": inercias})


def achar_cotovelo(k, inercias) -> int:
    """Encontra o cotovelo pelo método da maior distância: o ponto da curva
    mais afastado da reta que liga o primeiro ao último ponto."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(inercias, dtype=float)
    # Normaliza os dois eixos para a escala não influenciar a distância
    kn = (k - k.min()) / (k.max() - k.min())
    wn = (w - w.min()) / (w.max() - w.min())
    distancias = np.abs(kn + wn - 1) / np.sqrt(2)  # reta de (0,1) até (1,0)
    return int(k[np.argmax(distancias)])


def estatistica_gap(X, k_max: int = 10, n_referencias: int = 10, seed: int = 42) -> pd.DataFrame:
    """Estatística Gap (Tibshirani, Walther e Hastie, 2001).

    Compara log(Wk) dos dados reais com a média de log(Wk*) em dados aleatórios
    uniformes gerados na mesma faixa de valores (referência sem grupos):
        Gap(k) = média(log Wk*) - log Wk
    """
    rng = np.random.default_rng(seed)
    minimos, maximos = X.min(axis=0), X.max(axis=0)
    referencias = [rng.uniform(minimos, maximos, size=X.shape) for _ in range(n_referencias)]

    linhas = []
    for k in range(1, k_max + 1):
        log_wk = np.log(KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X).inertia_)
        log_ref = np.array([
            np.log(KMeans(n_clusters=k, n_init=3, random_state=seed).fit(ref).inertia_)
            for ref in referencias
        ])
        gap = log_ref.mean() - log_wk
        # Erro-padrão corrigido pelo número de simulações
        s_k = log_ref.std() * np.sqrt(1 + 1 / n_referencias)
        linhas.append({"k": k, "gap": gap, "s_k": s_k})
    return pd.DataFrame(linhas)


def melhor_k_gap(resultado: pd.DataFrame) -> int:
    """Menor k tal que Gap(k) >= Gap(k+1) - s(k+1)."""
    gaps, erros = resultado["gap"].values, resultado["s_k"].values
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - erros[i + 1]:
            return int(resultado["k"].iloc[i])
    return int(resultado["k"].iloc[-1])


def coeficiente_silhueta(X, k_max: int = 10, amostra: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Silhueta média para k = 2..k_max (de -1 a 1; quanto maior, mais separados os grupos).
    Calculada numa amostra de pontos porque o cálculo completo é lento."""
    silhuetas = []
    for k in range(2, k_max + 1):
        rotulos = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        silhuetas.append(silhouette_score(X, rotulos, sample_size=min(amostra, len(X)),
                                          random_state=seed))
    return pd.DataFrame({"k": range(2, k_max + 1), "silhueta": silhuetas})


def gerar_grafico_cotovelo(cotovelo: pd.DataFrame, k_escolhido: int, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cotovelo["k"], cotovelo["inercia"], marker="o", color="steelblue")
    linha = cotovelo[cotovelo["k"] == k_escolhido]
    ax.scatter(linha["k"], linha["inercia"], s=180, facecolors="none",
               edgecolors="indianred", linewidths=2, zorder=3, label=f"Cotovelo: k = {k_escolhido}")
    ax.set_title("Método do Cotovelo")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inércia (WCSS)")
    ax.set_xticks(cotovelo["k"])
    ax.legend()
    return ax


def gerar_grafico_gap(gap: pd.DataFrame, k_escolhido: int, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(gap["k"], gap["gap"], yerr=gap["s_k"], marker="o", color="steelblue",
                ecolor="gray", capsize=4)
    linha = gap[gap["k"] == k_escolhido]
    ax.scatter(linha["k"], linha["gap"], s=180, facecolors="none",
               edgecolors="indianred", linewidths=2, zorder=3, label=f"Gap: k = {k_escolhido}")
    ax.set_title("Estatística Gap")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Gap(k)")
    ax.set_xticks(gap["k"])
    ax.legend()
    return ax


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else None
    k_max = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    Xp, X = preparar_matriz(carregar_dados(caminho))
    print(f"Dataset: {len(X)} títulos, {X.shape[1]} variáveis")

    cotovelo = metodo_cotovelo(Xp, k_max)
    k_cotovelo = achar_cotovelo(cotovelo["k"], cotovelo["inercia"])
    print(f"Cotovelo: k = {k_cotovelo}")

    gap = estatistica_gap(Xp, k_max)
    k_gap = melhor_k_gap(gap)
    print(f"Gap: k = {k_gap}")

    silhueta = coeficiente_silhueta(Xp, k_max)
    k_silhueta = int(silhueta.loc[silhueta["silhueta"].idxmax(), "k"])
    print(f"Silhueta: k = {k_silhueta}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gerar_grafico_cotovelo(cotovelo, k_cotovelo, axes[0])
    gerar_grafico_gap(gap, k_gap, axes[1])
    plt.tight_layout()
    plt.savefig("kmeans_cotovelo_gap.png", dpi=150)
    plt.show()
    print("Gráficos salvos como 'kmeans_cotovelo_gap.png'.")


if __name__ == "__main__":
    main()
