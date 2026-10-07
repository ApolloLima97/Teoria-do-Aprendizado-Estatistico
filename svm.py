import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Pasta onde o script está localizado
PASTA_SCRIPT = Path(__file__).resolve().parent

# Caminho do dataset, relativo à pasta do script
CAMINHO_DATASET = PASTA_SCRIPT / "netflix_titles.csv"


def carregar_dados(caminho=None) -> pd.DataFrame:
    if caminho is None:
        caminho = CAMINHO_DATASET
    return pd.read_csv(caminho)


def preparar_dados(df: pd.DataFrame):
    """Monta as variáveis numéricas (X) e o alvo (y = Filme ou Série)."""
    X = pd.DataFrame()
    X["ano_lancamento"] = df["release_year"]
    X["ano_adicionado"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce").dt.year
    X["tem_diretor"] = df["director"].notna().astype(int)
    X["qtd_elenco"] = df["cast"].fillna("").str.count(",") + df["cast"].notna()
    X["qtd_paises"] = df["country"].fillna("").str.count(",") + df["country"].notna()
    X["qtd_generos"] = df["listed_in"].str.count(",") + 1

    y = df["type"]

    # Remove as linhas sem data de adição
    validos = X.notna().all(axis=1)
    return X[validos], y[validos]


def treinar_svm(X, y, kernel: str = "rbf", C: float = 1.0, seed: int = 42):
    # 70% dos títulos para treino e 30% para teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Padroniza os dados (o SVM usa distâncias) e treina o modelo
    modelo = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C))
    modelo.fit(X_treino, y_treino)

    previsto = modelo.predict(X_teste)
    return y_teste, previsto


def gerar_matriz_confusao(y_teste, previsto):
    matriz = confusion_matrix(y_teste, previsto, labels=["Movie", "TV Show"])
    ConfusionMatrixDisplay(matriz, display_labels=["Filme", "Série"]).plot(cmap="Blues")
    plt.title("SVM - Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("svm_matriz_confusao.png", dpi=150)
    plt.show()


def gerar_comparacao_kernels(X, y):
    """Treina o SVM com cada kernel e compara a acurácia num gráfico de barras."""
    kernels = ["linear", "rbf", "poly"]
    acuracias = []
    for k in kernels:
        y_teste, previsto = treinar_svm(X, y, kernel=k)
        acuracias.append(accuracy_score(y_teste, previsto))
        print(f"Acurácia kernel {k}: {acuracias[-1]:.1%}")

    plt.figure(figsize=(7, 5))
    barras = plt.bar(kernels, acuracias, color="steelblue", edgecolor="black")
    plt.bar_label(barras, labels=[f"{a:.1%}" for a in acuracias])
    plt.ylim(0, 1)
    plt.title("SVM - Acurácia por Kernel")
    plt.xlabel("Kernel")
    plt.ylabel("Acurácia no teste")
    plt.tight_layout()
    plt.savefig("svm_comparacao_kernels.png", dpi=150)
    plt.show()


def gerar_fronteira_decisao(X, y, var_x="ano_lancamento", var_y="qtd_elenco", seed=42):
    """Treina o SVM só com duas variáveis e desenha a região que ele usa
    para separar Filmes e Séries."""
    X2 = X[[var_x, var_y]]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X2, y, test_size=0.3, random_state=seed, stratify=y
    )
    modelo = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    modelo.fit(X_treino, y_treino)
    acuracia = modelo.score(X_teste, y_teste)

    # Grade de pontos cobrindo o gráfico: o modelo classifica cada ponto
    xx, yy = np.meshgrid(
        np.linspace(X2[var_x].min() - 1, X2[var_x].max() + 1, 200),
        np.linspace(X2[var_y].min() - 1, X2[var_y].max() + 1, 200),
    )
    grade = pd.DataFrame({var_x: xx.ravel(), var_y: yy.ravel()})
    zz = (modelo.predict(grade) == "TV Show").reshape(xx.shape)

    # Amostra de pontos do teste (com pequeno deslocamento para não sobrepor)
    amostra = X_teste.assign(tipo=y_teste).sample(1500, random_state=seed)
    ruido = np.random.default_rng(seed).uniform(-0.3, 0.3, size=(len(amostra), 2))

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["steelblue", "indianred"], alpha=0.25)
    for tipo, cor, nome in [("Movie", "steelblue", "Filme"), ("TV Show", "indianred", "Série")]:
        pontos = amostra[amostra["tipo"] == tipo]
        r = ruido[(amostra["tipo"] == tipo).values]
        plt.scatter(pontos[var_x] + r[:, 0], pontos[var_y] + r[:, 1],
                    s=12, alpha=0.6, color=cor, label=nome)
    plt.title(f"SVM - Fronteira de Decisão ({var_x} x {var_y}) - acurácia {acuracia:.1%}")
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    plt.xlim(1960, X2[var_x].max() + 1)
    plt.ylim(-1, 30)
    plt.legend()
    plt.tight_layout()
    plt.savefig("svm_fronteira_decisao.png", dpi=150)
    plt.show()


def main():
    # Sem argumento, usa o netflix_titles.csv da mesma pasta do script
    caminho = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_DATASET
    kernel = sys.argv[2] if len(sys.argv) > 2 else "rbf"

    df = carregar_dados(caminho)
    X, y = preparar_dados(df)
    print(f"Dataset: {len(X)} títulos ({(y == 'Movie').sum()} filmes e {(y == 'TV Show').sum()} séries)")

    y_teste, previsto = treinar_svm(X, y, kernel=kernel)
    print(f"Kernel: {kernel}")
    print(f"Acurácia no teste: {accuracy_score(y_teste, previsto):.1%}")

    gerar_matriz_confusao(y_teste, previsto)
    print("Matriz de confusão salva como 'svm_matriz_confusao.png'.")

    gerar_comparacao_kernels(X, y)
    print("Comparação salva como 'svm_comparacao_kernels.png'.")

    gerar_fronteira_decisao(X, y)
    print("Fronteira salva como 'svm_fronteira_decisao.png'.")


if __name__ == "__main__":
    main()
