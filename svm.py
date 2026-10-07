import sys
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


if __name__ == "__main__":
    main()
