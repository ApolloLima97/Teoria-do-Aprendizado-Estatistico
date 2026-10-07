import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Pasta onde o script está localizado
PASTA_SCRIPT = Path(__file__).resolve().parent

# Caminho do dataset, relativo à pasta do script
CAMINHO_DATASET = PASTA_SCRIPT / "netflix_titles.csv"


def carregar_dados(caminho: str = None) -> pd.DataFrame:
    if caminho is None:
        caminho = CAMINHO_DATASET
    df = pd.read_csv(caminho)
    return df


def bootstrap_dataset(df: pd.DataFrame, n_amostras: int, seed: int = 42) -> pd.DataFrame:
    """Gera um dataset simulado maior a partir da amostra original,
    usando reamostragem com reposição (bootstrap).

    n_amostras: tamanho do dataset simulado desejado (ex: 5000 linhas).
    """
    df_bootstrap = df.sample(n=n_amostras, replace=True, random_state=seed)
    df_bootstrap = df_bootstrap.reset_index(drop=True)
    return df_bootstrap


def extrair_duracao_minutos(valor):
    """Extrai o número de minutos de strings como '90 min'.
    Para séries ('1 Season', '2 Seasons'), retorna None (serão tratadas à parte)."""
    if pd.isna(valor):
        return None
    match = re.search(r"(\d+)\s*min", str(valor))
    if match:
        return int(match.group(1))
    return None


def extrair_temporadas(valor):
    #Extrai o número de temporadas de strings como '2 Seasons'.
    if pd.isna(valor):
        return None
    match = re.search(r"(\d+)\s*Season", str(valor))
    if match:
        return int(match.group(1))
    return None


def gerar_histograma_data_lancamento(df: pd.DataFrame, periodo: str = "Q"):
    # periodo: "M" (mês), "Q" (trimestre) ou "Y" (ano)
    datas = pd.to_datetime(df["date_added"], format="%d/%m/%Y %H:%M", errors="coerce")

    if datas.isna().all():
        datas = pd.to_datetime(df["date_added"], dayfirst=True, errors="coerce")

    datas = datas.dropna()

    plt.figure(figsize=(10, 6))
    datas.dt.to_period(periodo).value_counts().sort_index().plot(kind="bar", width=1, edgecolor="black")
    plt.title(f"Histograma - Data de Lançamento ({periodo})")
    plt.xlabel("Período")
    plt.ylabel("Quantidade de títulos")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("histograma_data_lancamento.png", dpi=150)
    plt.show()


def gerar_histograma_duracao(df: pd.DataFrame):
    df = df.copy()
    df["duracao_min"] = df["duration"].apply(extrair_duracao_minutos)
    df["temporadas"] = df["duration"].apply(extrair_temporadas)

    filmes = df["duracao_min"].dropna()
    series = df["temporadas"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma de duração dos filmes (em minutos)
    if not filmes.empty:
        axes[0].hist(filmes, bins=20, color="steelblue", edgecolor="black")
    axes[0].set_title("Duração dos Filmes (minutos)")
    axes[0].set_xlabel("Minutos")
    axes[0].set_ylabel("Quantidade de filmes")

    # Histograma de número de temporadas das séries
    if not series.empty:
        bins = range(1, int(series.max()) + 2)
        axes[1].hist(series, bins=bins, color="indianred", edgecolor="black", align="left")
    axes[1].set_title("Duração das Séries (nº de temporadas)")
    axes[1].set_xlabel("Temporadas")
    axes[1].set_ylabel("Quantidade de séries")

    plt.tight_layout()
    plt.savefig("histograma_duracao.png", dpi=150)
    plt.show()


def main():
    # Sem argumento, usa o netflix_titles.csv da mesma pasta do script
    caminho = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_DATASET
    # Tamanho do dataset simulado via bootstrap (pode ajustar esse valor)
    tamanho_bootstrap = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    df = carregar_dados(caminho)
    print(f"Dataset original: {len(df)} linhas")

    df_bootstrap = bootstrap_dataset(df, tamanho_bootstrap)
    print(f"Dataset simulado (bootstrap): {len(df_bootstrap)} linhas")

    gerar_histograma_data_lancamento(df_bootstrap)
    gerar_histograma_duracao(df_bootstrap)

    print("Histogramas salvos como 'histograma_data_lancamento.png' e 'histograma_duracao.png'.")


if __name__ == "__main__":
    main()