import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from bootstrep import carregar_dados, bootstrap_dataset, extrair_duracao_minutos, extrair_temporadas

dados = carregar_dados()

st.title("Metodo Bootstrap")
st.write("Reamostragem com reposicao para gerar um dataset simulado maior a partir da base original.")

st.sidebar.header("Parametros")
tamanho = st.sidebar.slider("Tamanho do dataset simulado", 1000, 20000, 5000, step=1000)
periodo = st.sidebar.selectbox("Periodo do histograma", ["Ano", "Trimestre"])

simulado = bootstrap_dataset(dados, tamanho)

col1, col2 = st.columns(2)
col1.metric("Dataset original", f"{len(dados)} linhas")
col2.metric("Dataset simulado", f"{len(simulado)} linhas")

# Compara a proporcao de filmes e series nas duas bases
st.subheader("Proporcao de filmes e series")
proporcao = pd.DataFrame({
    "Original": dados["type"].value_counts(normalize=True),
    "Bootstrap": simulado["type"].value_counts(normalize=True),
}).rename(index={"Movie": "Filme", "TV Show": "Serie"})
st.dataframe((proporcao * 100).round(1).astype(str) + "%")

st.subheader("Histograma - data de adicao na Netflix")
datas = pd.to_datetime(simulado["date_added"].str.strip(), errors="coerce").dropna()
codigo = "Y" if periodo == "Ano" else "Q"
contagem = datas.dt.to_period(codigo).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
contagem.plot(kind="bar", width=1, edgecolor="black", ax=ax, color="steelblue")
ax.set_xlabel("Periodo")
ax.set_ylabel("Quantidade de titulos")
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Histograma - duracao")
filmes = simulado["duration"].apply(extrair_duracao_minutos).dropna()
series = simulado["duration"].apply(extrair_temporadas).dropna()
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(filmes, bins=20, color="steelblue", edgecolor="black")
axes[0].set_title("Filmes (minutos)")
axes[0].set_xlabel("Minutos")
axes[0].set_ylabel("Quantidade")
axes[1].hist(series, bins=range(1, int(series.max()) + 2), color="indianred",
             edgecolor="black", align="left")
axes[1].set_title("Series (temporadas)")
axes[1].set_xlabel("Temporadas")
plt.tight_layout()
st.pyplot(fig2)

st.write(
    f"O dataset simulado tem {len(simulado)} titulos, com media de {filmes.mean():.1f} minutos "
    f"por filme e {series.mean():.1f} temporadas por serie."
)
