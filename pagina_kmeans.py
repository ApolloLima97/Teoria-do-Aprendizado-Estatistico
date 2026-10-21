import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

from kmeans import (achar_cotovelo, coeficiente_silhueta, estatistica_gap, gerar_grafico_cotovelo,
                    gerar_grafico_gap, melhor_k_gap, metodo_cotovelo, preparar_matriz)
from svm import carregar_dados, preparar_dados

dados = carregar_dados()
X_completo, y = preparar_dados(dados)

st.title("Clusterizacao K-means")
st.write("Agrupa os titulos por semelhanca e usa o metodo do cotovelo e a estatistica gap "
         "para escolher o numero de grupos (k).")

st.sidebar.header("Parametros")
variaveis = st.sidebar.multiselect("Variaveis", list(X_completo.columns), default=list(X_completo.columns))
k_max = st.sidebar.slider("k maximo testado", 4, 15, 10)
n_referencias = st.sidebar.slider("Simulacoes do gap (B)", 5, 30, 10, step=5)

if len(variaveis) < 2:
    st.warning("Escolha pelo menos duas variaveis.")
    st.stop()


# Guarda os resultados para nao recalcular a cada clique
@st.cache_data
def calcular(variaveis, k_max, n_referencias):
    Xp, X = preparar_matriz(dados, variaveis)
    cotovelo = metodo_cotovelo(Xp, k_max)
    gap = estatistica_gap(Xp, k_max, n_referencias)
    silhueta = coeficiente_silhueta(Xp, k_max)
    return Xp, X, cotovelo, gap, silhueta


with st.spinner("Rodando o K-means para cada k..."):
    Xp, X, cotovelo, gap, silhueta = calcular(tuple(variaveis), k_max, n_referencias)

k_cotovelo = achar_cotovelo(cotovelo["k"], cotovelo["inercia"])
k_gap = melhor_k_gap(gap)
k_silhueta = int(silhueta.loc[silhueta["silhueta"].idxmax(), "k"])

col1, col2, col3 = st.columns(3)
col1.metric("k pelo cotovelo", k_cotovelo)
col2.metric("k pela estatistica gap", k_gap)
col3.metric("k pela silhueta", k_silhueta)
col3.caption(f"silhueta media = {silhueta['silhueta'].max():.2f}")

st.subheader("Metodo do cotovelo")
st.write("Inercia (WCSS) = soma das distancias ao quadrado de cada titulo ate o centro do seu grupo. "
         "O k ideal e onde a curva para de cair rapido.")
fig, ax = plt.subplots(figsize=(9, 4.5))
gerar_grafico_cotovelo(cotovelo, k_cotovelo, ax)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Estatistica gap")
st.write("Compara a dispersao dos dados reais com dados aleatorios sem grupos. "
         "Escolhe-se o menor k em que Gap(k) >= Gap(k+1) - s(k+1). As barras mostram o erro-padrao s(k).")
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
gerar_grafico_gap(gap, k_gap, ax2)
plt.tight_layout()
st.pyplot(fig2)

st.subheader("Coeficiente de silhueta")
st.write("Vai de -1 a 1: quanto mais alto, mais separados e coesos estao os grupos.")
fig3, ax3 = plt.subplots(figsize=(9, 4))
barras = ax3.bar(silhueta["k"], silhueta["silhueta"], color="steelblue", edgecolor="black")
barras[k_silhueta - 2].set_color("indianred")
ax3.bar_label(barras, labels=[f"{s:.2f}" for s in silhueta["silhueta"]])
ax3.set_xticks(silhueta["k"])
ax3.set_xlabel("Numero de clusters (k)")
ax3.set_ylabel("Silhueta media")
plt.tight_layout()
st.pyplot(fig3)

with st.expander("Tabela com os valores"):
    tabela = cotovelo.merge(gap, on="k").merge(silhueta, on="k", how="left")
    st.dataframe(tabela.round(3), hide_index=True)

st.subheader("Grupos encontrados")
k_final = st.select_slider("k usado no agrupamento", options=list(range(2, k_max + 1)),
                           value=max(2, k_cotovelo))


@st.cache_data
def agrupar(variaveis, k):
    Xp, _ = preparar_matriz(dados, variaveis)
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xp)


grupos = agrupar(tuple(variaveis), k_final)
resumo = X.assign(grupo=grupos, tipo=y.loc[X.index].map({"Movie": "Filme", "TV Show": "Serie"}))

perfil = resumo.groupby("grupo")[variaveis].mean().round(2)
perfil.insert(0, "titulos", resumo["grupo"].value_counts().sort_index())
perfil["% series"] = (resumo.groupby("grupo")["tipo"].apply(lambda t: (t == "Serie").mean()) * 100).round(1)
st.write("Media de cada variavel por grupo:")
st.dataframe(perfil)

if len(variaveis) >= 2:
    vx = st.selectbox("Eixo X", variaveis, index=0)
    vy = st.selectbox("Eixo Y", variaveis, index=min(len(variaveis) - 1, 3))
    amostra = resumo.sample(min(2000, len(resumo)), random_state=42)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    cores = plt.get_cmap("tab10")
    for g in sorted(amostra["grupo"].unique()):
        pontos = amostra[amostra["grupo"] == g]
        ax4.scatter(pontos[vx], pontos[vy], s=12, alpha=0.6, color=cores(g % 10), label=f"Grupo {g}")
    ax4.set_xlabel(vx)
    ax4.set_ylabel(vy)
    ax4.legend()
    plt.tight_layout()
    st.pyplot(fig4)

st.write(
    f"Com as variaveis escolhidas, o cotovelo sugere k = {k_cotovelo}, a estatistica gap sugere "
    f"k = {k_gap} e a silhueta sugere k = {k_silhueta}. Quando os metodos discordam, vale olhar "
    "o perfil de cada grupo e escolher o k que gera grupos mais faceis de interpretar."
)
if k_gap == 1:
    st.info(
        "A estatistica gap indicou k = 1: pelo criterio dela, os titulos nao formam grupos claramente "
        "separados (nao ha ganho significativo em relacao a dados aleatorios). Isso combina com a "
        f"silhueta baixa ({silhueta['silhueta'].max():.2f}). O agrupamento acima ainda pode ser util "
        "para segmentar os titulos, mas os grupos tem fronteiras pouco nitidas."
    )
