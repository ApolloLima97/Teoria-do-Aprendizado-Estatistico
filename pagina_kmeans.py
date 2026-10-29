import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from kmeans import (achar_cotovelo, coeficiente_silhueta, estatistica_gap, gerar_grafico_cotovelo,
                    gerar_grafico_gap, melhor_k_gap, metodo_cotovelo, preparar_matriz)
from svm import carregar_dados, preparar_dados

dados = carregar_dados()
X_completo, y = preparar_dados(dados)

# Nome legivel de cada variavel (usado nos eixos, tabelas e legendas)
NOMES = {
    "ano_lancamento": "Ano de lancamento",
    "ano_adicionado": "Ano em que entrou na Netflix",
    "tem_diretor": "Tem diretor cadastrado (0 = nao, 1 = sim)",
    "qtd_elenco": "Quantidade de atores no elenco",
    "qtd_paises": "Quantidade de paises de producao",
    "qtd_generos": "Quantidade de generos",
}
# Versao curta, usada para descrever os eixos do PCA
CURTOS = {
    "ano_lancamento": "ano de lancamento",
    "ano_adicionado": "ano de entrada na Netflix",
    "tem_diretor": "ter diretor",
    "qtd_elenco": "atores no elenco",
    "qtd_paises": "paises de producao",
    "qtd_generos": "generos",
}
ORIGEM = {
    "ano_lancamento": "coluna release_year",
    "ano_adicionado": "ano da coluna date_added",
    "tem_diretor": "1 se a coluna director esta preenchida",
    "qtd_elenco": "nomes separados por virgula na coluna cast",
    "qtd_paises": "paises separados por virgula na coluna country",
    "qtd_generos": "generos separados por virgula na coluna listed_in",
}

st.title("Clusterizacao K-means")
st.write("Agrupa os titulos por semelhanca e usa o metodo do cotovelo e a estatistica gap "
         "para escolher o numero de grupos (k).")

with st.expander("Quais dados estao sendo usados", expanded=True):
    st.write(f"Cada linha e um titulo da Netflix ({len(X_completo)} titulos do netflix_titles.csv, "
             "sem os que nao tem data de entrada). Para cada titulo o K-means olha estas variaveis:")
    st.dataframe(pd.DataFrame({
        "Variavel": [NOMES[c] for c in X_completo.columns],
        "Como foi calculada": [ORIGEM[c] for c in X_completo.columns],
        "Faixa de valores": [f"{a} a {b}" for a, b in zip(X_completo.min(), X_completo.max())],
        "Media": X_completo.mean().round(1).values,
    }), hide_index=True)
    st.caption("O tipo (Filme ou Serie) NAO entra no agrupamento: ele so aparece depois, para conferir os grupos.")

st.sidebar.header("Parametros")
variaveis = st.sidebar.multiselect("Variaveis", list(X_completo.columns), default=list(X_completo.columns),
                                   format_func=lambda c: NOMES[c])
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
resumo = X.assign(grupo=grupos, tipo=y.loc[X.index].map({"Movie": "Filme", "TV Show": "Serie"}),
                  titulo=dados.loc[X.index, "title"])
NOMES_GRUPOS = [f"Grupo {g}" for g in range(k_final)]

perfil = resumo.groupby("grupo")[variaveis].mean().round(2)
perfil.insert(0, "Titulos no grupo", resumo["grupo"].value_counts().sort_index())
perfil["% que sao series"] = (resumo.groupby("grupo")["tipo"].apply(lambda t: (t == "Serie").mean()) * 100).round(1)
perfil.index = NOMES_GRUPOS
st.write("Media de cada variavel por grupo (o perfil de cada grupo):")
st.dataframe(perfil.rename(columns=NOMES))


def tabela_pontos(amostra):
    """Dados mostrados ao passar o mouse: nome do titulo, tipo e valores reais de cada variavel."""
    tabela = pd.DataFrame({
        "Titulo": amostra["titulo"].values,
        "Tipo": amostra["tipo"].values,
        "Grupo": "Grupo " + amostra["grupo"].astype(str).values,
    })
    for v in variaveis:
        tabela[NOMES[v]] = amostra[v].values
    return tabela


def espalhar(valores, rng, tamanho=0.25):
    # As variaveis sao numeros inteiros: sem um pequeno deslocamento os pontos ficariam empilhados
    return valores + rng.uniform(-tamanho, tamanho, size=len(valores))


def grafico_dispersao(tabela, eixos, titulos_eixos, altura):
    funcao = px.scatter_3d if len(eixos) == 3 else px.scatter
    argumentos = dict(zip(["x", "y", "z"], eixos))
    fig = funcao(
        tabela, **argumentos, color="Grupo", height=altura, opacity=0.7,
        hover_name="Titulo", hover_data={c: True for c in ["Tipo"] + [NOMES[v] for v in variaveis]}
        | {e: False for e in eixos},
        category_orders={"Grupo": NOMES_GRUPOS},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    if len(eixos) == 3:
        fig.update_traces(marker_size=3)
        fig.update_layout(scene=dict(xaxis_title=titulos_eixos[0], yaxis_title=titulos_eixos[1],
                                     zaxis_title=titulos_eixos[2],
                                     camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))))
    else:
        fig.update_traces(marker_size=6)
        fig.update_layout(xaxis_title=titulos_eixos[0], yaxis_title=titulos_eixos[1])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_itemsizing="constant",
                      legend_title_text="Grupo (K-means)")
    return fig


st.subheader("Dispersao 2D")
amostra = resumo.sample(min(2000, len(resumo)), random_state=42)
c1, c2 = st.columns(2)
vx = c1.selectbox("Eixo X (horizontal)", variaveis, index=0, format_func=lambda c: NOMES[c])
vy = c2.selectbox("Eixo Y (vertical)", variaveis, index=min(len(variaveis) - 1, 3),
                  format_func=lambda c: NOMES[c])
st.caption(f"Cada ponto e um titulo ({len(amostra)} sorteados dos {len(resumo)}, para o grafico nao ficar pesado). "
           "A cor e o grupo em que o K-means colocou o titulo. Passe o mouse num ponto para ver qual titulo e.")
rng = np.random.default_rng(42)
tabela2d = tabela_pontos(amostra)
tabela2d["_x"] = espalhar(amostra[vx].values, rng)
tabela2d["_y"] = espalhar(amostra[vy].values, rng)
st.plotly_chart(grafico_dispersao(tabela2d, ["_x", "_y"], [NOMES[vx], NOMES[vy]], 500))

st.subheader("Dispersao 3D")
if len(variaveis) < 3:
    st.warning("Escolha pelo menos tres variaveis na barra lateral para ver o grafico 3D.")
else:
    modo = st.radio("O que vai nos eixos", ["Tres variaveis a escolha", "Resumo de todas as variaveis (PCA)"],
                    horizontal=True)
    amostra3d = resumo.sample(min(3000, len(resumo)), random_state=42)
    tabela3d = tabela_pontos(amostra3d)
    rng = np.random.default_rng(7)

    if modo == "Tres variaveis a escolha":
        c1, c2, c3 = st.columns(3)
        ex = c1.selectbox("Eixo X", variaveis, index=0, format_func=lambda c: NOMES[c])
        ey = c2.selectbox("Eixo Y", variaveis, index=1, format_func=lambda c: NOMES[c])
        ez = c3.selectbox("Eixo Z (altura)", variaveis, index=min(3, len(variaveis) - 1),
                          format_func=lambda c: NOMES[c])
        if len({ex, ey, ez}) < 3:
            st.warning("Escolha tres variaveis diferentes.")
            ex, ey, ez = variaveis[:3]
        for eixo, v in zip(["_x", "_y", "_z"], [ex, ey, ez]):
            tabela3d[eixo] = espalhar(amostra3d[v].values, rng)
        titulos = [CURTOS[v][0].upper() + CURTOS[v][1:] for v in (ex, ey, ez)]
        st.caption("Cada ponto e um titulo, posicionado pelos valores reais das tres variaveis escolhidas.")
    else:
        # PCA: cria 3 eixos novos, cada um uma combinacao das variaveis padronizadas
        pca = PCA(n_components=3, random_state=42).fit(Xp)
        componentes = pca.transform(Xp[resumo.index.get_indexer(amostra3d.index)])
        explicada = pca.explained_variance_ratio_
        pesos = pd.DataFrame(pca.components_.T, index=variaveis, columns=["Eixo 1", "Eixo 2", "Eixo 3"])

        def descrever(coluna):
            # As 3 variaveis com maior peso (positivo = cresce junto, negativo = cresce ao contrario)
            principais = pesos[coluna].abs().sort_values(ascending=False).index[:3]
            mais = [CURTOS[v] for v in principais if pesos.loc[v, coluna] > 0]
            menos = [CURTOS[v] for v in principais if pesos.loc[v, coluna] < 0]
            def juntar(itens):
                return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]

            partes = []
            if mais:
                partes.append("mais " + juntar(mais))
            if menos:
                partes.append("menos " + juntar(menos))
            return ", ".join(partes)

        descricoes = [f"Eixo {i + 1}: {descrever(f'Eixo {i + 1}')}" for i in range(3)]
        titulos = [f"Eixo {i + 1}" for i in range(3)]
        for i, eixo in enumerate(["_x", "_y", "_z"]):
            tabela3d[eixo] = componentes[:, i]

        st.write(
            "O PCA (Analise de Componentes Principais) junta as variaveis em 3 eixos novos. "
            "Cada eixo e uma mistura das variaveis originais, e o numero no eixo nao tem unidade: "
            "0 e o titulo medio, valores positivos e negativos indicam o quanto o titulo foge da media."
        )
        st.markdown("\n".join(
            f"- **{descricoes[i]}** (resume {explicada[i]:.0%} da variacao dos dados)" for i in range(3)
        ))
        st.caption(f"Os 3 eixos juntos mostram {explicada.sum():.0%} da informacao das {len(variaveis)} variaveis. "
                   "Na tabela abaixo, quanto maior o peso (positivo ou negativo), mais a variavel influencia o eixo.")
        st.dataframe(pesos.rename(index=NOMES).style.format("{:.2f}")
                     .background_gradient(cmap="RdBu", vmin=-1, vmax=1, axis=None))

    st.plotly_chart(grafico_dispersao(tabela3d, ["_x", "_y", "_z"], titulos, 700))
    st.caption("Arraste para girar, use a roda do mouse para zoom, passe o mouse num ponto para ver o titulo "
               "e clique na legenda para esconder ou mostrar grupos.")

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
