import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svm import carregar_dados, preparar_dados, treinar_svm

X, y = preparar_dados(carregar_dados())

st.title("Algoritmo SVM")
st.write("Classificacao de cada titulo como Filme ou Serie usando Support Vector Machine.")

st.sidebar.header("Parametros")
kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])


# Guarda os resultados para nao treinar de novo a cada clique
@st.cache_data
def resultado_svm(kernel):
    y_teste, previsto = treinar_svm(X, y, kernel=kernel)
    return y_teste, previsto


y_teste, previsto = resultado_svm(kernel)
acuracia = accuracy_score(y_teste, previsto)

col1, col2, col3 = st.columns(3)
col1.metric("Titulos usados", len(X))
col2.metric("Titulos no teste", len(y_teste))
col3.metric("Acuracia", f"{acuracia:.1%}")

st.subheader("Variaveis usadas")
st.dataframe(X.describe().round(2))

st.subheader("Matriz de confusao")
matriz = confusion_matrix(y_teste, previsto, labels=["Movie", "TV Show"])
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(matriz, display_labels=["Filme", "Serie"]).plot(cmap="Blues", ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Acuracia por kernel")
kernels = ["linear", "rbf", "poly"]
acuracias = [accuracy_score(*resultado_svm(k)) for k in kernels]
fig2, ax2 = plt.subplots(figsize=(7, 4))
barras = ax2.bar(kernels, acuracias, color="steelblue", edgecolor="black")
ax2.bar_label(barras, labels=[f"{a:.1%}" for a in acuracias])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Acuracia no teste")
plt.tight_layout()
st.pyplot(fig2)

st.subheader("Fronteira de decisao (2 variaveis)")
var_x = st.selectbox("Eixo X", list(X.columns), index=0)
var_y = st.selectbox("Eixo Y", list(X.columns), index=3)


@st.cache_data
def fronteira(var_x, var_y, kernel):
    X2 = X[[var_x, var_y]]
    X_treino, X_teste, y_treino, y_teste2 = train_test_split(
        X2, y, test_size=0.3, random_state=42, stratify=y
    )
    modelo = make_pipeline(StandardScaler(), SVC(kernel=kernel))
    modelo.fit(X_treino, y_treino)
    xx, yy = np.meshgrid(
        np.linspace(X2[var_x].min() - 1, X2[var_x].max() + 1, 150),
        np.linspace(X2[var_y].min() - 1, X2[var_y].max() + 1, 150),
    )
    grade = pd.DataFrame({var_x: xx.ravel(), var_y: yy.ravel()})
    zz = (modelo.predict(grade) == "TV Show").reshape(xx.shape)
    return xx, yy, zz, modelo.score(X_teste, y_teste2), X_teste, y_teste2


if var_x == var_y:
    st.warning("Escolha duas variaveis diferentes.")
else:
    xx, yy, zz, acc2, X_teste2, y_teste2 = fronteira(var_x, var_y, kernel)
    amostra = X_teste2.assign(tipo=y_teste2).sample(1500, random_state=42)
    ruido = np.random.default_rng(42).uniform(-0.3, 0.3, size=(len(amostra), 2))

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["steelblue", "indianred"], alpha=0.25)
    for tipo, cor, nome in [("Movie", "steelblue", "Filme"), ("TV Show", "indianred", "Serie")]:
        filtro = (amostra["tipo"] == tipo).values
        ax3.scatter(amostra.loc[filtro, var_x] + ruido[filtro, 0],
                    amostra.loc[filtro, var_y] + ruido[filtro, 1],
                    s=12, alpha=0.6, color=cor, label=nome)
    ax3.set_xlabel(var_x)
    ax3.set_ylabel(var_y)
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    st.write(f"Acuracia usando so {var_x} e {var_y}: {acc2:.1%}")

st.write(
    f"Com kernel {kernel}, o SVM acertou {acuracia:.1%} dos titulos do teste. "
    "A variavel tem_diretor e a que mais ajuda, porque quase nenhuma serie tem diretor cadastrado."
)
