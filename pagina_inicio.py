import streamlit as st

from bootstrep import carregar_dados

dados = carregar_dados()

st.title("FATEC COTIA")
st.header("Curso de Ciencia de Dados")
st.write("Teoria do Aprendizado Estatistico - Bootstrap, SVM e K-means")
st.write("Alunos: Deyvis Pires de Oliveira e Apolinario Dario de Lima")

col1, col2, col3 = st.columns(3)
col1.metric("Total de titulos", len(dados))
col2.metric("Filmes", (dados["type"] == "Movie").sum())
col3.metric("Series", (dados["type"] == "TV Show").sum())

st.subheader("Base de dados (netflix_titles.csv)")
st.dataframe(dados)
