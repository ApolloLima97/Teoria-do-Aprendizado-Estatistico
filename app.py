import streamlit as st

pagina_inicio = st.Page("pagina_inicio.py", title="Inicio", default=True)
pagina_bootstrap = st.Page("pagina_bootstrap.py", title="Bootstrap", url_path="bootstrap")
pagina_svm = st.Page("pagina_svm.py", title="SVM", url_path="svm")

pg = st.navigation([pagina_inicio, pagina_bootstrap, pagina_svm])
pg.run()
