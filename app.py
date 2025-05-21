import streamlit as st
import pandas as pd

# Configuración de la app
st.set_page_config(page_title="Mi primera app", page_icon=":tada:", layout="centered")

# CSS personalizado
st.markdown("""
    <style>
        /* Fondo de la aplicación */
        .stApp {
            background-color: #f5f5f5;
        }

        /* Estilo para los encabezados */
        h1, h2, h3 {
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
        }

        /* Estilo para los textos */
        .stText, .stMarkdown {
            font-family: 'Verdana', sans-serif;
            color: #333;
        }

        /* Estilo para las columnas */
        .stColumn > div {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Estilo para los tabs */
        .stTabs [role="tablist"] {
            background-color: #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }

        .stTabs [role="tab"] {
            color: white;
            font-weight: bold;
        }

        .stTabs [role="tab"]:hover {
            background-color: #45a049;
        }

        /* Estilo para las tablas */
        .stDataFrame {
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)


st.title("Mi primera app")  # Título de la app

# Sidebar
st.sidebar.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", caption="Streamlit", use_container_width=True)
st.sidebar.text("Desliza")

# Tabs
tab_1, tab_2, tab_3 = st.tabs(["Graficos", "Tablas", "Tab 3"])

with tab_1:
    st.write("Contenido del Graficos")
    st.header("Encabezado de la app")
    st.subheader("Subtitulo de la app")
    st.text("Texto de la app")
    st.markdown("Hello **mundo**")
    st.latex(r"""a^2 + b^2 = c^2""")
    st.code("print('Hello world')", language="python")
    st.info("Esto es un mensaje de info")
    st.success("Esto es un mensaje de exito")
    st.warning("Esto es un mensaje de advertencia")
    st.error("Esto es un mensaje de error")
    st.exception("Esto es un mensaje de excepcion")

with tab_2:
    st.write("Contenido de las Tablas")
    st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", caption="Streamlit")
    st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", caption="Streamlit", use_container_width=True)
    st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3")
    st.video("https://www.youtube.com/watch?v=2Vv-BfVoq4g")

with tab_3:
    st.write("Contenido del Tab 3")
    df = pd.DataFrame({
        "Nombre": ["Juan", "Pedro", "Maria", "Ana"],
        "Edad": [20, 30, 25, 35],
        "Ciudad": ["Madrid", "Barcelona", "Valencia", "Sevilla"]
    })
    st.dataframe(df)

# Columnas
col1, col2 = st.columns(2)
with col1:
    st.write("Contenido de la columna 1")
    st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", caption="Streamlit")

with col2:
    st.write("Contenido de la columna 2")
    st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", caption="Streamlit")



if 'contador' not in st.session_state:
    st.session_state.contador = 0

def incrementar_contador():
    st.session_state.contador += 1     
def decrementar_contador():
    st.session_state.contador -= 1
def reiniciar_contador():
    st.session_state.contador = 0


st.title("Contador")
st.write("Contador: ", st.session_state.contador)

col1, col2, col3 = st.columns(3)
with col1:
    st.button("Incrementar", on_click=incrementar_contador) 
with col2:
    st.button("Decrementar", on_click=decrementar_contador)
with col3:
    st.button("Reiniciar", on_click=reiniciar_contador)