import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis de Campañas de Marketing", layout="wide")

# Título de la aplicación
st.title("Dashboard de Análisis de Campañas de Marketing")

# Cargar los datos
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\Andrei.Baidurov\Marketing_Proyecto\data\marketingcampaigns_clean.csv")
    return data

try:
    # Cargamos el dataset
    df = load_data()
    st.success("Datos cargados correctamente!")
    
    # Mostramos una muestra de los datos
    st.subheader("Muestra de Datos")
    st.dataframe(df.head())
    
    # Estadísticas básicas
    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe())
    
    # Sidebar para filtros
    st.sidebar.header("Filtros")
    
    # Sección de análisis de datos
    st.header("Análisis de Datos")
    
    # Pestañas para visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["Rendimiento de Campañas", "Demografía de Clientes", "Análisis de Conversión", "Gráfico Personalizado"])
    
    with tab1:
        st.subheader("Rendimiento de Campañas")
        # Buscamos columnas relacionadas con campañas y conversiones
        campaign_cols = [col for col in df.columns if 'campaign' in col.lower()]
        metric_cols = [col for col in df.columns if any(x in col.lower() for x in ['conversion', 'revenue', 'sale', 'response'])]
        
        if campaign_cols and metric_cols:
            campaign_col = st.selectbox("Seleccionar Columna de Campaña", campaign_cols)
            metric_col = st.selectbox("Seleccionar Métrica de Rendimiento", metric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            campaign_perf = df.groupby(campaign_col)[metric_col].mean().sort_values(ascending=False)
            campaign_perf.plot(kind='bar', ax=ax)
            plt.title(f'Promedio de {metric_col} por {campaign_col}')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No se encontraron columnas adecuadas para este análisis.")
    
    with tab2:
        st.subheader("Demografía de Clientes")
        # Buscamos columnas demográficas
        demo_cols = [col for col in df.columns if any(x in col.lower() for x in ['age', 'gender', 'income', 'education'])]
        
        if demo_cols:
            demo_col = st.selectbox("Seleccionar Variable Demográfica", demo_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if df[demo_col].nunique() <= 10:  # Datos categóricos
                df[demo_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                plt.title(f'Distribución por {demo_col}')
            else:  # Datos numéricos
                sns.histplot(df[demo_col], kde=True, ax=ax)
                plt.title(f'Distribución de {demo_col}')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No se encontraron columnas demográficas.")
    
    with tab3:
        st.subheader("Análisis de Conversión")
        # Columnas para factores y objetivos
        factor_cols = [col for col in df.columns if col not in metric_cols]
        
        if factor_cols and metric_cols:
            factor_col = st.selectbox("Seleccionar Factor", factor_cols)
            target_col = st.selectbox("Seleccionar Métrica Objetivo", metric_cols, key="target_metric")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if df[factor_col].nunique() <= 20:  # Evitar demasiadas categorías
                conv_by_factor = df.groupby(factor_col)[target_col].mean().sort_values(ascending=False)
                conv_by_factor.plot(kind='bar', ax=ax)
                plt.title(f'Promedio de {target_col} por {factor_col}')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning(f"La columna {factor_col} tiene demasiados valores únicos para visualizar.")
        else:
            st.write("No se encontraron columnas adecuadas para este análisis.")
    
    with tab4:
        st.subheader("Gráfico Personalizado")
        x_axis = st.selectbox("Seleccionar Eje X", df.columns)
        y_axis = st.selectbox("Seleccionar Eje Y", df.columns)
        chart_type = st.selectbox("Seleccionar Tipo de Gráfico", ["Dispersión", "Línea", "Barras", "Caja"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "Dispersión":
            plt.scatter(df[x_axis], df[y_axis])
            plt.title(f'{y_axis} vs {x_axis}')
        elif chart_type == "Línea":
            sorted_data = df.sort_values(by=x_axis)
            plt.plot(sorted_data[x_axis], sorted_data[y_axis])
            plt.title(f'{y_axis} vs {x_axis}')
        elif chart_type == "Barras":
            if df[x_axis].nunique() <= 20:
                df.groupby(x_axis)[y_axis].mean().plot(kind='bar', ax=ax)
                plt.title(f'Promedio de {y_axis} por {x_axis}')
            else:
                st.warning("Demasiados valores únicos en el eje X.")
        elif chart_type == "Caja":
            if df[x_axis].nunique() <= 20:
                sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
                plt.title(f'Distribución de {y_axis} por {x_axis}')
            else:
                st.warning("Demasiados valores únicos en el eje X.")
        
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.tight_layout()
        st.pyplot(fig)

    # Matriz de correlación
    st.header("Matriz de Correlación")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No hay suficientes columnas numéricas para el análisis de correlación.")

except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {str(e)}")
    st.info("Asegúrate de que el archivo 'marketingcampaigns_clean.csv' esté en el mismo directorio que este script.")

    