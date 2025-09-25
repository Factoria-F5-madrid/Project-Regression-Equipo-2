# ==============================================================================
# APLICACIÓN STREAMLIT PARA PREDICCIÓN DE CONSUMO ENERGÉTICO
# ==============================================================================
# 
# Esta aplicación web permite evaluar y utilizar múltiples modelos de regresión
# pre-entrenados para predecir el consumo energético de edificios.
#
# CARACTERÍSTICAS PRINCIPALES:
# - Soporte para diferentes algoritmos (SVR, Linear, Tree, KNN, XGBoost)
# - Visualizaciones interactivas con Plotly
# - Predicciones individuales en tiempo real
# - Análisis completo de métricas y residuos
# - Interfaz responsive con sidebar de configuración
#
# ESTRUCTURA:
# 1. Configuración inicial y carga de modelos
# 2. Interfaz de usuario (sidebar + tabs)
# 3. Exploración de datos con visualizaciones
# 4. Evaluación de modelos pre-entrenados
# 5. Dashboard de métricas
# 6. Herramienta de predicción individual
# ==============================================================================

# IMPORTACIÓN DE LIBRERÍAS
# ------------------------
# Se importan todas las dependencias necesarias para la aplicación we

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# streamlit run streamlit.py

# Configuración de la página principal
# Esta debe ser la PRIMERA llamada de Streamlit en cualquier app
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DEL SISTEMA
# ==============================================================================

# Detección inteligente del directorio del proyecto
# Problema: Streamlit puede ejecutarse desde diferentes directorios
# Solución: Detectar automáticamente la ubicación y subir un nivel desde src/

try:
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Subir un nivel desde src/
except NameError:
    SCRIPT_DIR = os.path.dirname(os.getcwd())

# ==============================================================================
# INTERFAZ DE USUARIO - ENCABEZADO PRINCIPAL
# ==============================================================================

st.title("⚡ Energy Consumption Prediction System")
st.markdown("---")

# ==============================================================================
# CONFIGURACIÓN DE MODELOS DISPONIBLES
# ==============================================================================

# Diccionario que mapea nombres de modelos a sus rutas de archivo
# Usar os.path.join() garantiza compatibilidad entre sistemas operativos

AVAILABLE_MODELS = {
    "Support Vector Regression (SVR)": os.path.join(SCRIPT_DIR, "data", "results", "svr_model.pkl"),
    "Linear Regression": os.path.join(SCRIPT_DIR, "data", "results", "linear_regression_model.pkl"),
    "Decision Tree": os.path.join(SCRIPT_DIR, "data", "results", "decision_tree_model.pkl"), 
    "K-Nearest Neighbors (KNN)": os.path.join(SCRIPT_DIR, "data", "results", "knn_model.pkl"),
    "Ensemble (XGBoost)": os.path.join(SCRIPT_DIR, "data", "results", "ensemble_model.pkl")
}

# ==============================================================================
# FUNCIONES DE CARGA Y VERIFICACIÓN
# ==============================================================================
# Función para verificar qué modelos están disponibles

@st.cache_data # Cache para evitar verificaciones repetitivas
def check_available_models():
    """
    Verifica qué modelos están disponibles en el sistema de archivos.
    
    Returns:
        dict: Diccionario con modelos disponibles {nombre: ruta}
    
    Ventajas del cache:
    - Solo verifica archivos una vez por sesión
    - Mejora significativamente la velocidad de carga
    - Se actualiza automáticamente si cambian los archivos
    """
    available = {}
    for name, path in AVAILABLE_MODELS.items():
        if os.path.exists(path):
            available[name] = path
    return available

# Función para cargar modelo pre-entrenado
@st.cache_resource # Cache de recursos (modelos cargados en memoria)
def load_pretrained_model(model_path):
    """
    Carga un modelo pre-entrenado desde disco usando joblib.
    
    Args:
        model_path (str): Ruta completa al archivo .pkl del modelo
        
    Returns:
        tuple: (modelo_cargado, éxito_booleano)
        
    Cache de recursos:
    - Los modelos se mantienen en memoria durante toda la sesión
    - Evita recargar modelos pesados repetitivamente
    - Se limpia automáticamente cuando cambia el código
    """
    try:
        model = joblib.load(model_path)
        return model, True
    except Exception as e:
        # Mostrar error específico al usuario para debugging
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, False

# Función para cargar datos
@st.cache_data # Cache de datos
def load_data():
    """
    Carga los datasets de entrenamiento y prueba desde archivos CSV.
    
    Returns:
        tuple: (df_train, df_test, éxito_booleano)
        
    Manejo de errores:
    - Muestra rutas específicas que está buscando
    - Proporciona información de debugging detallada
    - Retorna None en caso de error para manejo controlado
    """
    try:
        # Construcción de rutas absolutas para compatibilidad
        train_path = os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed.csv")
        test_path = os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed_test.csv")
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        return df_train, df_test, True
    except FileNotFoundError as e:
        # Error específico con información útil para el usuario
        st.error(f"Error: No se encontraron los archivos de datos: {str(e)}")
        st.error("Asegúrate de que existan los archivos:")
        st.code(f"""
        {os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed.csv")}
        {os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed_test.csv")}
        """)
        return None, None, False

# ==============================================================================
# SIDEBAR - PANEL DE CONFIGURACIÓN
# ==============================================================================

st.sidebar.header("🔧 Configuración del Modelo")

# Verificar modelos disponibles
available_models = check_available_models()

# Control de flujo: detener la app si no hay modelos
if not available_models:
    # Interfaz de error completa con información de debugging
    st.sidebar.error("❌ No se encontraron modelos entrenados")
    st.error("No se encontraron modelos pre-entrenados en data/results/")
    st.error("Asegúrate de que existan los archivos:")
    
    # Mostrar rutas exactas que está buscando
    st.code(f"""
    {os.path.join(SCRIPT_DIR, "data", "results", "svr_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "linear_regression_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "decision_tree_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "knn_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "ensemble_model.pkl")}
    """)
    
    # Debug: mostrar directorio actual y archivos encontrados
    st.write(f"**Directorio del script:** `{SCRIPT_DIR}`")
    st.write(f"**Directorio actual:** `{os.getcwd()}`")

    # Mostrar archivos en el directorio results
    results_dir = os.path.join(SCRIPT_DIR, "data", "results")
    st.write(f"**Archivos en {results_dir}:**")
    files = os.listdir(results_dir)
    st.write(files)
        
    st.stop() # Terminar ejecución de la app

# Selector de modelos (solo modelos disponibles)
available_names = list(available_models.keys())
model_type = st.sidebar.selectbox(
    "Selecciona el Algoritmo de Regresión:",
    available_names,
    index=0
)

# Mostrar estado de modelos
st.sidebar.subheader("📊 Estado de Modelos")
for name, path in AVAILABLE_MODELS.items():
    if name in available_models:
        st.sidebar.success(f"✅ {name}")
    else:
        st.sidebar.error(f"❌ {name}")

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================

# Cargar datos con manejo de errores
df_train, df_test, data_loaded = load_data()

# Control de flujo: detener si no se pueden cargar los datos
if not data_loaded:
    st.stop()

# ==============================================================================
# CONFIGURACIÓN DE DATASET Y MODELO
# ==============================================================================

# Selección del dataset
dataset_option = st.sidebar.selectbox(
    "Selecciona el Dataset:",
    ["Dataset de Entrenamiento", "Dataset de Test", "Ambos Datasets"],
    index=1  # Por defecto test para evaluar modelo pre-entrenado
)

# Información del modelo seleccionado
st.sidebar.info(f"🤖 Modelo cargado: {model_type}")

# Cargar el modelo seleccionado
model_path = available_models[model_type]
loaded_model, model_loaded_successfully = load_pretrained_model(model_path)

# Control de flujo: verificar carga exitosa del modelo
if not model_loaded_successfully:
    st.error(f"No se pudo cargar el modelo: {model_type}")
    st.stop()

# Seleccionar dataset según la opción
if dataset_option == "Dataset de Entrenamiento":
    df = df_train.copy()
    st.info(f"📊 Usando Dataset de Entrenamiento ({len(df)} registros)")
elif dataset_option == "Dataset de Test":
    df = df_test.copy()
    st.info(f"📊 Usando Dataset de Test ({len(df)} registros)")
else:  # Ambos datasets
    df = pd.concat([df_train, df_test], ignore_index=True)
    st.info(f"📊 Usando Ambos Datasets ({len(df)} registros: {len(df_train)} entrenamiento + {len(df_test)} test)")

# ==============================================================================
# INTERFAZ PRINCIPAL - SISTEMA DE PESTAÑAS
# ==============================================================================


# Crear tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración de Datos", "🔮 Evaluación", "📈 Métricas del Modelo", "🎯 Predicción Individual"])

# ==============================================================================
# TAB 1: EXPLORACIÓN DE DATOS
# ==============================================================================

with tab1:
    st.header("Exploración de Datos")

    # Layout de dos columnas para mejor organización
    col1, col2 = st.columns(2)
    
    # COLUMNA IZQUIERDA: Información básica del dataset
    with col1:
        st.subheader("Información del Dataset")
        st.write(f"**Forma del dataset:** {df.shape}")
        st.write("**Primeras 5 filas:**")
        st.dataframe(df.head())
        
        st.write("**Estadísticas descriptivas:**")
        st.dataframe(df.describe())

    # COLUMNA DERECHA: Visualizaciones
    with col2:
        # Gráfico 1: Distribución de la variable objetivo
        st.subheader("Distribución del Consumo de Energía")
        fig = px.histogram(df, x='Energy Consumption', nbins=30, 
                          title="Distribución del Consumo de Energía")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico 2: Consumo por tipo de edificio
        st.subheader("Consumo por Tipo de Edificio")
        
        # MANEJO INTELIGENTE DE DATOS PREPROCESSADOS
        # Los datos pueden venir con OneHotEncoding aplicado
        # Esta lógica reconstruye las categorías originales para visualización

        if 'Building Type_Commercial' in df.columns:
            # Crear columna temporal Building Type desde las columnas OneHot
            df_temp = df.copy()
            conditions = [
                df_temp['Building Type_Commercial'] == 1,
                df_temp['Building Type_Industrial'] == 1,
                df_temp['Building Type_Residential'] == 1
            ]
            choices = ['Commercial', 'Industrial', 'Residential']
            df_temp['Building Type'] = np.select(conditions, choices, default='Unknown')
            
            fig2 = px.box(df_temp, x='Building Type', y='Energy Consumption',
                         title="Consumo de Energía por Tipo de Edificio")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Si no hay columnas OneHot, usar la columna original
            if 'Building Type' in df.columns:
                fig2 = px.box(df, x='Building Type', y='Energy Consumption',
                             title="Consumo de Energía por Tipo de Edificio")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No se puede mostrar el gráfico por tipo de edificio (columnas no encontradas)")
    
    # VISUALIZACIÓN A ANCHO COMPLETO: Matriz de correlación
    st.subheader("Matriz de Correlación")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()
    
    # Heatmap interactivo con valores mostrados
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Matriz de Correlación")
    st.plotly_chart(fig3, use_container_width=True)

# ==============================================================================
# TAB 2: EVALUACIÓN DE MODELOS
# ==============================================================================
with tab2:
    st.header("Evaluación del Modelo Pre-entrenado")
    
    # Botón principal para ejecutar evaluación
    if st.button("🔍 Evaluar Modelo", type="primary"):
        # Spinner para mostrar progreso durante operación pesada
        with st.spinner(f"Evaluando modelo {model_type}..."):
            try:
                # Separar características (X) y variable objetivo (y)
                X = df.drop('Energy Consumption', axis=1)
                y = df['Energy Consumption']
                
                # Hacer predicciones con el modelo cargado
                y_pred = loaded_model.predict(X)
                
                # ALMACENAMIENTO EN SESSION STATE
                # Permite mantener datos entre interacciones del usuario
                st.session_state.model = loaded_model
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.y_pred = y_pred
                st.session_state.model_name = model_type
                
                st.success("✅ Modelo evaluado exitosamente!")
                
            except Exception as e:
                # Manejo de errores con información útil para el usuario
                st.error(f"Error al hacer predicciones: {str(e)}")
                st.info("💡 Posibles causas:")
                st.write("- El modelo no es compatible con la estructura actual de datos")
                st.write("- Faltan columnas o el preprocesamiento es diferente")
                st.write("- El modelo requiere un formato específico de datos")
        
        # VISUALIZACIONES POST-EVALUACIÓN
        # Solo se muestran si la evaluación fue exitosa
        if 'y_pred' in st.session_state:
            col1, col2 = st.columns(2)
            
            # Gráfico 1: Predicciones vs Valores Reales
            with col1:
                st.subheader("Predicciones vs Valores Reales")
                fig = go.Figure()
                # Scatter plot de predicciones
                fig.add_trace(go.Scatter(x=y, y=y_pred, mode='markers',
                                       name='Predicciones', opacity=0.7))
                # Línea de predicción perfecta (y=x)
                fig.add_trace(go.Scatter(x=[y.min(), y.max()], 
                                       y=[y.min(), y.max()],
                                       mode='lines', name='Línea Perfecta', 
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(title="Predicciones vs Valores Reales",
                                xaxis_title="Valores Reales", yaxis_title="Predicciones")
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico 2: Análisis de residuos (errores)
            with col2:
                st.subheader("Análisis de Errores")
                residuals = y - y_pred
                fig = px.scatter(x=y_pred, y=residuals, title="Errores vs Predicciones")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(xaxis_title="Predicciones", yaxis_title="Errores")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Mensaje de instrucciones cuando no se ha evaluado aún
        st.info("👆 Haz clic en 'Evaluar Modelo' para ver el rendimiento del modelo pre-entrenado")

# ==============================================================================
# TAB 3: MÉTRICAS DEL MODELO
# ==============================================================================

with tab3:
    st.header("Métricas del Modelo")
    
    # Verificar si hay resultados de evaluación disponibles
    if 'y_pred' in st.session_state:
        # Recuperar datos del session state
        y = st.session_state.y
        y_pred = st.session_state.y_pred
        model_name = st.session_state.model_name
        
        # CÁLCULO DE MÉTRICAS PRINCIPALES
        mse = mean_squared_error(y, y_pred)     # Error cuadrático medio
        rmse = np.sqrt(mse)                     # Raíz del error cuadrático medio
        mae = mean_absolute_error(y, y_pred)    # Error absoluto medio
        r2 = r2_score(y, y_pred)                # Coeficiente de determinación
        
        # DASHBOARD DE MÉTRICAS
        st.subheader(f"📊 Métricas del Modelo: {model_name}")
        
        # Layout de 4 columnas para métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        # INTERPRETACIÓN AUTOMÁTICA DE R²
        st.subheader("📋 Interpretación de Métricas")
        
        # Lógica de clasificación del rendimiento
        if r2 >= 0.9:
            r2_interpretation = "🟢 Excelente"
            r2_color = "green"
        elif r2 >= 0.8:
            r2_interpretation = "🟡 Bueno"
            r2_color = "orange"
        elif r2 >= 0.6:
            r2_interpretation = "🟠 Regular" 
            r2_color = "orange"
        else:
            r2_interpretation = "🔴 Deficiente"
            r2_color = "red"
        
        # Mostrar interpretación con colores
        st.markdown(f"**R² Score:** <span style='color:{r2_color}'>{r2_interpretation}</span> - El modelo explica {r2*100:.1f}% de la varianza", 
                   unsafe_allow_html=True)
        
        st.write(f"**RMSE:** {rmse:.2f} kWh - Error promedio en las predicciones")
        st.write(f"**MAE:** {mae:.2f} kWh - Error absoluto promedio")
        
        # ANÁLISIS DETALLADO DE ERRORES
        st.subheader("📈 Distribución de Errores")
        residuals = y - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de residuos
            fig = px.histogram(residuals, nbins=30, title="Distribución de Errores")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Estadísticas descriptivas de errores
            st.write("**Estadísticas de Errores:**")
            error_stats = pd.DataFrame({
                'Métrica': ['Media', 'Desv. Estándar', 'Mínimo', 'Máximo'],
                'Valor': [residuals.mean(), residuals.std(), residuals.min(), residuals.max()]
            })
            st.dataframe(error_stats)
        
    else:
        # Mensaje cuando no hay evaluación previa
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación'")

# ==============================================================================
# TAB 4: PREDICCIÓN INDIVIDUAL
# ==============================================================================

with tab4:
    st.header("Predicción Individual")
    
    # Verificar que hay un modelo cargado
    if 'model' in st.session_state:
        st.subheader("Ingresa los datos para hacer una predicción:")
        
        # FORMULARIO DE ENTRADA DE DATOS
        # Layout de dos columnas para organizar los inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Inputs de la columna izquierda
            building_type = st.selectbox("Tipo de Edificio", ["Residential", "Commercial", "Industrial"])
            square_footage = st.number_input("Metros Cuadrados", min_value=500, max_value=100000, value=25000)
            num_occupants = st.number_input("Número de Ocupantes", min_value=1, max_value=200, value=50)
        
        with col2:
            # Inputs de la columna derecha
            appliances = st.number_input("Número de Electrodomésticos", min_value=1, max_value=100, value=25)
            temperature = st.number_input("Temperatura Promedio (°C)", min_value=-10.0, max_value=50.0, value=22.0)
            day_of_week = st.selectbox("Día de la Semana", ["Weekday", "Weekend"])
        
        # BOTÓN DE PREDICCIÓN
        if st.button("🔮 Hacer Predicción"):
            try:
                # PREPROCESAMIENTO DE DATOS DE ENTRADA
                # Los modelos esperan datos en formato OneHotEncoded
                
                # Crear DataFrame base con características numéricas
                input_data = pd.DataFrame({
                    'Square Footage': [square_footage],
                    'Number of Occupants': [num_occupants],
                    'Appliances Used': [appliances],
                    'Average Temperature': [temperature]
                })
                
                # APLICACIÓN MANUAL DE ONEHOTENCODING
                # Para Building Type (solo una puede ser 1)# Aplicar OneHotEncoding manual para Building Type
                input_data['Building Type_Commercial'] = 1 if building_type == 'Commercial' else 0
                input_data['Building Type_Industrial'] = 1 if building_type == 'Industrial' else 0
                input_data['Building Type_Residential'] = 1 if building_type == 'Residential' else 0
                
                # Para Day of Week (solo una puede ser 1)
                input_data['Day of Week_Weekday'] = 1 if day_of_week == 'Weekday' else 0
                input_data['Day of Week_Weekend'] = 1 if day_of_week == 'Weekend' else 0
                
                # ORDENAMIENTO DE COLUMNAS
                # Crucial: el modelo espera las columnas en un orden específico
                column_order = [
                    'Square Footage', 'Number of Occupants', 'Appliances Used', 
                    'Average Temperature', 'Building Type_Commercial', 
                    'Building Type_Industrial', 'Building Type_Residential', 
                    'Day of Week_Weekday', 'Day of Week_Weekend'
                ]
                
                # Asegurarse de que todas las columnas estén presentes
                for col in column_order:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                # Reordenar las columnas
                input_data = input_data[column_order]
                
                # Hacer predicción con el modelo cargado
                prediction = st.session_state.model.predict(input_data)[0]
                
                # Mostrar resultados
                st.success(f"⚡ **Consumo de Energía Predicho: {prediction:.2f} kWh**")
                st.info(f"🤖 **Modelo utilizado:** {st.session_state.model_name}")
                
                # Interpretación automática
                if prediction < 3000:
                    interpretation = "🟢 Consumo Bajo"
                    color = "green"
                elif prediction < 4500:
                    interpretation = "🟡 Consumo Medio"
                    color = "orange"
                else:
                    interpretation = "🔴 Consumo Alto"
                    color = "red"
                
                st.markdown(f"**Interpretación:** <span style='color:{color}'>{interpretation}</span>", 
                           unsafe_allow_html=True)
                
                # Comparar con promedio del dataset
                avg_consumption = df['Energy Consumption'].mean()
                difference = prediction - avg_consumption
                percentage = (difference / avg_consumption) * 100
                
                if difference > 0:
                    st.info(f"📊 Esta predicción es {difference:.2f} kWh ({percentage:.1f}%) **mayor** que el promedio del dataset ({avg_consumption:.2f} kWh)")
                else:
                    st.info(f"📊 Esta predicción es {abs(difference):.2f} kWh ({abs(percentage):.1f}%) **menor** que el promedio del dataset ({avg_consumption:.2f} kWh)")
                
                # Mostrar los datos procesados para debug
                st.subheader("📋 Datos Procesados (enviados al modelo):")
                st.dataframe(input_data)
                    
            except Exception as e:
                # Manejo de errores detallado
                st.error(f"Error al hacer la predicción: {str(e)}")
                st.info("💡 Detalles del error:")
                st.write(f"- Forma de los datos: {input_data.shape if 'input_data' in locals() else 'No creados'}")
                st.write(f"- Columnas: {list(input_data.columns) if 'input_data' in locals() else 'No disponibles'}")
                st.write("- Asegúrate de que el modelo fue entrenado con la misma estructura de datos")
    
    else:
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación'")

# ==============================================================================
# FOOTER DE LA APLICACIÓN
# ==============================================================================

st.markdown("---")
st.markdown(f"**Energy Consumption Predictor** - Usando modelos pre-entrenados | Modelos disponibles: {len(available_models)}/5")