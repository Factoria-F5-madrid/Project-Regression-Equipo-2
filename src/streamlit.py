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

# Configuración de la página
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Obtener el directorio del script y subir un nivel para acceder a data/
try:
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Subir un nivel desde src/
except NameError:
    # Si __file__ no está definido, usar directorio padre del actual
    SCRIPT_DIR = os.path.dirname(os.getcwd())

# Título principal
st.title("⚡ Energy Consumption Prediction System")
st.markdown("---")

# Configuración de modelos disponibles (rutas absolutas)
AVAILABLE_MODELS = {
    "Support Vector Regression (SVR)": os.path.join(SCRIPT_DIR, "data", "results", "svr_model.pkl"),
    "Linear Regression": os.path.join(SCRIPT_DIR, "data", "results", "linear_regression_model.pkl"),
    "Decision Tree": os.path.join(SCRIPT_DIR, "data", "results", "decision_tree_model.pkl"), 
    "K-Nearest Neighbors (KNN)": os.path.join(SCRIPT_DIR, "data", "results", "knn_model.pkl")
}

# Función para verificar qué modelos están disponibles
@st.cache_data
def check_available_models():
    available = {}
    for name, path in AVAILABLE_MODELS.items():
        if os.path.exists(path):
            available[name] = path
    return available

# Función para cargar modelo pre-entrenado
@st.cache_resource
def load_pretrained_model(model_path):
    try:
        model = joblib.load(model_path)
        return model, True
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, False

# Función para cargar datos
@st.cache_data
def load_data():
    try:
        # Cargar los archivos reales (rutas absolutas)
        train_path = os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed.csv")
        test_path = os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed_test.csv")
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        return df_train, df_test, True
    except FileNotFoundError as e:
        st.error(f"Error: No se encontraron los archivos de datos: {str(e)}")
        st.error("Asegúrate de que existan los archivos:")
        st.code(f"""
        {os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed.csv")}
        {os.path.join(SCRIPT_DIR, "data", "processed", "energy_data_processed_test.csv")}
        """)
        return None, None, False

# Sidebar para configuración
st.sidebar.header("🔧 Configuración del Modelo")

# Verificar modelos disponibles
available_models = check_available_models()

if not available_models:
    st.sidebar.error("❌ No se encontraron modelos entrenados")
    st.error("No se encontraron modelos pre-entrenados en data/results/")
    st.error("Asegúrate de que existan los archivos:")
    
    # Mostrar rutas exactas que está buscando
    st.code(f"""
    {os.path.join(SCRIPT_DIR, "data", "results", "svr_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "linear_regression_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "decision_tree_model.pkl")}
    {os.path.join(SCRIPT_DIR, "data", "results", "knn_model.pkl")}
    """)
    
    # Debug: mostrar directorio actual y archivos encontrados
    st.write(f"**Directorio del script:** `{SCRIPT_DIR}`")
    st.write(f"**Directorio actual:** `{os.getcwd()}`")
    
    # Verificar si existe el directorio
    results_dir = os.path.join(SCRIPT_DIR, "data", "results")
    if os.path.exists(results_dir):
        st.write(f"**Archivos en {results_dir}:**")
        files = os.listdir(results_dir)
        st.write(files)
    else:
        st.error(f"El directorio {results_dir} no existe")
    
    st.stop()

# Mostrar solo modelos disponibles
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

# Cargar datos
df_train, df_test, data_loaded = load_data()

if not data_loaded:
    st.stop()

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

# Crear tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración de Datos", "🔮 Evaluación", "📈 Métricas del Modelo", "🎯 Predicción Individual"])

with tab1:
    st.header("Exploración de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información del Dataset")
        st.write(f"**Forma del dataset:** {df.shape}")
        st.write("**Primeras 5 filas:**")
        st.dataframe(df.head())
        
        st.write("**Estadísticas descriptivas:**")
        st.dataframe(df.describe())
    
    with col2:
        st.subheader("Distribución del Consumo de Energía")
        fig = px.histogram(df, x='Energy Consumption', nbins=30, 
                          title="Distribución del Consumo de Energía")
        st.plotly_chart(fig, use_container_width=True)
        
        # Crear una columna temporal para Building Type desde las columnas OneHot
        st.subheader("Consumo por Tipo de Edificio")
        
        # Reconstruir Building Type desde las columnas OneHot
        if 'Building Type_Commercial' in df.columns:
            # Crear columna temporal Building Type
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
    
    st.subheader("Matriz de Correlación")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()
    
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Matriz de Correlación")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("Evaluación del Modelo Pre-entrenado")
    
    if st.button("🔍 Evaluar Modelo", type="primary"):
        with st.spinner(f"Evaluando modelo {model_type}..."):
            try:
                # Separar características y target
                X = df.drop('Energy Consumption', axis=1)
                y = df['Energy Consumption']
                
                # Hacer predicciones con el modelo cargado
                y_pred = loaded_model.predict(X)
                
                # Guardar en session state
                st.session_state.model = loaded_model
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.y_pred = y_pred
                st.session_state.model_name = model_type
                
                st.success("✅ Modelo evaluado exitosamente!")
                
            except Exception as e:
                st.error(f"Error al hacer predicciones: {str(e)}")
                st.info("💡 Posibles causas:")
                st.write("- El modelo no es compatible con la estructura actual de datos")
                st.write("- Faltan columnas o el preprocesamiento es diferente")
                st.write("- El modelo requiere un formato específico de datos")
        
        # Mostrar resultados
        if 'y_pred' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predicciones vs Valores Reales")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y, y=y_pred, mode='markers',
                                       name='Predicciones', opacity=0.7))
                fig.add_trace(go.Scatter(x=[y.min(), y.max()], 
                                       y=[y.min(), y.max()],
                                       mode='lines', name='Línea Perfecta', 
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(title="Predicciones vs Valores Reales",
                                xaxis_title="Valores Reales", yaxis_title="Predicciones")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Análisis de Errores")
                residuals = y - y_pred
                fig = px.scatter(x=y_pred, y=residuals, title="Errores vs Predicciones")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(xaxis_title="Predicciones", yaxis_title="Errores")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Haz clic en 'Evaluar Modelo' para ver el rendimiento del modelo pre-entrenado")

with tab3:
    st.header("Métricas del Modelo")
    
    if 'y_pred' in st.session_state:
        y = st.session_state.y
        y_pred = st.session_state.y_pred
        model_name = st.session_state.model_name
        
        # Calcular métricas
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Mostrar métricas principales
        st.subheader(f"📊 Métricas del Modelo: {model_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        # Interpretación del R²
        st.subheader("📋 Interpretación de Métricas")
        
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
        
        st.markdown(f"**R² Score:** <span style='color:{r2_color}'>{r2_interpretation}</span> - El modelo explica {r2*100:.1f}% de la varianza", 
                   unsafe_allow_html=True)
        
        st.write(f"**RMSE:** {rmse:.2f} kWh - Error promedio en las predicciones")
        st.write(f"**MAE:** {mae:.2f} kWh - Error absoluto promedio")
        
        # Distribución de errores
        st.subheader("📈 Distribución de Errores")
        residuals = y - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(residuals, nbins=30, title="Distribución de Errores")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Estadísticas de errores
            st.write("**Estadísticas de Errores:**")
            error_stats = pd.DataFrame({
                'Métrica': ['Media', 'Desv. Estándar', 'Mínimo', 'Máximo'],
                'Valor': [residuals.mean(), residuals.std(), residuals.min(), residuals.max()]
            })
            st.dataframe(error_stats)
        
    else:
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación'")

with tab4:
    st.header("Predicción Individual")
    
    if 'model' in st.session_state:
        st.subheader("Ingresa los datos para hacer una predicción:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            building_type = st.selectbox("Tipo de Edificio", ["Residential", "Commercial", "Industrial"])
            square_footage = st.number_input("Metros Cuadrados", min_value=500, max_value=100000, value=25000)
            num_occupants = st.number_input("Número de Ocupantes", min_value=1, max_value=200, value=50)
        
        with col2:
            appliances = st.number_input("Número de Electrodomésticos", min_value=1, max_value=100, value=25)
            temperature = st.number_input("Temperatura Promedio (°C)", min_value=-10.0, max_value=50.0, value=22.0)
            day_of_week = st.selectbox("Día de la Semana", ["Weekday", "Weekend"])
        
        if st.button("🔮 Hacer Predicción"):
            # Crear dataframe con los datos ingresados
            input_data = pd.DataFrame({
                'Building Type': [building_type],
                'Square Footage': [square_footage],
                'Number of Occupants': [num_occupants],
                'Appliances Used': [appliances],
                'Average Temperature': [temperature],
                'Day of Week': [day_of_week]
            })
            
            # Hacer predicción con el modelo cargado
            try:
                prediction = st.session_state.model.predict(input_data)[0]
                
                # Mostrar resultado
                st.success(f"⚡ **Consumo de Energía Predicho: {prediction:.2f} kWh**")
                st.info(f"🤖 **Modelo utilizado:** {st.session_state.model_name}")
                
                # Mostrar interpretación
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
                
                # Mostrar los datos ingresados en una tabla
                st.subheader("📋 Datos Ingresados:")
                st.dataframe(input_data)
                    
            except Exception as e:
                st.error(f"Error al hacer la predicción: {str(e)}")
                st.info("💡 Asegúrate de que el modelo fue entrenado con la misma estructura de datos")
    
    else:
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación'")

# Footer
st.markdown("---")
st.markdown(f"**Energy Consumption Predictor** - Usando modelos pre-entrenados | Modelos disponibles: {len(available_models)}/4")