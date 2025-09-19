#  Proyect## 📈 Dataset
El conjunto de datos analiza el consumo energético con las siguientes variables clave:

### Variables Objetivo
- **Energy Consumption**: Consumo energético (variable a predecir)

### Variables Estructurales
- **Square Footage**: Superficie del edificio en pies cuadrados
- **Number of Occupants**: Número de ocupantes en el edificio
- **Appliances Used**: Cantidad de electrodomésticos en uso

### Variables Ambientales
- **Average Temperature**: Temperatura promedio del ambiente
- El análisis mostró una correlación significativa (0.72) con el consumo energético

### Variables Temporales
- **Day of Week**: Día de la semana (variable categórica)
- Se observaron patrones de consumo diferentes según el día

### Características del Dataset
- **Tamaño**: Conjunto de datos robusto sin valores faltantes
- **Calidad**: Alta calidad con mínima necesidad de limpieza
- **Distribución**: Variables numéricas con distribuciones bien definidas
- **Correlaciones**: Fuertes relaciones identificadas entre variables estructurales y consumolisis de Consumo Energético - Equipo 2

##  Descripción del Proyecto
Este proyecto se centra en el análisis y predicción del consumo energético técnicas de Machine Learning. A través de un análisis exploratorio detallado y la implementación de modelos de regresión, buscamos comprender y predecir patrones de consumo energético.

##  Objetivos
- Analizar patrones de consumo energético
- Identificar factores clave que influyen en el consumo
- Desarrollar un modelo predictivo preciso

##  Dataset
El conjunto de datos incluye información sobre:
- Consumo energético (variable objetivo)
- Características estructurales del edificio
- Número de ocupantes
- Uso de electrodomésticos
- Condiciones ambientales
- Patrones temporales

##  Principales Hallazgos

### Análisis Numérico
- **Correlaciones Significativas**: 
  - Fuerte relación entre tamaño del edificio y consumo
  - Impacto notable del número de ocupantes
  - Influencia de la temperatura promedio

### Patrones Temporales
- Variaciones significativas por día de la semana
- Patrones de consumo identificables


### Variables Categóricas
- Diferencias significativas entre grupos
- Distribuciones balanceadas
- Patrones claros por categoría

##  Preprocesamiento
1. **Limpieza de Datos**
   - Sin valores faltantes
   - Tratamiento de outliers
   - Validación de tipos de datos

2. **Transformaciones**
   - Codificación one-hot de variables categóricas
   - Normalización de variables numéricas
   - Escalado de características

##  Estructura del Proyecto
```
Project-Regression-Equipo-2/
│
├── data/
│   ├── raw/          # Datos originales
│   ├── processed/    # Datos procesados
│   └── external/     # Datos externos
│
├── notebooks/
│   ├── eda_project_regression.ipynb    # Análisis exploratorio
│   └── regression_analysis.ipynb       # Modelado
│
└── src/              # Código fuente
```

##  Insights Clave
1. El consumo energético está fuertemente correlacionado con el tamaño del edificio
2. Los patrones de ocupación son predictores importantes
3. Existen variaciones temporales significativas
4. Las variables ambientales tienen un impacto medible

##  Próximos Pasos
- Implementación de modelos de regresión avanzados
- Validación cruzada y optimización
- Evaluación de rendimiento del modelo

##  Tecnologías Utilizadas
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

##  Equipo
- Proyecto desarrollado como parte del Bootcamp de IA en Factoría F5

##  Estado del Proyecto
- Análisis exploratorio completado
- En proceso de desarrollo del modelo de regresión

---
*Este proyecto forma parte de una iniciativa más amplia para comprender y optimizar el consumo energético en edificios.*
