# Agroasis — Sistema de Recomendación, Clasificación y Optimización de Fertilizante en Parcelas Agrícolas

## ¿Qué es Agroasis?

Sistema web que permite al agricultor analizar las propiedades fisicoquímicas de su suelo, clasificar cultivos compatibles mediante Machine Learning (Random Forest), calcular el déficit nutricional usando Álgebra Lineal (vectorización 5D), optimizar la dosis de fertilizante con Cálculo Diferencial, y proyectar la evolución del suelo a lo largo de múltiples temporadas (Estadística).

---

## Estructura del Proyecto

```
Pruebas01/
│
├── app.py                      # Archivo principal — servidor Flask y todas las rutas API
├── config.py                   # Configuración centralizada (rutas, features, umbrales)
├── requirements.txt            # Dependencias de Python (flask, pandas, numpy, scikit-learn)
│
├── models/                     # Modelos de Machine Learning y lógica matemática
│   ├── __init__.py
│   ├── crop_classifier.py      # [ML] Clasificador Random Forest de cultivos
│   ├── soil_analyzer.py        # [Estadística] Análisis de salud del suelo
│   └── fertilizer_recommender_v2.py  # [Álgebra Lineal] Vectorización 5D y déficit
│
├── utils/                      # Utilidades auxiliares
│   ├── __init__.py
│   ├── data_loader.py          # Carga y limpieza de datos CSV
│   └── validators.py           # Validación y sanitización de entradas
│
├── templates/                  # Vistas HTML (Jinja2)
│   ├── base.html               # Plantilla base con Bootstrap y navegación
│   ├── index.html              # Página de inicio — selector de modo
│   ├── csv_analisis.html       # Modo CSV — pipeline completo con gráficas
│   ├── manual_mode.html        # Modo Manual — inferencia sin datos de laboratorio
│   └── fertilizer_optimization.html  # Gráfica de optimización diferencial
│
├── static/                     # Archivos estáticos
│   ├── css/style.css           # Estilos personalizados
│   └── js/
│       ├── charts.js           # Gráficas con Plotly.js
│       └── fertilizer_optimization.js  # Lógica de la gráfica de optimización
│
├── trained_models/             # Modelos serializados (pickle)
│   ├── crop_classifier.pkl     # Random Forest entrenado
│   ├── scaler.pkl              # StandardScaler (normalización de features)
│   └── encoder_texture.pkl     # LabelEncoder (textura del suelo)
│
├── cultivos_mexico_v3_final.csv    # Dataset de entrenamiento (143 registros INIFAP)
├── fertilizantes.csv               # Catálogo de fertilizantes mexicanos con precios
├── Analisis_VariasParcelas_VariasMUestra.csv  # CSV de ejemplo para demostración
│
└── LEEME.md                    # Este archivo
```

---

## Tecnologías Utilizadas

### 
| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.13 | Lenguaje principal del servidor |
| Flask | 3.x | Framework web para rutas y API REST |
| pandas | 2.x | Manipulación y lectura de datos CSV |
| NumPy | 2.x | Operaciones vectoriales y matriciales |
| scikit-learn | 1.x | Random Forest, StandardScaler, LabelEncoder |

### 
| Tecnología | Uso |
|---|---|
| HTML5 | Estructura de las páginas |
| CSS3 | Estilos personalizados (diseño responsive, tarjetas, badges) |
| JavaScript (ES6+) | Lógica de interacción, renderizado dinámico, fetch API |
| Bootstrap 5 | Framework CSS para layout y componentes base |
| Plotly.js | Gráficas interactivas (curvas de optimización, proyección temporal) |
| Bootstrap Icons | Iconografía del sistema |

### Despliegue
| Plataforma | Uso |
|---|---|
| PythonAnywhere | Hosting del servidor Flask en producción |

---

## Cómo Ejecutar Localmente

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Instalar dependencias
pip install flask pandas numpy scikit-learn

# 3. Ejecutar servidor
python app.py
# Abrir en navegador: http://127.0.0.1:5000
```

---

## Fundamentos Matemáticos

El sistema integra cuatro disciplinas:

### 1. Machine Learning — Clasificación de Cultivos
Clasificador **Random Forest** con 200 árboles entrenado sobre 143 registros de cultivos mexicanos (INIFAP). Predice los cultivos más compatibles con las propiedades del suelo del agricultor.
- **Archivo:** `models/crop_classifier.py`

### 2. Álgebra Lineal — Vectorización y Déficit 5D
Representa suelo y cultivo como vectores en R⁵ = [N, P, K, pH, MO]. El déficit se calcula como resta de vectores: **ΔV = V_demanda − V_suelo**. Solo los componentes positivos generan recomendación de fertilización.
- **Archivo:** `models/fertilizer_recommender_v2.py`

### 3. Cálculo Diferencial — Optimización de Dosis
Modelo de beneficio neto **B(u) = R(u) − D(u)** donde R(u) = a·ln(u+1)+b (rendimientos decrecientes) y D(u) = c·u² (degradación cuadrática). El punto óptimo **u*** se encuentra derivando e igualando a cero: **B'(u) = 0**.
- **Archivo:** `app.py` (endpoint `/api/fertilizer/optimization-for-crop`)

### 4. Estadística — Promedios, Proyección y Prueba de Hipótesis
- **Promedio de muestras:** Media aritmética de múltiples muestras por parcela.
- **Prueba de hipótesis:** t-Student (n<30) o Z Normal (n≥30) para validar si el déficit detectado es estadísticamente significativo.
- **Proyección estacional:** Simula 5 temporadas de agotamiento del suelo sin reposición.
- **Archivos:** `app.py`, `templates/csv_analisis.html` (cálculos en JavaScript)

---

## Declaratoria de Uso de IA

En la elaboración de este prototipo se utilizó **Gemini** (Google DeepMind) como herramienta de apoyo para:

1. **Generación y depuración de código:** Asistencia en la escritura de funciones de Python y JavaScript, corrección de errores de sintaxis y lógica en los módulos del backend (`app.py`, `models/`, `utils/`) y en las plantillas HTML del frontend (`templates/`).

2. **Diseño de la interfaz web:** Sugerencias para la maquetación con Bootstrap, estilos CSS, y la lógica de renderizado dinámico de resultados en JavaScript.

El contenido final fue revisado, validado y adaptado por el equipo. La lógica del negocio (datos agronómicos, umbrales de INIFAP, catálogo de fertilizantes mexicanos y dataset de cultivos) fue investigada y curada manualmente por los integrantes del equipo.


