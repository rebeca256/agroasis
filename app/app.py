# ============================================================================
# Agroasis: Sistema de Recomendación, Clasificación y Optimización
#           de Fertilizante en Parcelas Agrícolas

# Descripción:
#   Sistema web (Flask) que permite al agricultor analizar las propiedades
#   fisicoquímicas de su suelo, clasificar cultivos compatibles mediante
#   Machine Learning (Random Forest), calcular el déficit nutricional
#   usando Álgebra Lineal (vectorización 5D), optimizar la dosis de
#   fertilizante con Cálculo Diferencial, y proyectar la evolución
#   del suelo a lo largo de múltiples temporadas (Estadística).
#
# Rutas principales:
#   GET  /                -> Selector de modalidades (index)
#   GET  /csv-analisis    -> Pipeline CSV: 1 o N parcelas
#   GET  /manual          -> Inferencia sin datos de laboratorio
#   POST /api/csv/analizar              -> Análisis desde archivo CSV
#   POST /api/csv/demo                  -> Análisis con CSV de ejemplo
#   POST /api/manual/inferir            -> Modo manual / simple
#   POST /api/fertilizer/optimization   -> Optimización diferencial
# ============================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import io
import re
from datetime import datetime

from config import FLASK_CONFIG, SOIL_FEATURES, SOIL_TEXTURES
from utils.data_loader import cargar_y_preparar_datos_principales
from utils.validators import DataValidator, InputSanitizer
from models.crop_classifier import CropClassifier
from models.soil_analyzer import SoilAnalyzer
from models.fertilizer_recommender_v2 import obtener_recomendador

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config.update(FLASK_CONFIG)

CLASIFICADOR = None
DATA_LOADER  = None


def inicializar_aplicacion():
    global CLASIFICADOR, DATA_LOADER
    print("\n" + "=" * 60)
    print("  Agroasis - Inicializando aplicacion")
    print("=" * 60)
    DATA_LOADER = cargar_y_preparar_datos_principales()
    CLASIFICADOR = CropClassifier()
    if not CLASIFICADOR.modelos_guardados:
        print("\n Entrenando modelo por primera vez...")
        df_limpio = DATA_LOADER.limpiar_datos(DATA_LOADER.df)
        CLASIFICADOR.entrenar(df_limpio)
    else:
        print("\n Modelos entrenados cargados correctamente")
    print("\n Aplicacion lista para usar\n")


inicializar_aplicacion()


# ---------------------------------------------------------------------------
# Helpers compartidos
# Funciones auxiliares reutilizadas por todos los endpoints.
# ---------------------------------------------------------------------------

_ESTADO_COLOR = {
    "excelente": "#27ae60", "bueno": "#9acd32",
    "aceptable": "#f39c12", "pobre": "#e67e22", "muy_pobre": "#e74c3c",
}

def _estado_a_color(estado: str) -> str:
    return _ESTADO_COLOR.get(estado.lower(), "#6c757d")


def _normalizar_textura(valor: str) -> str:
    """Franco Arcilloso -> franco_arcilloso, ARENOSO -> arenoso."""
    return str(valor).lower().strip().replace(' ', '_').replace('-', '_')


def _completar_columnas_faltantes(df: pd.DataFrame):
    """Rellena columnas faltantes con valores neutrales (validación permisiva)."""
    valores_neutros = {
        "ph": 7.0, "mo_pct": 2.5, "N_mgkg": 45.0, "P_mgkg": 15.0,
        "K_mgkg": 120.0, "Ca_mgkg": 1500.0, "Mg_mgkg": 250.0,
        "S_mgkg": 25.0, "Fe_mgkg": 12.0, "Zn_mgkg": 1.5,
        "Mn_mgkg": 8.0, "Cu_mgkg": 0.8, "B_mgkg": 0.5,
        "ce_ds_m": 0.8, "precipitacion_mm": 750.0,
        "temp_media_c": 22.0, "altitud_msnm": 1500.0, "textura_suelo": "franco"
    }
    faltantes = []
    for col in SOIL_FEATURES + ["textura_suelo"]:
        if col not in df.columns:
            df[col] = valores_neutros.get(col, 0.0)
            faltantes.append(col)
    if "textura_suelo" in df.columns:
        df["textura_suelo"] = df["textura_suelo"].apply(_normalizar_textura)
    return df, faltantes


def _cargar_csv_desde_request(archivo) -> pd.DataFrame:
    """Lee CSV desde FileStorage de Flask."""
    stream = io.StringIO(archivo.stream.read().decode("utf-8", errors="replace"), newline=None)
    df = pd.read_csv(stream)
    if df.empty:
        raise ValueError("El archivo CSV está vacío.")
    return df


def _analizar_parcela_desde_dict(datos: dict, hectareas: float = 1.0) -> dict:
    """
    Núcleo de análisis de una parcela (DRY).
    Retorna analisis_suelo + plan_fertilizacion.
    """
    if "textura_suelo" in datos:
        datos["textura_suelo"] = _normalizar_textura(datos["textura_suelo"])

    # Validación de rangos físicos (ph 3-10, nutrientes 0-2000, etc.)
    es_valido, datos_limpios, errores = DataValidator.sanitizar_diccionario_suelo(datos)
    if not es_valido:
        raise ValueError(f"Datos inválidos: {errores}")

    # Valores NaN residuales se reemplazan por un neutro para no romper el modelo
    for key in datos_limpios:
        if isinstance(datos_limpios[key], float) and pd.isna(datos_limpios[key]):
            datos_limpios[key] = 50.0

    # [Machine Learning] Clasificación de cultivos con Random Forest
    analisis = SoilAnalyzer().analizar_suelo(datos_limpios)
    cultivo_top = (analisis.get("cultivos_compatibles") or [None])[0]
    plan_fertilizacion = None

    # [Álgebra Lineal] Vectorización 5D y cálculo de déficit
    if cultivo_top:
        recom = obtener_recomendador().recomendar_fertilizantes(
            cultivo=cultivo_top["cultivo"],
            propiedades_suelo=datos_limpios,
            hectareas=hectareas
        )
        plan_fertilizacion = {
            "cultivo":              cultivo_top["cultivo"],
            "analisis_deficit":     recom["analisis_deficit"],
            "fertilizantes":        recom["recomendaciones"]["fertilizantes"],
            "costo_total_estimado": recom["recomendaciones"]["costo_total"],
            "mensaje":              recom["recomendaciones"].get("mensaje", ""),
            "hectareas":            hectareas,
        }

    return {
        "analisis_suelo":    analisis,
        "plan_fertilizacion": plan_fertilizacion,
        "datos_entrada": {k: float(v) if not isinstance(v, str) else v for k, v in datos_limpios.items()},
    }


def _inferir_propiedades_por_cultivo(cultivos_cultivados: list) -> dict:
    """Infiere propiedades del suelo desde cultivos previos (modo manual)."""
    rec = obtener_recomendador()
    cultivo = cultivos_cultivados[0].lower().replace(" ", "_") if cultivos_cultivados else "maiz_grano_blanco"
    demanda = rec.demandas_cultivo.get(
        cultivo,
        list(rec.demandas_cultivo.values())[0] if rec.demandas_cultivo
        else {"n": 150, "p": 50, "k": 100, "ph": 6.5, "mo": 2.5}
    )
    return {
        "N_mgkg": float(demanda.get("n", 150)) / 2.0,
        "P_mgkg": float(demanda.get("p",  50)) / 2.0,
        "K_mgkg": float(demanda.get("k", 100)) / 2.0,
        "mo_pct": float(demanda.get("mo", 2.5)),
        "ph":     float(demanda.get("ph", 6.5)),
    }


# ---------------------------------------------------------------------------
# Paginas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/csv-analisis")
def csv_analisis_mode():
    return render_template("csv_analisis.html")

@app.route("/manual")
def manual_mode():
    return render_template("manual_mode.html", soil_textures=SOIL_TEXTURES)


# ---------------------------------------------------------------------------
# API: CSV Unificado (pipeline principal)
# [Estadística] Agrupa muestras por parcela, calcula media aritmética
#   y genera perfil representativo para cada parcela.
# ---------------------------------------------------------------------------

def _procesar_df_parcelas(df: pd.DataFrame):
    """Procesa un DataFrame con 1 o N parcelas. Agrupa por id_parcela,
    promedia las muestras de cada grupo y ejecuta el análisis completo."""
    df, columnas_faltantes = _completar_columnas_faltantes(df)

    # Detectar columna de agrupación por parcela
    col_parcela = next((c for c in ["id_parcela", "parcela", "id"] if c in df.columns), None)

    def _prefijo_parcela(nombre: str) -> str:
        sufijo = re.sub(r'[\s_\-]*(m|muestra|s|sample|r)?\s*\d+\s*$', '', str(nombre), flags=re.IGNORECASE).strip()
        return sufijo if sufijo else str(nombre)

    if col_parcela:
        df["_grupo"] = df[col_parcela].apply(lambda v: _prefijo_parcela(str(v)))
        grupos = {p: df[df["_grupo"] == p] for p in df["_grupo"].unique()}
    elif len(df) == 1:
        grupos = {"Parcela_1": df}
    else:
        grupos = {f"Parcela_{i+1}": df.iloc[[i]] for i in range(len(df))}

    es_unico = len(grupos) == 1
    resultados = []

    for nombre, df_grupo in grupos.items():
        muestras_individuales = []
        for _, fila in df_grupo.iterrows():
            muestra = {col: float(fila[col]) for col in SOIL_FEATURES if col in fila}
            if "textura_suelo" in fila:
                muestra["textura_suelo"] = fila["textura_suelo"]
            try:
                muestras_individuales.append(_analizar_parcela_desde_dict(muestra))
            except Exception:
                pass

        datos_prom = {col: float(df_grupo[col].mean()) for col in SOIL_FEATURES if col in df_grupo.columns}
        if "textura_suelo" in df_grupo.columns:
            datos_prom["textura_suelo"] = df_grupo["textura_suelo"].mode()[0]

        try:
            analisis = _analizar_parcela_desde_dict(datos_prom)
            estado = analisis.get("analisis_suelo", {}).get("salud_general", {}).get("estado", "pobre")
            analisis.update({
                "parcela": nombre, "num_muestras": len(df_grupo),
                "estado_suelo": estado, "color": _estado_a_color(estado),
                "muestras_individuales": muestras_individuales,
            })
            resultados.append(analisis)
        except Exception as e:
            resultados.append({
                "parcela": nombre, "num_muestras": len(df_grupo),
                "error": str(e), "estado_suelo": "error",
                "color": "#6c757d", "muestras_individuales": [],
            })

    if es_unico:
        r = resultados[0]
        if not r.get("parcela"):
            r["parcela"] = list(grupos.keys())[0]
        return jsonify({"exito": True, "modo": "unico", "resultado": r,
                        "columnas_usadas_por_defecto": columnas_faltantes})

    matriz = [{"parcela": r["parcela"], "estado": r.get("estado_suelo", "?"),
               "color": r.get("color", "#6c757d"),
               "score": r.get("analisis_suelo", {}).get("salud_general", {}).get("score", 0)}
              for r in resultados]
    return jsonify({"exito": True, "modo": "multiple", "total_parcelas": len(resultados),
                    "resultados": resultados, "matriz_colores": matriz,
                    "columnas_usadas_por_defecto": columnas_faltantes})

@app.route("/api/csv/analizar", methods=["POST"])
def api_csv_analizar():
    """
    Pipeline unificado de ingesta CSV.
    - 1 parcela  -> modo 'único'
    - N parcelas -> modo 'múltiple' (agrupadas por id_parcela si existe)
    """
    try:
        if "archivo" not in request.files:
            return jsonify({"error": "No se proporcionó archivo."}), 400

        archivo = request.files["archivo"]
        if not archivo.filename.endswith(".csv"):
            return jsonify({"error": "Formato no válido. Use CSV."}), 400

        df = _cargar_csv_desde_request(archivo)
        return _procesar_df_parcelas(df)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/csv/demo", methods=["POST"])
def api_csv_demo():
    """Endpoint directo para cargar el CSV de ejemplo sin subir archivo."""
    try:
        import os
        ruta = os.path.join(app.root_path, "Analisis_VariasParcelas_VariasMUestra.csv")
        if not os.path.exists(ruta):
            return jsonify({"error": "No se encontró el archivo de ejemplo localmente."}), 404
            
        df = pd.read_csv(ruta)
        if df.empty:
            raise ValueError("El archivo CSV de ejemplo está vacío.")
        
        return _procesar_df_parcelas(df)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Simple (una parcela via JSON o CSV)
# ---------------------------------------------------------------------------

@app.route("/api/simple/analizar", methods=["POST"])
def api_simple_analizar():
    try:
        if "archivo" in request.files:
            archivo = request.files["archivo"]
            if not archivo.filename.endswith(".csv"):
                return jsonify({"error": "Formato no válido. Use CSV."}), 400
            datos = _cargar_csv_desde_request(archivo).iloc[0].to_dict()
        else:
            datos = request.json

        if not datos:
            return jsonify({"error": "No se proporcionaron datos."}), 400

        hectareas = float(datos.pop("hectareas", 1.0))
        return jsonify({"exito": True, **_analizar_parcela_desde_dict(datos, hectareas)})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Manual (inferencia sin laboratorio)
# Permite al agricultor obtener una recomendación aproximada sin datos
# de laboratorio, usando cultivos previos y observaciones de campo.
# ---------------------------------------------------------------------------

@app.route("/api/manual/inferir", methods=["POST"])
def api_manual_inferir():
    """Infiere propiedades del suelo a partir de cultivos previos y clima."""
    try:
        datos            = request.json
        cultivos_previos = datos.get("cultivos_previos", [])
        tipo_suelo       = datos.get("tipo_suelo")

        if not cultivos_previos or not tipo_suelo:
            return jsonify({"error": "Información incompleta."}), 400

        propiedades = _inferir_propiedades_por_cultivo(cultivos_previos)
        propiedades.update({
            "textura_suelo":    tipo_suelo,
            "ph":               float(datos.get("ph", 6.5)),
            "precipitacion_mm": float(datos.get("precipitacion_mm", 700)),
            "temp_media_c":     float(datos.get("temp_media_c", 20)),
            "altitud_msnm":     float(datos.get("altitud_msnm", 1500)),
            "ce_ds_m":          float(datos.get("ce_ds_m", 0.8)),
        })
        for feat in SOIL_FEATURES:
            propiedades.setdefault(feat, 50)

        return jsonify({
            "exito": True,
            "propiedades_inferidas": propiedades,
            **_analizar_parcela_desde_dict(propiedades),
            "advertencia": "Resultados basados en inferencias. Se recomienda análisis de laboratorio.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Optimización diferencial de fertilizante
# [Cálculo Diferencial] Modelo de beneficio neto B(u) = R(u) - D(u)
#   R(u) = a*ln(u+1) + b   (rendimiento con retornos decrecientes)
#   D(u) = c*u^2            (degradación cuadrática por exceso)
#   B'(u) = 0 => punto óptimo u* via fórmula cuadrática
# ---------------------------------------------------------------------------

@app.route("/api/fertilizer/optimization-for-crop", methods=["POST"])
def api_fertilizer_optimization_for_crop():
    """Calcula la dosis óptima de fertilizante donde el beneficio neto
    es máximo, usando derivadas para encontrar el punto crítico."""
    try:
        datos             = request.json
        cultivo           = datos.get("cultivo", "maiz")
        propiedades_suelo = datos.get("propiedades_suelo", {})
        total_bags        = float(datos.get("total_bags", 0))

        analisis    = obtener_recomendador().calcular_deficit(cultivo, propiedades_suelo)
        deficit_total = sum(analisis["deficit_aplicable"][:3])  # N + P + K
        exceso_total  = sum(analisis["exceso"].values())

        # Parámetros dinámicos del modelo diferencial
        a = max(1.0, 1.5 + deficit_total / 100.0)  # tasa de respuesta
        b = max(1.0, 5.0 - exceso_total  / 50.0)   # rendimiento base

        if total_bags == 0:
            c = 0.1; u_opt = 0.0
        else:
            # c calibrado para que u* coincida con la recomendación vectorial
            c = a / (2 * total_bags * (total_bags + 1))
            disc = c**2 + 2 * c * a  # discriminante de la cuadrática
            u_opt = (-c + np.sqrt(disc)) / (2 * c) if disc > 0 else 0.0

        # Generación de 200 puntos para las curvas
        u   = np.linspace(0, max(40, u_opt * 1.5), 200)
        R   = a * np.log(u + 1) + b  # curva de rendimiento (logarítmica)
        D   = c * u ** 2
        B   = R - D
        idx = int(np.argmin(np.abs(u - u_opt)))

        return jsonify({
            "exito": True, "cultivo": cultivo,
            "u_values": u.tolist(), "rendimiento": R.tolist(),
            "degradacion": D.tolist(), "beneficio_neto": B.tolist(),
            "punto_optimo": {
                "dosis":         round(float(u_opt), 2),
                "rendimiento":   round(float(R[idx]), 3),
                "beneficio_neto": round(float(B[idx]), 2),
                "costo_acumulado": round(float(u_opt * 500), 2),
            },
            "parametros": {"a": round(a, 3), "b": round(b, 3), "c": round(c, 4)},
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Catálogo e info
# ---------------------------------------------------------------------------

@app.route("/api/fertilizantes")
def api_fertilizantes_lista():
    rec = obtener_recomendador()
    return jsonify([
        {"id": r["id_fertilizante"], "nombre": r["nombre_comercial"],
         "n": r["n"], "p": r["p"], "k": r["k"],
         "precio_costal": r["precio_mxn"], "peso_costal": r["peso_costal_kg"]}
        for _, r in rec.df_fertilizantes.iterrows()
    ])

@app.route("/api/info/soil-features")
def api_info_soil_features():
    return jsonify({"features": SOIL_FEATURES, "total": len(SOIL_FEATURES)})

@app.route("/api/info/textures")
def api_info_textures():
    return jsonify(SOIL_TEXTURES)

@app.route("/salud")
def health_check():
    return jsonify({
        "estado": "activo",
        "timestamp": datetime.now().isoformat(),
        "clasificador_entrenado": CLASIFICADOR.modelos_guardados if CLASIFICADOR else False,
    })


# ---------------------------------------------------------------------------
# Manejo de errores
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def no_encontrado(e):
    return jsonify({"error": "Ruta no encontrada."}), 404

@app.errorhandler(500)
def error_servidor(e):
    return jsonify({"error": "Error interno del servidor."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
