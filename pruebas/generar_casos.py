# ============================================================================
# Prueba de paridad Flask vs. motor JavaScript (parte 1 de 2).
# Genera peticiones variadas, las ejecuta contra el backend Flask original
# y guarda petición + respuesta esperada en pruebas/salida/casos.json.
#
# Uso (desde la raíz del repo):
#     app\venv\Scripts\python pruebas\generar_casos.py
#     node pruebas\comparar.js
# ============================================================================
import io
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "app"))

import pandas as pd  # noqa: E402
from app import app, SOIL_FEATURES, SOIL_TEXTURES  # noqa: E402

SALIDA = RAIZ / "pruebas" / "salida"
SALIDA.mkdir(exist_ok=True)
cliente = app.test_client()
rnd = random.Random(2026)
casos = []


def registrar(tipo, ruta, peticion, resp):
    casos.append({"tipo": tipo, "ruta": ruta, "peticion": peticion,
                  "estado": resp.status_code, "respuesta": resp.get_json(silent=True)})
    return resp.get_json(silent=True)


def optimizaciones_de(cuerpo):
    """Replica las llamadas que hace csv_analisis.html para la gráfica."""
    if not cuerpo or not cuerpo.get("exito"):
        return
    rs = [cuerpo["resultado"]] if cuerpo.get("modo") == "unico" else cuerpo.get("resultados", [])
    for r in rs:
        plan = r.get("plan_fertilizacion")
        if not plan:
            continue
        bags = sum(float(f.get("numero_costales_50kg") or 0) for f in plan.get("fertilizantes", []))
        enviar_json("/api/fertilizer/optimization-for-crop",
                    {"cultivo": plan["cultivo"], "propiedades_suelo": r.get("datos_entrada") or {},
                     "total_bags": bags})


def enviar_csv(texto, nombre="datos.csv"):
    resp = cliente.post("/api/csv/analizar",
                        data={"archivo": (io.BytesIO(texto.encode("utf-8")), nombre)},
                        content_type="multipart/form-data")
    cuerpo = registrar("csv", "/api/csv/analizar", {"nombre": nombre, "texto": texto}, resp)
    optimizaciones_de(cuerpo)


def enviar_json(ruta, datos):
    resp = cliente.post(ruta, json=datos)
    return registrar("json", ruta, {"json": datos}, resp)


# 1. Demo y CSV de ejemplo -------------------------------------------------
optimizaciones_de(registrar("demo", "/api/csv/demo", {}, cliente.post("/api/csv/demo")))
for f in ["Analisis_1Parcelas_1MUestra.csv", "Analisis_VariasParcelas_VariasMUestra.csv"]:
    enviar_csv((RAIZ / "app" / f).read_text(encoding="utf-8"))

# 2. Filas reales del dataset de entrenamiento con ruido --------------------
df_c = pd.read_csv(RAIZ / "app" / "cultivos_mexico_v3_final.csv")
COLS = SOIL_FEATURES + ["textura_suelo"]


def fila_aleatoria(ruido=0.15):
    base = df_c.iloc[rnd.randrange(len(df_c))]
    fila = {}
    for c in SOIL_FEATURES:
        v = float(base[c])
        if c in ("N_mgkg", "P_mgkg", "K_mgkg"):
            v = v / 2.0  # el dataset está en kg/ha, el usuario captura mg/kg
        v *= 1 + rnd.uniform(-ruido, ruido)
        fila[c] = round(v, rnd.choice([1, 2, 3]))
    fila["textura_suelo"] = base["textura_suelo"]
    return fila


def a_csv(filas, columnas, id_col="parcela", ids=None):
    lineas = [",".join(([id_col] if ids else []) + columnas)]
    for i, f in enumerate(filas):
        vals = [str(f[c]) for c in columnas]
        lineas.append(",".join(([str(ids[i])] if ids else []) + vals))
    return "\n".join(lineas) + "\n"


for _ in range(250):
    enviar_csv(a_csv([fila_aleatoria()], COLS))

# 3. Varias parcelas con varias muestras ------------------------------------
for _ in range(40):
    filas, ids = [], []
    for p in range(rnd.randint(2, 5)):
        for m in range(rnd.randint(1, 4)):
            filas.append(fila_aleatoria(0.3))
            ids.append(rnd.choice([f"Parcela_{p}_M{m+1}", f"Lote {p} muestra {m+1}", f"P{p}-S{m+1}", f"Campo{p}"]))
    enviar_csv(a_csv(filas, COLS, ids=ids))

# 4. Casos raros -------------------------------------------------------------
f0 = fila_aleatoria()
raros = [
    a_csv([f0], COLS[:5]),                                   # faltan columnas
    a_csv([fila_aleatoria() for _ in range(3)], COLS),       # sin id, varias filas
    a_csv([fila_aleatoria() for _ in range(4)], COLS, "id", [1, 1, 2, 2]),  # ids numéricos
    a_csv([fila_aleatoria() for _ in range(2)], COLS, "id_parcela", ["A_M1", "A_M2"]),
    a_csv([{**f0, "textura_suelo": "Franco Arcilloso"}], COLS),
    a_csv([{**f0, "textura_suelo": "arenoso"}], COLS),      # textura que el encoder no conoce
    a_csv([{**f0, "textura_suelo": "lodo"}], COLS),         # textura inválida
    a_csv([{**f0, "ph": 12}], COLS),                         # fuera de rango
    a_csv([{**f0, "ph": "abc"}], COLS),                      # texto en columna numérica
    a_csv([{**f0, "N_mgkg": ""}], COLS),                     # celda vacía
    a_csv([{**f0, "N_mgkg": ""}, fila_aleatoria()], COLS, ids=["X_M1", "X_M2"]),
    a_csv([{**f0, "Ca_mgkg": 4800}], COLS),
    a_csv([{**f0, "mo_pct": 0}], COLS),
    "﻿" + a_csv([f0], COLS).replace("\n", "\r\n"),     # BOM + CRLF
    a_csv([f0], COLS).replace("franco", '"franco"'),
    ",".join(COLS) + "\n",                                    # solo encabezado
    "",                                                        # vacío
    a_csv([f0], COLS) + "1,2,3" + ",9" * 30 + "\n",           # fila con campos de más
    a_csv([f0], COLS) + "\n\n",
]
for t in raros:
    enviar_csv(t)
enviar_csv(a_csv([f0], COLS), "datos.txt")
enviar_csv(a_csv([f0], COLS), "DATOS.CSV")

# 5. Modo manual -------------------------------------------------------------
cultivos = sorted(df_c["cultivo"].unique())
for _ in range(150):
    datos = {
        "cultivos_previos": rnd.sample(cultivos + ["Maiz grano blanco", "desconocido"], rnd.randint(1, 3)),
        "tipo_suelo": rnd.choice(SOIL_TEXTURES + ["Franco Arcilloso"]),
        "ph": rnd.choice(["5.5", "6.5", "7.5", "8.0", 6.8]),
        "materia_organica": rnd.choice(["baja", "media", "alta"]),
        "precipitacion_mm": str(rnd.randint(200, 2500)),
        "temp_media_c": str(rnd.randint(8, 32)),
        "altitud_msnm": str(rnd.randint(0, 3000)),
        "ce_ds_m": 0.8,
    }
    cuerpo = enviar_json("/api/manual/inferir", datos)
for datos in [{"cultivos_previos": [], "tipo_suelo": "franco"},
              {"cultivos_previos": ["maiz_grano_blanco"], "tipo_suelo": ""},
              {"cultivos_previos": ["maiz_grano_blanco"], "tipo_suelo": "franco", "ph": ""},
              {"cultivos_previos": ["maiz_grano_blanco"], "tipo_suelo": "franco", "ph": "11"}]:
    enviar_json("/api/manual/inferir", datos)

# 6. Optimización con entradas directas -------------------------------------
for _ in range(60):
    enviar_json("/api/fertilizer/optimization-for-crop",
                {"cultivo": rnd.choice(cultivos + ["maiz"]), "propiedades_suelo": fila_aleatoria(),
                 "total_bags": rnd.choice([0, 0.5, 1.37, 4, 12.25])})

with open(SALIDA / "casos.json", "w", encoding="utf-8") as f:
    json.dump(casos, f, ensure_ascii=False)
print(f"{len(casos)} casos guardados en {SALIDA / 'casos.json'}")
