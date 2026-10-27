# ============================================================================
# Módulo: build_web.py
# Descripción: Genera la versión estática de Agroasis (carpeta ../web) que
#   corre completamente en el navegador, sin servidor:
#     1. Exporta el Random Forest, el scaler, el encoder, las demandas por
#        cultivo y el catálogo de fertilizantes a JSON.
#     2. Renderiza las plantillas de Flask a HTML estático.
#     3. Inyecta el motor JavaScript (static/js/agroasis-engine.js), que
#        intercepta las llamadas a /api/... y las resuelve en el navegador.
#
# Uso (desde la carpeta app/):
#     python build_web.py
#
# Volver a ejecutarlo cada vez que se reentrene el modelo o se modifiquen
# las plantillas o los CSV.
# ============================================================================
import json
import shutil
from pathlib import Path

import numpy as np

from app import app, CLASIFICADOR
from models.fertilizer_recommender_v2 import obtener_recomendador

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR.parent / "web"
DATA_DIR = WEB_DIR / "static" / "data"

PAGINAS = {
    "/": "index.html",
    "/csv-analisis": "csv-analisis.html",
    "/manual": "manual.html",
}


def exportar_modelo() -> dict:
    """Random Forest -> JSON. Cada árbol se guarda como arreglos planos;
    las hojas guardan solo las probabilidades distintas de cero, ya
    normalizadas igual que sklearn (value / value.sum())."""
    modelo = CLASIFICADOR.modelo
    arboles = []
    for est in modelo.estimators_:
        t = est.tree_
        hojas = {}
        for nodo in np.where(t.children_left == -1)[0]:
            v = t.value[nodo, 0, :]
            s = v.sum()
            v = v / (s if s != 0 else 1.0)
            hojas[int(nodo)] = [x for i in np.nonzero(v)[0] for x in (int(i), float(v[i]))]
        arboles.append({
            "izq": t.children_left.tolist(),
            "der": t.children_right.tolist(),
            "var": t.feature.tolist(),
            "umbral": t.threshold.tolist(),
            "hojas": hojas,
        })
    return {
        "clases": [str(c) for c in modelo.classes_],
        "arboles": arboles,
        "scaler_media": CLASIFICADOR.scaler.mean_.tolist(),
        "scaler_escala": CLASIFICADOR.scaler.scale_.tolist(),
        "texturas_encoder": [str(c) for c in CLASIFICADOR.encoder_textura.classes_],
    }


def exportar_recomendador() -> dict:
    rec = obtener_recomendador()
    return {
        # Lista (no dict) para conservar el orden de pandas.groupby
        "demandas_cultivo": [[k, v] for k, v in rec.demandas_cultivo.items()],
        "fertilizantes": [
            {
                "id": int(r["id_fertilizante"]),
                "nombre": str(r["nombre_comercial"]),
                "n": float(r["n"]), "p": float(r["p"]), "k": float(r["k"]),
                "precio_kg": float(r["precio_por_kg"]),
                "precio_costal": float(r["precio_mxn"]),
                "peso_costal": float(r["peso_costal_kg"]),
                "clasificacion": str(r["clasificacion_objetivo"]),
                "funcion": str(r["funcion_algoritmica"]),
            }
            for _, r in rec.df_fertilizantes.iterrows()
        ],
    }


def convertir_html(html: str) -> str:
    """Ajusta rutas absolutas de Flask a rutas relativas de sitio estático
    e inyecta el motor JS antes de cualquier otro script."""
    for ruta, archivo in PAGINAS.items():
        html = html.replace(f'href="{ruta}"', f'href="{archivo}"')
    html = html.replace('"/static/', '"static/')
    motor = '    <script src="static/js/agroasis-engine.js"></script>\n'
    return html.replace("</head>", motor + "</head>", 1)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "modelo.json", "w", encoding="utf-8") as f:
        json.dump(exportar_modelo(), f, separators=(",", ":"))
    with open(DATA_DIR / "recomendador.json", "w", encoding="utf-8") as f:
        json.dump(exportar_recomendador(), f, ensure_ascii=False, separators=(",", ":"))
    shutil.copy(APP_DIR / "Analisis_VariasParcelas_VariasMUestra.csv", DATA_DIR / "demo.csv")

    shutil.copytree(APP_DIR / "static" / "img", WEB_DIR / "static" / "img", dirs_exist_ok=True)

    cliente = app.test_client()
    for ruta, archivo in PAGINAS.items():
        html = cliente.get(ruta).get_data(as_text=True)
        (WEB_DIR / archivo).write_text(convertir_html(html), encoding="utf-8")

    print(f"Sitio estático generado en {WEB_DIR}")


if __name__ == "__main__":
    main()
