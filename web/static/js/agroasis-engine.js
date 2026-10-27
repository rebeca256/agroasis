// ============================================================================
// agroasis-engine.js
// Motor de Agroasis en el navegador: réplica en JavaScript del backend Flask
// (app.py, models/, utils/). Intercepta las llamadas fetch("/api/...") de las
// páginas y las resuelve localmente, con las mismas respuestas JSON que
// devolvería el servidor.
//
// Los datos (Random Forest, scaler, encoder, demandas, fertilizantes) los
// genera app/build_web.py en static/data/. No editar a mano esos JSON.
//
// Funciona también en Node (para las pruebas de paridad con Python):
//   const motor = require("./agroasis-engine.js"); motor.inicializar(datos)
// ============================================================================
(function (raiz) {
  "use strict";

  // ---------------------------------------------------------------------------
  // config.py
  // ---------------------------------------------------------------------------
  const SOIL_FEATURES = [
    "ph", "mo_pct", "N_mgkg", "P_mgkg", "K_mgkg",
    "Ca_mgkg", "Mg_mgkg", "S_mgkg",
    "Fe_mgkg", "Zn_mgkg", "Mn_mgkg", "Cu_mgkg", "B_mgkg",
    "ce_ds_m", "precipitacion_mm", "temp_media_c", "altitud_msnm",
  ];

  const NUTRIENT_LIMITS = {
    N_mgkg: { bajo: 20, medio: 50, alto: 100 },
    P_mgkg: { bajo: 10, medio: 25, alto: 50 },
    K_mgkg: { bajo: 100, medio: 250, alto: 500 },
    Ca_mgkg: { bajo: 1000, medio: 2500, alto: 5000 },
    Mg_mgkg: { bajo: 100, medio: 250, alto: 500 },
    S_mgkg: { bajo: 10, medio: 25, alto: 50 },
  };

  const SOIL_TEXTURES = [
    "arenoso", "franco_arenoso", "franco", "franco_limoso", "limoso",
    "franco_arcilloso", "arcillo_arenoso", "arcillo_limoso", "arcilloso",
  ];

  // ---------------------------------------------------------------------------
  // Utilidades que imitan el comportamiento de Python / NumPy / pandas
  // ---------------------------------------------------------------------------

  class PyError extends Error {
    constructor(tipo, mensaje) { super(mensaje); this.tipo = tipo; }
  }

  // repr() de un str de Python (suficiente para los mensajes de error)
  function pyRepr(s) {
    const comilla = s.includes("'") && !s.includes('"') ? '"' : "'";
    const esc = s.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
    return comilla + (comilla === "'" ? esc.replace(/'/g, "\\'") : esc) + comilla;
  }

  // Sintaxis aceptada por float(str) en Python
  const RE_PY_FLOAT = /^[+-]?(?:(?:\d(?:_?\d)*)?\.?\d(?:_?\d)*(?:[eE][+-]?\d(?:_?\d)*)?|\d(?:_?\d)*\.(?:[eE][+-]?\d(?:_?\d)*)?|inf(?:inity)?|nan)$/i;

  function pyTypeName(v) {
    if (v === null || v === undefined) return "NoneType";
    if (Array.isArray(v)) return "list";
    if (typeof v === "object") return "dict";
    return typeof v;
  }

  // float(v) de Python
  function pyFloat(v) {
    if (typeof v === "number") return v;
    if (typeof v === "boolean") return v ? 1 : 0;
    if (typeof v === "string") {
      const s = v.trim();
      if (RE_PY_FLOAT.test(s)) {
        const limpio = s.replace(/_/g, "").toLowerCase();
        const signo = limpio.startsWith("-") ? -1 : 1;
        const cuerpo = limpio.replace(/^[+-]/, "");
        if (cuerpo === "nan") return NaN;
        if (cuerpo === "inf" || cuerpo === "infinity") return signo * Infinity;
        return Number(limpio);
      }
      throw new PyError("ValueError", `could not convert string to float: ${pyRepr(v)}`);
    }
    throw new PyError("TypeError", `float() argument must be a string or a real number, not '${pyTypeName(v)}'`);
  }

  // Representación decimal exacta de un double (no exponencial)
  function decimalExacto(x) {
    // toFixed admite hasta 100 decimales; suficiente para detectar empates
    return Math.abs(x).toFixed(100);
  }

  // format(x, f".{n}f") de Python: redondeo exacto, empates al par
  function pyFixed(x, n) {
    if (!isFinite(x)) return isNaN(x) ? "nan" : (x > 0 ? "inf" : "-inf");
    const negativo = x < 0 || Object.is(x, -0);
    const exacto = decimalExacto(x);
    const [ent, frac] = exacto.split(".");
    const resto = frac.slice(n);
    let digitos = (ent + frac.slice(0, n)).split("").map(Number);
    let subir;
    if (resto[0] > "5") subir = true;
    else if (resto[0] < "5") subir = false;
    else if (/[1-9]/.test(resto.slice(1))) subir = true;
    else subir = digitos[digitos.length - 1] % 2 === 1; // empate exacto: al par
    if (subir) {
      let i = digitos.length - 1;
      while (i >= 0) { if (digitos[i] === 9) { digitos[i] = 0; i--; } else { digitos[i]++; break; } }
      if (i < 0) digitos.unshift(1);
    }
    const s = digitos.join("");
    const parteEnt = s.slice(0, s.length - n).replace(/^0+(?=\d)/, "") || "0";
    const cuerpo = n > 0 ? parteEnt + "." + s.slice(s.length - n) : parteEnt;
    return (negativo ? "-" : "") + cuerpo;
  }

  // round(x, n) de Python
  function pyRound(x, n) {
    if (!isFinite(x)) return x;
    return Number(pyFixed(x, n));
  }

  // round(x, n) sobre un np.float64: NumPy multiplica, aplica rint
  // (empates al par) y divide, que no siempre coincide con round() de Python
  function npRound(x, n) {
    const f = 10 ** n;
    const y = x * f;
    let r = Math.round(y);
    if (Math.abs(y % 1) === 0.5) r = 2 * Math.round(y / 2);
    return r / f;
  }

  // str(float) de Python
  function pyStrNumero(x, esEntero) {
    if (esEntero) return String(x);
    if (isNaN(x)) return "nan";
    if (!isFinite(x)) return x > 0 ? "inf" : "-inf";
    if (Number.isInteger(x) && Math.abs(x) < 1e16) return (Object.is(x, -0) ? "-0" : String(x)) + ".0";
    let s = String(x);
    s = s.replace(/e([+-])(\d)$/, "e$10$2"); // 1e-7 -> 1e-07
    return s;
  }

  // Suma por pares de NumPy (np.add.reduce sobre float64)
  function sumaPorPares(a, inicio, n) {
    if (n < 8) {
      let r = -0.0;
      for (let i = 0; i < n; i++) r += a[inicio + i];
      return r;
    }
    if (n <= 128) {
      const r = new Array(8);
      for (let j = 0; j < 8; j++) r[j] = a[inicio + j];
      let i;
      for (i = 8; i < n - (n % 8); i += 8) {
        for (let j = 0; j < 8; j++) r[j] += a[inicio + i + j];
      }
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[inicio + i];
      return res;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return sumaPorPares(a, inicio, n2) + sumaPorPares(a, inicio + n2, n - n2);
  }

  function npMean(valores) {
    return sumaPorPares(valores, 0, valores.length) / valores.length;
  }

  // np.linspace(inicio, fin, num)
  function linspace(inicio, fin, num) {
    const div = num - 1;
    const paso = (fin - inicio) / div;
    const y = new Array(num);
    for (let i = 0; i < num; i++) y[i] = (paso !== 0 ? i * paso : (i * (fin - inicio)) / div) + inicio;
    y[num - 1] = fin;
    return y;
  }

  // json.dumps(sort_keys=True) de Flask: ordena las llaves recursivamente
  function ordenarLlaves(v) {
    if (Array.isArray(v)) return v.map(ordenarLlaves);
    if (v && typeof v === "object") {
      const o = {};
      for (const k of Object.keys(v).sort()) o[k] = ordenarLlaves(v[k]);
      return o;
    }
    return v;
  }

  // ---------------------------------------------------------------------------
  // Lectura de CSV (subconjunto de pandas.read_csv con opciones por defecto)
  // ---------------------------------------------------------------------------

  const NA_VALUES = new Set([
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
  ]);
  const RE_ENTERO = /^\s*[+-]?\d+\s*$/;
  const RE_DECIMAL = /^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/;
  const RE_INF = /^\s*[+-]?inf(?:inity)?\s*$/i;

  function tokenizarCsv(texto) {
    texto = texto.replace(/\r\n?/g, "\n");
    if (texto.charCodeAt(0) === 0xfeff) texto = texto.slice(1);
    const filas = [];
    let campo = "", fila = [], enComillas = false, huboComillas = false;
    for (let i = 0; i < texto.length; i++) {
      const c = texto[i];
      if (enComillas) {
        if (c === '"') {
          if (texto[i + 1] === '"') { campo += '"'; i++; } else enComillas = false;
        } else campo += c;
      } else if (c === '"') { enComillas = true; huboComillas = true; }
      else if (c === ",") { fila.push(campo); campo = ""; }
      else if (c === "\n") {
        fila.push(campo);
        if (!(fila.length === 1 && fila[0] === "" && !huboComillas)) filas.push(fila);
        fila = []; campo = ""; huboComillas = false;
      } else campo += c;
    }
    if (campo !== "" || fila.length || huboComillas) {
      fila.push(campo);
      if (!(fila.length === 1 && fila[0] === "" && !huboComillas)) filas.push(fila);
    }
    return filas;
  }

  // Devuelve { columnas: [...], datos: {col: {tipo, valores}}, n }
  function leerCsv(texto) {
    const filas = tokenizarCsv(texto);
    if (filas.length === 0) throw new PyError("EmptyDataError", "No columns to parse from file");
    const encabezado = filas[0];
    const columnas = [];
    const vistos = {};
    encabezado.forEach((nombre, i) => {
      let n = nombre === "" ? `Unnamed: ${i}` : nombre;
      if (vistos[n] !== undefined) {
        let k = vistos[n] + 1;
        while (vistos[`${n}.${k}`] !== undefined) k++;
        vistos[n] = k;
        n = `${n}.${k}`;
      }
      vistos[n] = 0;
      columnas.push(n);
    });
    const cuerpo = filas.slice(1);
    cuerpo.forEach((f, i) => {
      if (f.length > columnas.length) {
        throw new PyError("ParserError",
          `Error tokenizing data. C error: Expected ${columnas.length} fields in line ${i + 2}, saw ${f.length}\n`);
      }
    });
    const datos = {};
    columnas.forEach((col, j) => {
      const crudos = cuerpo.map((f) => (j < f.length ? f[j] : ""));
      const esNA = crudos.map((v) => NA_VALUES.has(v));
      const presentes = crudos.filter((_, i) => !esNA[i]);
      let tipo, valores;
      if (presentes.length === 0) {
        tipo = "float"; valores = crudos.map(() => NaN);
      } else if (presentes.every((v) => /^(true|false)$/i.test(v.trim())) && !esNA.some(Boolean)) {
        tipo = "bool"; valores = crudos.map((v) => /^true$/i.test(v.trim()));
      } else if (presentes.every((v) => RE_ENTERO.test(v)) && !esNA.some(Boolean)) {
        tipo = "int"; valores = crudos.map((v) => Number(v.trim()));
      } else if (presentes.every((v) => RE_DECIMAL.test(v) || RE_INF.test(v))) {
        tipo = "float";
        valores = crudos.map((v, i) => {
          if (esNA[i]) return NaN;
          if (RE_INF.test(v)) return v.trim().startsWith("-") ? -Infinity : Infinity;
          return Number(v.trim());
        });
      } else {
        tipo = "str"; valores = crudos.map((v, i) => (esNA[i] ? NaN : v));
      }
      datos[col] = { tipo, valores };
    });
    return { columnas, datos, n: cuerpo.length };
  }

  // ---------------------------------------------------------------------------
  // Estado cargado desde static/data/
  // ---------------------------------------------------------------------------
  let MODELO = null;       // modelo.json
  let RECOM = null;        // recomendador.json
  let DEMO_CSV = null;     // texto del CSV de ejemplo
  let DEMANDAS = null;     // Map cultivo -> demanda (orden de pandas.groupby)

  function inicializar(datos) {
    MODELO = datos.modelo;
    RECOM = datos.recomendador;
    DEMO_CSV = datos.demoCsv;
    DEMANDAS = new Map(RECOM.demandas_cultivo);
  }

  function primeraDemanda() {
    return DEMANDAS.size ? DEMANDAS.values().next().value : null;
  }

  // ---------------------------------------------------------------------------
  // utils/validators.py — DataValidator
  // ---------------------------------------------------------------------------

  function validarNumerico(valor, nombre, rango) {
    let v;
    try { v = pyFloat(valor); } catch (e) { return [false, `${nombre} debe ser un número válido`]; }
    if (rango && (v < rango[0] || v > rango[1])) return [false, `${nombre} debe estar entre ${rango[0]} y ${rango[1]}`];
    return [true, v];
  }

  const VALIDADORES_ESPECIALES = {
    ph: (v) => validarNumerico(v, "pH", [3, 10]),
    mo_pct: (v) => validarNumerico(v, "Porcentaje", [0, 100]),
    temp_media_c: (v) => validarNumerico(v, "Temperatura", [-20, 50]),
    precipitacion_mm: (v) => validarNumerico(v, "Precipitación", [0, 5000]),
    altitud_msnm: (v) => validarNumerico(v, "Altitud", [0, 5000]),
    textura_suelo: (t) => (SOIL_TEXTURES.includes(t)
      ? [true, t]
      : [false, `Textura inválida. Opciones: ${SOIL_TEXTURES.join(", ")}`]),
  };

  function validarNutriente(valor, nombre) {
    const max = nombre.includes("Ca") || nombre.toLowerCase().includes("ca_") ? 5000 : 2000;
    return validarNumerico(valor, nombre, [0, max]);
  }

  function sanitizarDiccionarioSuelo(datos) {
    const errores = [];
    const limpios = {};
    for (const feature of SOIL_FEATURES.concat(["textura_suelo"])) {
      if (!(feature in datos)) { errores.push(`Falta el campo: ${feature}`); continue; }
      const validador = VALIDADORES_ESPECIALES[feature];
      const [ok, res] = validador ? validador(datos[feature]) : validarNutriente(datos[feature], feature);
      if (ok) limpios[feature] = res; else errores.push(res);
    }
    return [errores.length === 0, limpios, errores];
  }

  // ---------------------------------------------------------------------------
  // models/crop_classifier.py — Random Forest
  // ---------------------------------------------------------------------------

  function prepararEntrada(datos) {
    const textura = datos.textura_suelo;
    const cod = MODELO.texturas_encoder.indexOf(textura);
    if (cod < 0) return null; // LabelEncoder: etiqueta desconocida
    const x = [];
    for (const f of SOIL_FEATURES) {
      let v = pyFloat(f in datos ? datos[f] : 0);
      if (f === "N_mgkg" || f === "P_mgkg" || f === "K_mgkg") v = v * 2.0;
      x.push(v);
    }
    x.push(cod);
    // StandardScaler (float64) y luego float32, como hace sklearn en los árboles
    return x.map((v, i) => Math.fround((v - MODELO.scaler_media[i]) / MODELO.scaler_escala[i]));
  }

  function predictProba(x) {
    const nClases = MODELO.clases.length;
    const total = new Array(nClases).fill(0);
    for (const arbol of MODELO.arboles) {
      let nodo = 0;
      while (arbol.izq[nodo] !== -1) {
        nodo = x[arbol.var[nodo]] <= arbol.umbral[nodo] ? arbol.izq[nodo] : arbol.der[nodo];
      }
      const hoja = arbol.hojas[nodo];
      for (let i = 0; i < hoja.length; i += 2) total[hoja[i]] += hoja[i + 1];
    }
    return total.map((v) => v / MODELO.arboles.length);
  }

  function predecirCultivos(datos, nMejores) {
    let x;
    try { x = prepararEntrada(datos); } catch (e) { x = null; }
    if (!x) return null;
    const proba = predictProba(x);
    // np.argsort(p)[-n:][::-1]; los empates exactos se ordenan por índice descendente
    const indices = proba.map((_, i) => i).sort((a, b) => proba[b] - proba[a] || b - a).slice(0, nMejores);
    return indices.map((i) => ({
      cultivo: MODELO.clases[i],
      confianza: proba[i],
      porcentaje: proba[i] * 100,
    }));
  }

  // ---------------------------------------------------------------------------
  // models/fertilizer_recommender_v2.py
  // ---------------------------------------------------------------------------

  const UMBRALES_ESTADO = {
    N: { exceso_toxico: 200, exceso: 30, optimo_max: 10, optimo_min: -10, critico: -50 },
    P: { exceso_toxico: 100, exceso: 20, optimo_max: 10, optimo_min: -10, critico: -40 },
    K: { exceso_toxico: 300, exceso: 50, optimo_max: 20, optimo_min: -20, critico: -80 },
    pH: { exceso_toxico: 1.5, exceso: 0.5, optimo_max: 0.3, optimo_min: -0.3, critico: -1.0 },
    MO: { exceso_toxico: 3.0, exceso: 1.0, optimo_max: 0.5, optimo_min: -0.5, critico: -1.5 },
  };

  function clasificarEstado(parametro, balance) {
    const u = UMBRALES_ESTADO[parametro] || UMBRALES_ESTADO.N;
    const neg = -balance;
    if (neg >= u.exceso_toxico) return "Exceso Toxico";
    if (neg >= u.exceso) return "Exceso";
    if (u.optimo_min <= balance && balance <= u.optimo_max) return "Optimo";
    if (balance <= u.critico) return "Critico";
    return "Deficit";
  }

  function formatearDosis(kgHa) {
    return `${pyFixed(kgHa, 1)} kg/ha (Equivalente a ${pyFixed(kgHa / 50.0, 1)} costales de 50 kg)`;
  }

  function normalizarNombreCultivo(c) {
    return String(c).toLowerCase().split(" ").join("_");
  }

  function calcularDeficit(cultivo, prop) {
    let demanda = DEMANDAS.get(normalizarNombreCultivo(cultivo));
    if (demanda === undefined) {
      demanda = DEMANDAS.has("maiz_grano_blanco")
        ? DEMANDAS.get("maiz_grano_blanco")
        : (primeraDemanda() || { n: 180, p: 80, k: 120, ph: 6.5, mo: 3.0 });
    }
    const get = (k, def) => pyFloat(k in prop ? prop[k] : def);
    const vc = [demanda.n, demanda.p, demanda.k, demanda.ph, demanda.mo].map(pyFloat);
    const vs = [get("N_mgkg", 0) * 2.0, get("P_mgkg", 0) * 2.0, get("K_mgkg", 0) * 2.0, get("ph", 7.0), get("mo_pct", 2.0)];
    const delta = vc.map((v, i) => v - vs[i]);
    const aplicable = [Math.max(0.0, delta[0]), Math.max(0.0, delta[1]), Math.max(0.0, delta[2]), 0.0, 0.0];
    const params = ["N", "P", "K", "pH", "MO"];
    const unidades = { N: "kg/ha", P: "kg/ha", K: "kg/ha", pH: "pH", MO: "%" };
    const tabla = params.map((p, i) => ({
      parametro: p,
      demanda_cultivo: pyRound(vc[i], 2),
      aporte_suelo: pyRound(vs[i], 2),
      balance: pyRound(delta[i], 2),
      estado: clasificarEstado(p, delta[i]),
      unidad: unidades[p],
    }));
    const exceso = (i) => Math.max(0.0, vs[i] - vc[i]);
    return {
      v_demanda: vc,
      v_suelo: vs,
      deficit_real: delta,
      deficit_aplicable: aplicable,
      cultivo: cultivo,
      tabla_analisis: tabla,
      exceso: { n: exceso(0), p: exceso(1), k: exceso(2), ph: exceso(3), mo: exceso(4) },
    };
  }

  function seleccionarFertilizantes(deficitNpk, hectareas) {
    const nutrientes = ["n", "p", "k"];
    const aCorregir = nutrientes
      .map((nut, i) => [i, nut, deficitNpk[i]])
      .filter(([, , v]) => v > 10.0);
    if (aCorregir.length === 0) {
      return { mensaje: "Suelo equilibrado, sin necesidad de fertilización adicional.", fertilizantes: [], costo_total: 0.0 };
    }
    const recs = [];
    for (const [idx, nut, deficit] of aCorregir) {
      let mejor = null, mejorScore = Infinity;
      for (const f of RECOM.fertilizantes) {
        const npk = [f.n, f.p, f.k];
        const conc = npk[idx];
        if (conc <= 0) continue;
        const dosis = deficit / conc;
        const costo = dosis * f.precio_kg * hectareas;
        const conflicto = [0, 1, 2].some((j) => j !== idx && npk[j] > 0 && deficitNpk[j] <= 0);
        if (conflicto) continue;
        const score = costo / (conc + 1);
        if (score < mejorScore) { mejorScore = score; mejor = { f, dosis, costo, nut }; }
      }
      if (mejor) recs.push(mejor);
    }
    let costoTotal = 0;
    for (const r of recs) costoTotal += r.costo;
    return {
      fertilizantes: recs.map((r) => ({
        id: r.f.id,
        nombre_comercial: r.f.nombre,
        composicion: { n: r.f.n, p: r.f.p, k: r.f.k },
        nutriente_objetivo: r.nut,
        // dosis y costo son np.float64 en Python -> redondeo de NumPy
        dosis_kg_ha: npRound(r.dosis, 2),
        dosis_formateada: formatearDosis(r.dosis),
        dosis_total_kg: npRound(r.dosis * 1, 2),
        precio_kg: r.f.precio_kg,
        precio_total: npRound(r.costo, 2),
        numero_costales_50kg: npRound(r.dosis / 50, 2),
        costo_costales: npRound((r.dosis / 50) * r.f.precio_costal, 2),
        clasificacion: r.f.clasificacion,
        funcion: r.f.funcion,
      })),
      costo_total: costoTotal,
    };
  }

  function recomendarFertilizantes(cultivo, prop, hectareas) {
    const analisis = calcularDeficit(cultivo, prop);
    return {
      analisis_deficit: analisis,
      recomendaciones: seleccionarFertilizantes(analisis.deficit_aplicable.slice(0, 3), hectareas),
      cultivo,
      hectareas,
    };
  }

  // ---------------------------------------------------------------------------
  // models/soil_analyzer.py
  // ---------------------------------------------------------------------------

  function clasificarNutriente(valor, lim) {
    const bajo = lim.bajo !== undefined ? lim.bajo : valor / 2;
    const medio = lim.medio !== undefined ? lim.medio : valor * 1.5;
    const alto = lim.alto !== undefined ? lim.alto : valor * 3;
    if (valor < bajo) return "deficiente";
    if (valor < medio) return "bajo";
    if (valor < alto) return "medio";
    return "alto";
  }

  function analizarSuelo(datos) {
    const perfil = {};
    for (const nut of ["N_mgkg", "P_mgkg", "K_mgkg", "Ca_mgkg", "Mg_mgkg"]) {
      if (nut in datos) {
        const valor = pyFloat(datos[nut]);
        const lim = NUTRIENT_LIMITS[nut] || {};
        perfil[nut] = { valor, nivel: clasificarNutriente(valor, lim), limites: lim };
      }
    }

    const comparativa = {};
    for (const [nut, lim] of Object.entries(NUTRIENT_LIMITS)) {
      if (!(nut in datos)) continue;
      const valor = pyFloat(datos[nut]);
      let cl;
      if (valor < lim.bajo) cl = "bajo";
      else if (valor < lim.medio) cl = "medio";
      else if (valor < lim.alto) cl = "alto";
      else cl = "muy_alto";
      let rec;
      if (cl === "bajo") rec = `Aumentar a ${lim.medio} mg/kg`;
      else if (cl === "alto" || cl === "muy_alto") rec = `Mantener o reducir a ${lim.medio} mg/kg`;
      else rec = "Nivel adecuado, mantener";
      comparativa[nut] = { valor, clasificacion: cl, recomendacion: rec };
    }

    let vector = null;
    try {
      const vals = SOIL_FEATURES.filter((f) => f in datos).map((f) => pyFloat(datos[f]));
      if (vals.length) {
        const max = vals.reduce((a, b) => (b > a || isNaN(b) ? b : a));
        const min = vals.reduce((a, b) => (b < a || isNaN(b) ? b : a));
        const rango = max - min;
        vector = rango === 0 ? vals.map(() => 0) : vals.map((v) => (v - min) / rango);
      }
    } catch (e) { vector = null; }

    const compatibles = predecirCultivos(datos, 10);

    const requerimientos = {};
    if (compatibles && compatibles.length) {
      for (const info of compatibles.slice(0, 3)) {
        const base = DEMANDAS.get(normalizarNombreCultivo(info.cultivo)) || primeraDemanda() || { n: 150, p: 50, k: 100 };
        const nutrientes = {};
        for (const [nut, clave] of [["N", "n"], ["P", "p"], ["K", "k"]]) {
          const requerido = pyFloat(clave in base ? base[clave] : 0);
          const llave = `${nut}_mgkg`;
          const disponible = pyFloat(llave in datos ? datos[llave] : 0) * 2.0;
          nutrientes[nut] = {
            requerido, disponible,
            deficit: Math.max(0.0, requerido - disponible),
            es_suficiente: disponible >= requerido,
          };
        }
        requerimientos[info.cultivo] = { confianza: info.porcentaje, nutrientes };
      }
    }

    const mapa = { deficiente: 10, bajo: 30, medio: 75, alto: 85, muy_alto: 60 };
    const scores = Object.values(comparativa).map((c) => (c.clasificacion in mapa ? mapa[c.clasificacion] : 50));
    const score = scores.length ? npMean(scores) : 50.0;
    let estado;
    if (score >= 80) estado = "excelente";
    else if (score >= 65) estado = "bueno";
    else if (score >= 50) estado = "aceptable";
    else if (score >= 30) estado = "pobre";
    else estado = "muy_pobre";

    return {
      perfil_nutricional: perfil,
      comparativa_limites: comparativa,
      vector_normalizado: vector,
      cultivos_compatibles: compatibles,
      requerimientos_nutricionales: requerimientos,
      salud_general: { score, estado },
    };
  }

  // ---------------------------------------------------------------------------
  // app.py — helpers y endpoints
  // ---------------------------------------------------------------------------

  const ESTADO_COLOR = {
    excelente: "#27ae60", bueno: "#9acd32", aceptable: "#f39c12", pobre: "#e67e22", muy_pobre: "#e74c3c",
  };
  const estadoAColor = (e) => ESTADO_COLOR[String(e).toLowerCase()] || "#6c757d";

  function normalizarTextura(v) {
    return pyStr(v).toLowerCase().trim().split(" ").join("_").split("-").join("_");
  }

  // str(v) para valores que vienen del JSON o de una celda de CSV
  function pyStr(v) {
    if (typeof v === "string") return v;
    if (typeof v === "number") return pyStrNumero(v, false);
    if (typeof v === "boolean") return v ? "True" : "False";
    if (v === null || v === undefined) return "None";
    return String(v);
  }

  function analizarParcelaDesdeDict(datos, hectareas) {
    if ("textura_suelo" in datos) datos.textura_suelo = normalizarTextura(datos.textura_suelo);

    const [valido, limpios, errores] = sanitizarDiccionarioSuelo(datos);
    if (!valido) throw new PyError("ValueError", `Datos inválidos: [${errores.map(pyRepr).join(", ")}]`);

    for (const k of Object.keys(limpios)) {
      if (typeof limpios[k] === "number" && isNaN(limpios[k])) limpios[k] = 50.0;
    }

    const analisis = analizarSuelo(limpios);
    const top = (analisis.cultivos_compatibles && analisis.cultivos_compatibles.length
      ? analisis.cultivos_compatibles : [null])[0];
    let plan = null;
    if (top) {
      const recom = recomendarFertilizantes(top.cultivo, limpios, hectareas);
      plan = {
        cultivo: top.cultivo,
        analisis_deficit: recom.analisis_deficit,
        fertilizantes: recom.recomendaciones.fertilizantes,
        costo_total_estimado: recom.recomendaciones.costo_total,
        mensaje: recom.recomendaciones.mensaje || "",
        hectareas,
      };
    }
    const entrada = {};
    for (const [k, v] of Object.entries(limpios)) entrada[k] = typeof v === "string" ? v : pyFloat(v);
    return { analisis_suelo: analisis, plan_fertilizacion: plan, datos_entrada: entrada };
  }

  const VALORES_NEUTROS = {
    ph: 7.0, mo_pct: 2.5, N_mgkg: 45.0, P_mgkg: 15.0, K_mgkg: 120.0, Ca_mgkg: 1500.0, Mg_mgkg: 250.0,
    S_mgkg: 25.0, Fe_mgkg: 12.0, Zn_mgkg: 1.5, Mn_mgkg: 8.0, Cu_mgkg: 0.8, B_mgkg: 0.5,
    ce_ds_m: 0.8, precipitacion_mm: 750.0, temp_media_c: 22.0, altitud_msnm: 1500.0, textura_suelo: "franco",
  };

  function completarColumnasFaltantes(df) {
    const faltantes = [];
    for (const col of SOIL_FEATURES.concat(["textura_suelo"])) {
      if (!df.columnas.includes(col)) {
        const v = col in VALORES_NEUTROS ? VALORES_NEUTROS[col] : 0.0;
        df.columnas.push(col);
        df.datos[col] = { tipo: typeof v === "string" ? "str" : "float", valores: new Array(df.n).fill(v) };
        faltantes.push(col);
      }
    }
    const tex = df.datos.textura_suelo;
    tex.valores = tex.valores.map((v) => normalizarTextura(strCelda(tex.tipo, v)));
    tex.tipo = "str";
    return faltantes;
  }

  // str() de Python sobre una celda de pandas
  function strCelda(tipo, v) {
    if (tipo === "int") return String(v);
    if (tipo === "bool") return v ? "True" : "False";
    if (typeof v === "number") return pyStrNumero(v, false);
    return pyStr(v);
  }

  function prefijoParcela(nombre) {
    const s = String(nombre)
      .replace(/[\s_\-]*(m|muestra|s|sample|r)?\s*\p{Nd}+\s*$/iu, "")
      .trim();
    return s ? s : String(nombre);
  }

  // Media de una columna de pandas (ignora NaN)
  function mediaColumna(col, filas) {
    if (col.tipo === "str") throw new PyError("TypeError", "Cannot perform reduction 'mean' with string dtype");
    const vals = filas.map((i) => {
      const v = col.valores[i];
      return col.tipo === "bool" ? (v ? 1 : 0) : v;
    });
    const validos = vals.filter((v) => !isNaN(v));
    if (validos.length === 0) return NaN;
    return sumaPorPares(vals.map((v) => (isNaN(v) ? 0 : v)), 0, vals.length) / validos.length;
  }

  // Moda de pandas: la más frecuente; empates -> la menor en orden
  function modaColumna(col, filas) {
    const cuenta = new Map();
    for (const i of filas) {
      const v = col.valores[i];
      if (typeof v === "number" && isNaN(v)) continue;
      cuenta.set(v, (cuenta.get(v) || 0) + 1);
    }
    const max = Math.max(...cuenta.values());
    return [...cuenta.keys()].filter((k) => cuenta.get(k) === max).sort()[0];
  }

  function valorCelda(col, i) {
    const v = col.valores[i];
    if (col.tipo === "bool") return v ? 1 : 0;
    return v;
  }

  function procesarDfParcelas(df) {
    const faltantes = completarColumnasFaltantes(df);
    const colParcela = ["id_parcela", "parcela", "id"].find((c) => df.columnas.includes(c)) || null;

    const grupos = new Map();
    if (colParcela) {
      const col = df.datos[colParcela];
      for (let i = 0; i < df.n; i++) {
        const g = prefijoParcela(strCelda(col.tipo, col.valores[i]));
        if (!grupos.has(g)) grupos.set(g, []);
        grupos.get(g).push(i);
      }
    } else if (df.n === 1) {
      grupos.set("Parcela_1", [0]);
    } else {
      for (let i = 0; i < df.n; i++) grupos.set(`Parcela_${i + 1}`, [i]);
    }

    const esUnico = grupos.size === 1;
    const resultados = [];
    const presentes = SOIL_FEATURES.filter((c) => df.columnas.includes(c));

    for (const [nombre, filas] of grupos) {
      const individuales = [];
      for (const i of filas) {
        const muestra = {};
        for (const c of presentes) muestra[c] = pyFloat(valorCelda(df.datos[c], i));
        muestra.textura_suelo = df.datos.textura_suelo.valores[i];
        try { individuales.push(analizarParcelaDesdeDict(muestra, 1.0)); } catch (e) { /* igual que Python */ }
      }

      const prom = {};
      for (const c of presentes) prom[c] = pyFloat(mediaColumna(df.datos[c], filas));
      prom.textura_suelo = modaColumna(df.datos.textura_suelo, filas);

      try {
        const analisis = analizarParcelaDesdeDict(prom, 1.0);
        const estado = ((analisis.analisis_suelo || {}).salud_general || {}).estado || "pobre";
        Object.assign(analisis, {
          parcela: nombre, num_muestras: filas.length,
          estado_suelo: estado, color: estadoAColor(estado),
          muestras_individuales: individuales,
        });
        resultados.push(analisis);
      } catch (e) {
        resultados.push({
          parcela: nombre, num_muestras: filas.length,
          error: e.message, estado_suelo: "error",
          color: "#6c757d", muestras_individuales: [],
        });
      }
    }

    if (esUnico) {
      const r = resultados[0];
      if (!r.parcela) r.parcela = [...grupos.keys()][0];
      return { exito: true, modo: "unico", resultado: r, columnas_usadas_por_defecto: faltantes };
    }
    const matriz = resultados.map((r) => ({
      parcela: r.parcela,
      estado: r.estado_suelo !== undefined ? r.estado_suelo : "?",
      color: r.color !== undefined ? r.color : "#6c757d",
      score: ((r.analisis_suelo || {}).salud_general || {}).score ?? 0,
    }));
    return {
      exito: true, modo: "multiple", total_parcelas: resultados.length,
      resultados, matriz_colores: matriz, columnas_usadas_por_defecto: faltantes,
    };
  }

  function csvDesdeTexto(texto, mensajeVacio) {
    const df = leerCsv(texto);
    if (df.n === 0 || df.columnas.length === 0) throw new PyError("ValueError", mensajeVacio);
    return df;
  }

  // POST /api/csv/analizar
  function apiCsvAnalizar(archivo) {
    if (!archivo) return [400, { error: "No se proporcionó archivo." }];
    if (!archivo.nombre.endsWith(".csv")) return [400, { error: "Formato no válido. Use CSV." }];
    return [200, procesarDfParcelas(csvDesdeTexto(archivo.texto, "El archivo CSV está vacío."))];
  }

  // POST /api/csv/demo
  function apiCsvDemo() {
    return [200, procesarDfParcelas(csvDesdeTexto(DEMO_CSV, "El archivo CSV de ejemplo está vacío."))];
  }

  function inferirPropiedadesPorCultivo(cultivos) {
    const cultivo = cultivos.length ? pyStr(cultivos[0]).toLowerCase().split(" ").join("_") : "maiz_grano_blanco";
    let demanda = DEMANDAS.get(cultivo);
    if (demanda === undefined) demanda = primeraDemanda() || { n: 150, p: 50, k: 100, ph: 6.5, mo: 2.5 };
    const g = (k, d) => (k in demanda ? demanda[k] : d);
    return {
      N_mgkg: pyFloat(g("n", 150)) / 2.0,
      P_mgkg: pyFloat(g("p", 50)) / 2.0,
      K_mgkg: pyFloat(g("k", 100)) / 2.0,
      mo_pct: pyFloat(g("mo", 2.5)),
      ph: pyFloat(g("ph", 6.5)),
    };
  }

  // POST /api/manual/inferir
  function apiManualInferir(datos) {
    const get = (k, d) => (datos && k in datos ? datos[k] : d);
    const cultivos = get("cultivos_previos", []);
    const tipo = get("tipo_suelo", null);
    const vacio = (v) => v === null || v === undefined || v === "" || v === 0 || v === false ||
      (Array.isArray(v) && v.length === 0) || (typeof v === "object" && !Array.isArray(v) && Object.keys(v).length === 0);
    if (vacio(cultivos) || vacio(tipo)) return [400, { error: "Información incompleta." }];

    const prop = inferirPropiedadesPorCultivo(cultivos);
    Object.assign(prop, {
      textura_suelo: tipo,
      ph: pyFloat(get("ph", 6.5)),
      precipitacion_mm: pyFloat(get("precipitacion_mm", 700)),
      temp_media_c: pyFloat(get("temp_media_c", 20)),
      altitud_msnm: pyFloat(get("altitud_msnm", 1500)),
      ce_ds_m: pyFloat(get("ce_ds_m", 0.8)),
    });
    for (const f of SOIL_FEATURES) if (!(f in prop)) prop[f] = 50;

    const analisis = analizarParcelaDesdeDict(prop, 1.0);
    return [200, {
      exito: true,
      propiedades_inferidas: prop,
      ...analisis,
      advertencia: "Resultados basados en inferencias. Se recomienda análisis de laboratorio.",
    }];
  }

  // POST /api/fertilizer/optimization-for-crop
  function apiOptimizacion(datos) {
    const get = (k, d) => (k in datos ? datos[k] : d);
    const cultivo = get("cultivo", "maiz");
    const prop = get("propiedades_suelo", {});
    const totalBags = pyFloat(get("total_bags", 0));

    const analisis = calcularDeficit(cultivo, prop);
    const deficitTotal = (0 + analisis.deficit_aplicable[0]) + analisis.deficit_aplicable[1] + analisis.deficit_aplicable[2];
    let excesoTotal = 0;
    for (const v of Object.values(analisis.exceso)) excesoTotal += v;

    const a = Math.max(1.0, 1.5 + deficitTotal / 100.0);
    const b = Math.max(1.0, 5.0 - excesoTotal / 50.0);
    let c, uOpt;
    if (totalBags === 0) { c = 0.1; uOpt = 0.0; }
    else {
      c = a / (2 * totalBags * (totalBags + 1));
      const disc = c ** 2 + 2 * c * a;
      uOpt = disc > 0 ? (-c + Math.sqrt(disc)) / (2 * c) : 0.0;
    }

    const u = linspace(0, Math.max(40, uOpt * 1.5), 200);
    const R = u.map((x) => a * Math.log(x + 1) + b);
    const D = u.map((x) => c * (x * x));
    const B = R.map((r, i) => r - D[i]);
    let idx = 0;
    for (let i = 1; i < u.length; i++) if (Math.abs(u[i] - uOpt) < Math.abs(u[idx] - uOpt)) idx = i;

    return [200, {
      exito: true, cultivo,
      u_values: u, rendimiento: R, degradacion: D, beneficio_neto: B,
      punto_optimo: {
        dosis: pyRound(uOpt, 2),
        rendimiento: pyRound(R[idx], 3),
        beneficio_neto: pyRound(B[idx], 2),
        costo_acumulado: pyRound(uOpt * 500, 2),
      },
      // b es np.float64 en Python (sale de sumar los excesos de NumPy), salvo cuando vale 1.0
      parametros: { a: pyRound(a, 3), b: b === 1.0 ? 1.0 : npRound(b, 3), c: pyRound(c, 4) },
    }];
  }

  // Despachador: equivalente a las rutas de Flask
  function manejar(ruta, peticion) {
    try {
      switch (ruta) {
        case "/api/csv/analizar": return apiCsvAnalizar(peticion.archivo);
        case "/api/csv/demo": return apiCsvDemo();
        case "/api/manual/inferir": return apiManualInferir(peticion.json);
        case "/api/fertilizer/optimization-for-crop": return apiOptimizacion(peticion.json);
        default: return [404, { error: "Ruta no encontrada." }];
      }
    } catch (e) {
      if (!(e instanceof PyError)) console.error(e);
      // /api/manual/inferir y la optimización devuelven 500 ante cualquier error
      const es400 = e instanceof PyError && ["ValueError", "ParserError", "EmptyDataError"].includes(e.tipo) &&
        (ruta === "/api/csv/analizar" || ruta === "/api/csv/demo");
      return [es400 ? 400 : 500, { error: e.message }];
    }
  }

  function responder(ruta, peticion) {
    const [estado, cuerpo] = manejar(ruta, peticion);
    return { estado, cuerpo: ordenarLlaves(cuerpo) };
  }

  const motor = { inicializar, responder, leerCsv, pyFixed, pyRound, npRound };

  // ---------------------------------------------------------------------------
  // Navegador: carga los datos e intercepta fetch("/api/...")
  // ---------------------------------------------------------------------------
  if (typeof window !== "undefined" && typeof window.fetch === "function") {
    const fetchOriginal = window.fetch.bind(window);
    const base = document.currentScript
      ? new URL("../data/", document.currentScript.src)
      : new URL("static/data/", location.href);
    const cargarJson = (n) => fetchOriginal(new URL(n, base)).then((r) => r.json());
    const listo = Promise.all([
      cargarJson("modelo.json"),
      cargarJson("recomendador.json"),
      fetchOriginal(new URL("demo.csv", base)).then((r) => r.text()),
    ]).then(([modelo, recomendador, demoCsv]) => inicializar({ modelo, recomendador, demoCsv }));

    window.fetch = async function (recurso, opciones) {
      const url = typeof recurso === "string" ? recurso : recurso.url;
      if (!url.startsWith("/api/")) return fetchOriginal(recurso, opciones);
      await listo;
      const peticion = {};
      const cuerpo = opciones && opciones.body;
      if (cuerpo instanceof FormData) {
        const f = cuerpo.get("archivo");
        if (f && typeof f === "object") {
          const bytes = await f.arrayBuffer();
          peticion.archivo = { nombre: f.name, texto: new TextDecoder("utf-8").decode(bytes) };
        }
      } else if (typeof cuerpo === "string") {
        peticion.json = JSON.parse(cuerpo);
      }
      const { estado, cuerpo: respuesta } = responder(url.split("?")[0], peticion);
      return new Response(JSON.stringify(respuesta), {
        status: estado, headers: { "Content-Type": "application/json" },
      });
    };
    motor.listo = listo;
    window.AgroasisMotor = motor;
  }

  if (typeof module !== "undefined" && module.exports) module.exports = motor;
})(this);
