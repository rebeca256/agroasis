// ============================================================================
// Prueba de paridad Flask vs. motor JavaScript (parte 2 de 2).
// Ejecuta en el motor JS las mismas peticiones de pruebas/salida/casos.json
// y compara cada respuesta con la que dio Flask.
//
// Tolerancia numérica: 1e-9 relativo. sklearn suma las probabilidades de
// los 200 árboles en hilos paralelos, así que el propio Python varía en el
// último bit entre ejecuciones; y NumPy y JS pueden diferir 1 ulp en log().
// Los cultivos con probabilidad idéntica (empates exactos) se comparan
// como conjunto: NumPy los ordena de forma arbitraria según el CPU.
// ============================================================================
const fs = require("fs");
const path = require("path");

const RAIZ = path.resolve(__dirname, "..");
const DATOS = path.join(RAIZ, "web", "static", "data");
const motor = require(path.join(RAIZ, "web", "static", "js", "agroasis-engine.js"));

motor.inicializar({
  modelo: JSON.parse(fs.readFileSync(path.join(DATOS, "modelo.json"), "utf8")),
  recomendador: JSON.parse(fs.readFileSync(path.join(DATOS, "recomendador.json"), "utf8")),
  demoCsv: fs.readFileSync(path.join(DATOS, "demo.csv"), "utf8"),
});

const casos = JSON.parse(fs.readFileSync(path.join(__dirname, "salida", "casos.json"), "utf8"));
const TOL = 1e-9;

let empatesTop = 0;
const empata = (x, y) => Math.abs(x - y) <= TOL * 100;

// Empates exactos de probabilidad: NumPy decide su orden según el CPU, así
// que cuando caen en un corte (top 10, top 3 de requerimientos, o el 1er
// lugar que define el plan) la elección es arbitraria también en Python.
// Esos cultivos se marcan "<empate>" y se omiten de la comparación.
function canonizarResultado(r) {
  const a = r.analisis_suelo;
  const lista = a.cultivos_compatibles;
  if (!Array.isArray(lista) || !lista.length) return;
  const ambiguos = new Set();
  const marcarGrupo = (p) => lista.forEach((x) => { if (empata(x.porcentaje, p)) ambiguos.add(x.cultivo); });
  const ultimo = lista[lista.length - 1].porcentaje;
  lista.forEach((x) => { if (empata(x.porcentaje, ultimo)) { ambiguos.add(x.cultivo); x.cultivo = "<empate>"; } });
  if (lista.length > 3 && empata(lista[2].porcentaje, lista[3].porcentaje)) marcarGrupo(lista[2].porcentaje);
  const req = a.requerimientos_nutricionales || {};
  for (const k of Object.keys(req)) if (ambiguos.has(k)) delete req[k];
  if (lista.length > 1 && empata(lista[0].porcentaje, lista[1].porcentaje)) {
    empatesTop++;
    r.plan_fertilizacion = "<empate en primer lugar>";
  }
}

// Ordena los empates exactos de cultivos_compatibles de forma canónica
function canonizar(v) {
  if (v && typeof v === "object" && v.analisis_suelo && v.analisis_suelo.cultivos_compatibles) canonizarResultado(v);
  if (Array.isArray(v)) {
    const lista = v.map(canonizar);
    if (lista.length && lista.every((x) => x && typeof x === "object" && "cultivo" in x && "porcentaje" in x)) {
      lista.sort((a, b) => (Math.abs(a.porcentaje - b.porcentaje) > TOL * 100
        ? b.porcentaje - a.porcentaje
        : a.cultivo < b.cultivo ? -1 : a.cultivo > b.cultivo ? 1 : 0));
    }
    return lista;
  }
  if (v && typeof v === "object") {
    const o = {};
    for (const k of Object.keys(v)) o[k] = canonizar(v[k]);
    return o;
  }
  return v;
}

function diferencias(a, b, ruta, salida) {
  if (salida.length > 5) return;
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isNaN(a) && Number.isNaN(b)) return;
    if (a === b || Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b))) return;
    salida.push(`${ruta}: Flask=${a} JS=${b}`);
    return;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) { salida.push(`${ruta}: largo Flask=${a.length} JS=${b.length}`); return; }
    a.forEach((x, i) => diferencias(x, b[i], `${ruta}[${i}]`, salida));
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.join() !== kb.join()) { salida.push(`${ruta}: llaves Flask=[${ka}] JS=[${kb}]`); return; }
    ka.forEach((k) => diferencias(a[k], b[k], `${ruta}.${k}`, salida));
    return;
  }
  if (a !== b) salida.push(`${ruta}: Flask=${JSON.stringify(a)} JS=${JSON.stringify(b)}`);
}

let fallos = 0;
const porRuta = {};
for (const [i, c] of casos.entries()) {
  const peticion = c.tipo === "csv"
    ? { archivo: { nombre: c.peticion.nombre, texto: c.peticion.texto } }
    : c.peticion;
  const { estado, cuerpo } = motor.responder(c.ruta, peticion);
  // Ida y vuelta por JSON, como la respuesta real del navegador
  const js = JSON.parse(JSON.stringify(cuerpo));
  const difs = [];
  if (estado !== c.estado) difs.push(`estado HTTP: Flask=${c.estado} JS=${estado}`);
  diferencias(canonizar(c.respuesta), canonizar(js), "$", difs);
  porRuta[c.ruta] = porRuta[c.ruta] || { total: 0, ok: 0 };
  porRuta[c.ruta].total++;
  if (difs.length) {
    fallos++;
    if (fallos <= 15) {
      console.log(`\n✗ Caso ${i} (${c.ruta})`);
      difs.forEach((d) => console.log("   " + d));
    }
  } else porRuta[c.ruta].ok++;
}

console.log("\nResumen por endpoint:");
for (const [r, s] of Object.entries(porRuta)) console.log(`  ${r.padEnd(42)} ${s.ok}/${s.total}`);
// canonizar() se aplica a la respuesta de Flask y a la de JS: cada empate cuenta 2 veces
if (empatesTop) console.log(`\n(${empatesTop / 2} resultados con empate exacto en el 1er cultivo: plan no comparado)`);
console.log(fallos ? `\n${fallos} casos con diferencias` : `\nOK: ${casos.length} casos idénticos`);
process.exit(fallos ? 1 : 0);
