"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 12 - Inferencia y priorización de equipos por riesgo.

Toma el modelo corregido (XGBoost Survival AFT, sin variables de rezago y con
separación por equipo), predice el tiempo hasta la próxima falla (TTF) para el
estado más reciente de cada equipo, deriva las probabilidades de sobrevivencia
en varios horizontes y asigna un nivel de riesgo para apoyar la priorización del
mantenimiento. No reemplaza el criterio humano: ordena, no cronometra.

Nota sobre las variables: usa exclusivamente C.FEATS (las diez del modelo
corregido). Las variables de rezago NO se utilizan, porque constituían una fuga
de información (ver script 01 y matriz de ablación del script 02).

Nota sobre los umbrales: los cortes de la probabilidad de sobrevivencia a 24 h
(0,30 / 0,50 / 0,70) son umbrales operativos elegidos para escalonar la
priorización; no provienen de una optimización por costo y pueden ajustarse a la
política de mantenimiento.

Entradas:  conjunto de datos definido en config.py (toy por defecto);
           opcionalmente el mantenedor de equipos (config.py) para enriquecer.
Salidas:   outputs/modelo_corregido_aft.json          (modelo entrenado, reutilizable)
           outputs/predicciones_inferencia.csv         (reporte ordenado por riesgo)
           outputs/predicciones_inferencia.xlsx        (idem, en Excel)
           outputs/inferencia_distribucion_riesgo.png  (conteo por nivel)
           outputs/inferencia_curvas_supervivencia.png (S(t) media por nivel)
           outputs/12_inferencia.json                  (resumen)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\12_inferencia.py
"""
import sys
from pathlib import Path

# Permite importar config.py y lib.py, que viven en esta misma carpeta (src/).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana: las figuras se guardan directo en disco.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import pyarrow.parquet as pq

import config as C
import lib as L

# --- Parámetros de la corrida -----------------------------------------------
SAMPLE_N   = 1_000_000     # Submuestra para entrenar el modelo (igual que el script 02).
BALANCEADO = True          # 50/50 evento/censura al entrenar.
DEVICE     = "cpu"         # Cambiar a "cuda" para entrenar en GPU (P16).
HORIZONTES = [2, 6, 12, 24, 48, 72, 168]            # Horas para P(sobrevivir).
SIGMA      = float(C.XGB_AFT_PARAMS["aft_loss_distribution_scale"])  # Escala AFT (1,0).
CORTES     = (0.30, 0.50, 0.70)                      # Umbrales operativos sobre P(24 h).

# Selección de equipos a predecir:
#   None            -> muestra aleatoria de N_INFER equipos del dataset
#   [123, 456, ...] -> lista explícita de Numpos
#   "ruta/eq.csv"   -> CSV con una columna 'Numpos'
EQUIPOS = None
N_INFER = 320

# Niveles de riesgo: color Okabe-Ito + relleno (hatch) + marcador, para no
# depender solo del color (daltonismo: protanopia/deuteranopia, sin rojo-verde).
NIVELES = ["CRÍTICO", "ALTO", "MEDIO", "BAJO"]
ESTILO_NIVEL = {
    "CRÍTICO": dict(color=C.OI["naranja"],  hatch="xxx", marker="X"),
    "ALTO":    dict(color=C.OI["amarillo"], hatch="//",  marker="^"),
    "MEDIO":   dict(color=C.OI["celeste"],  hatch="..",  marker="s"),
    "BAJO":    dict(color=C.OI["azul"],     hatch="",    marker="o"),
}
ORDEN_NIVEL = {n: i for i, n in enumerate(NIVELES)}


def coma(x, d=1):
    """Formatea un número con coma decimal (convención del informe)."""
    return f"{x:.{d}f}".replace(".", ",")


def survival_aft(t, ttf, sigma=SIGMA):
    """P(T > t) bajo AFT con error normal en escala log (T es log-normal).
    ttf es la mediana predicha, en horas; t en horas."""
    mu = np.log(np.asarray(ttf, dtype=float) + 1e-6)
    z = (np.log(t + 1e-6) - mu) / sigma
    return np.clip(1.0 - stats.norm.cdf(z), 0.0, 1.0)


def clasificar(p24):
    """Asigna nivel de riesgo según la probabilidad de sobrevivir 24 h."""
    a, b, c = CORTES
    if p24 < a:
        return "CRÍTICO"
    if p24 < b:
        return "ALTO"
    if p24 < c:
        return "MEDIO"
    return "BAJO"


# ============================================================================
# PASO 1: ENTRENAR EL MODELO CORREGIDO (sin rezago, separación por equipo)
# ============================================================================
# Misma configuración válida del script 02. El modelo se guarda como artefacto
# para poder reutilizarlo en inferencias posteriores sin reentrenar.
print("=" * 70)
print("   PASO 1: ENTRENANDO MODELO CORREGIDO (sin variables de rezago)")
print("=" * 70)

d = L.load_clean(con_lags=False)
print(f"  {len(d):,} filas | {d[C.ID_COL].nunique()} equipos | eventos {coma(d['evt'].mean()*100)}%")
d = L.submuestrear(d, SAMPLE_N, balanceado=BALANCEADO)
idx = np.arange(len(d))
tr_g, te_g = L.gsplit(d, idx, 0.2, C.RS)
modelo = L.xgb_train(d, tr_g, feats=C.FEATS, device=DEVICE)

# Validación rápida sobre el test por equipo (debe rondar 0,71).
c_val = L.cidx(d, te_g, -L.xgb_predict(modelo, d, te_g, C.FEATS))
print(f"  C-index (test por equipo): {coma(c_val, 3)}")

modelo.save_model(str(C.OUTPUTS_DIR / "modelo_corregido_aft.json"))
print("  Modelo guardado en outputs/modelo_corregido_aft.json")

# ============================================================================
# PASO 2: ESTADO MÁS RECIENTE DE CADA EQUIPO A PREDECIR
# ============================================================================
print("\n" + "=" * 70)
print("   PASO 2: SELECCIÓN DE EQUIPOS Y ESTADO MÁS RECIENTE")
print("=" * 70)

disponibles = set(pq.ParquetFile(C.DATA_PATH).schema.names)
tiene_fecha = "fecha" in disponibles
cols_inf = [c for c in C.FEATS + [C.ID_COL] + (["fecha"] if tiene_fecha else []) if c in disponibles]
df_full = pd.read_parquet(C.DATA_PATH, columns=cols_inf)

# Resolver la fuente de equipos.
if EQUIPOS is None:
    rng = np.random.RandomState(C.RS)
    universo = df_full[C.ID_COL].unique()
    equipos = rng.choice(universo, size=min(N_INFER, len(universo)), replace=False).tolist()
    fuente = f"muestra aleatoria ({len(equipos)})"
elif isinstance(EQUIPOS, (list, tuple)):
    equipos = list(EQUIPOS)
    fuente = f"lista manual ({len(equipos)})"
else:
    equipos = pd.read_csv(EQUIPOS)[C.ID_COL].dropna().unique().tolist()
    fuente = f"CSV {Path(EQUIPOS).name} ({len(equipos)})"
print(f"  Fuente de equipos: {fuente}")

sel = df_full[df_full[C.ID_COL].isin(equipos)].copy()
# Estado más reciente por equipo: por fecha si existe; si no, último registro.
if tiene_fecha:
    sel = sel.sort_values("fecha").groupby(C.ID_COL, as_index=False).last()
else:
    sel = sel.groupby(C.ID_COL, as_index=False).last()

# Limpieza de features (mismo criterio que load_clean).
for col in C.FEATS:
    if col in sel.columns and sel[col].dtype.kind == "f":
        sel[col] = sel[col].replace([np.inf, -np.inf], np.nan)
        sel[col] = sel[col].fillna(sel[col].median())
print(f"  Equipos con datos: {len(sel)}")

# ============================================================================
# PASO 3: PREDICCIÓN DE TTF Y PROBABILIDADES DE SOBREVIVENCIA
# ============================================================================
print("\n" + "=" * 70)
print("   PASO 3: PREDICCIÓN Y NIVEL DE RIESGO")
print("=" * 70)

# El modelo se entrenó con la duración en HORAS, así que predict devuelve horas.
ttf_h = modelo.predict(xgb.DMatrix(sel[C.FEATS].values))
sel["TTF_Predicho_Horas"] = ttf_h
sel["TTF_Predicho_Dias"] = ttf_h / 24.0

for h in HORIZONTES:
    sel[f"P_Sobrevivir_{h}h"] = survival_aft(h, ttf_h)

sel["Nivel_Riesgo"] = sel["P_Sobrevivir_24h"].apply(clasificar)
print("  Distribución de riesgo:")
for n in NIVELES:
    k = int((sel["Nivel_Riesgo"] == n).sum())
    pct = 100 * k / len(sel) if len(sel) else 0
    print(f"    {n:<8}: {k:>5}  ({coma(pct)}%)")

# ============================================================================
# PASO 4: ENRIQUECIMIENTO OPCIONAL CON EL MANTENEDOR
# ============================================================================
try:
    man = pd.read_excel(C.MANTENEDOR_PATH, header=1)
    cols_man = [c for c in ["Numpos", "Comuna", "Operador", "Tipo de Equipo",
                            "Marca Modem", "Coordenadas x", "Coordenadas y", "Estado"]
                if c in man.columns]
    sel = sel.merge(man[cols_man], on=C.ID_COL, how="left")
    print("\n  Reporte enriquecido con el mantenedor de equipos.")
except Exception as e:
    print(f"\n  (Sin enriquecimiento del mantenedor: {type(e).__name__})")

# ============================================================================
# PASO 5: REPORTE ORDENADO POR RIESGO (CSV + XLSX)
# ============================================================================
print("\n" + "=" * 70)
print("   PASO 5: GENERANDO REPORTE")
print("=" * 70)

cols_base = ["Numpos", "TTF_Predicho_Horas", "TTF_Predicho_Dias", "Nivel_Riesgo",
             "P_Sobrevivir_2h", "P_Sobrevivir_6h", "P_Sobrevivir_24h", "P_Sobrevivir_72h"]
cols_extra = [c for c in ["Comuna", "Operador", "Tipo de Equipo",
                          "Coordenadas x", "Coordenadas y"] if c in sel.columns]
rep = sel[[c for c in cols_base + cols_extra if c in sel.columns]].copy()

rep["_o"] = rep["Nivel_Riesgo"].map(ORDEN_NIVEL)
rep = rep.sort_values(["_o", "TTF_Predicho_Horas"]).drop(columns="_o")
for c in [c for c in rep.columns if c.startswith("P_")]:
    rep[c] = (rep[c] * 100).round(1)          # Probabilidades en porcentaje.
rep["TTF_Predicho_Horas"] = rep["TTF_Predicho_Horas"].round(2)
rep["TTF_Predicho_Dias"] = rep["TTF_Predicho_Dias"].round(2)

rep.to_csv(C.OUTPUTS_DIR / "predicciones_inferencia.csv", index=False)
rep.to_excel(C.OUTPUTS_DIR / "predicciones_inferencia.xlsx", index=False)
print("  Guardado: outputs/predicciones_inferencia.csv y .xlsx")

# ============================================================================
# PASO 6: FIGURAS (Okabe-Ito + relleno/marcador, sin rojo-verde)
# ============================================================================
# Figura 1: conteo de equipos por nivel de riesgo.
conteo = [int((sel["Nivel_Riesgo"] == n).sum()) for n in NIVELES]
fig, ax = plt.subplots(figsize=(8, 5))
for i, (n, k) in enumerate(zip(NIVELES, conteo)):
    s = ESTILO_NIVEL[n]
    ax.bar(i, k, color=s["color"], hatch=s["hatch"], edgecolor="black", linewidth=1.2)
    if k > 0:
        pct = 100 * k / len(sel)
        ax.text(i, k, f"{k}\n({coma(pct)}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(NIVELES)))
ax.set_xticklabels(NIVELES)
ax.set_ylabel("Equipos")
ax.set_title("Distribución de equipos por nivel de riesgo")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "inferencia_distribucion_riesgo.png", dpi=150)
plt.close(fig)
print("  Guardado: outputs/inferencia_distribucion_riesgo.png")

# Figura 2: curva de sobrevivencia media por nivel (color + marcador distinto).
t_range = np.linspace(0.1, 168, 200)
fig, ax = plt.subplots(figsize=(9, 6))
for n in NIVELES:
    grp = sel[sel["Nivel_Riesgo"] == n]
    if len(grp) == 0:
        continue
    ttf_med = float(grp["TTF_Predicho_Horas"].mean())
    s = ESTILO_NIVEL[n]
    ax.plot(t_range, survival_aft(t_range, ttf_med), color=s["color"], linewidth=2.3,
            marker=s["marker"], markevery=25, markersize=7,
            label=f"{n} (n={len(grp)}, TTF≈{coma(ttf_med, 0)} h)")
ax.axvline(x=24, color=C.OI["negro"], linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(25, 0.96, "24 h", fontsize=9)
ax.set_xlabel("Tiempo (horas)")
ax.set_ylabel("Probabilidad de sobrevivencia S(t)")
ax.set_title("Curvas de sobrevivencia media por nivel de riesgo")
ax.set_xlim(0, 168)
ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", framealpha=0.95)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "inferencia_curvas_supervivencia.png", dpi=150)
plt.close(fig)
print("  Guardado: outputs/inferencia_curvas_supervivencia.png")

# ============================================================================
# PASO 7: RESUMEN (JSON)
# ============================================================================
res = {
    "c_index_test_equipo": round(c_val, 4),
    "n_equipos_inferidos": int(len(sel)),
    "fuente_equipos": fuente,
    "cortes_p24h": list(CORTES),
    "distribucion_riesgo": {n: int((sel["Nivel_Riesgo"] == n).sum()) for n in NIVELES},
    "ttf_horas": {
        "min": round(float(sel["TTF_Predicho_Horas"].min()), 2),
        "mediana": round(float(sel["TTF_Predicho_Horas"].median()), 2),
        "max": round(float(sel["TTF_Predicho_Horas"].max()), 2),
    },
}
(C.OUTPUTS_DIR / "12_inferencia.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("\n  Resumen en outputs/12_inferencia.json")
print("=" * 70)
print("   INFERENCIA COMPLETADA")
print("=" * 70)
