"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Chile, 2026.

Script 02 - Modelo corregido, matriz de ablación, línea base Cox y residuos.

Entrena el modelo de supervivencia sin las variables de rezago y con separación
por equipo, que es la configuración válida (sin fuga). Construye la matriz de
ablación de dos por dos -con y sin rezago, cruzado con partición aleatoria y
partición por equipo- que muestra cuánto del desempeño aparente provenía de la
fuga. Incluye una línea base con regresión de Cox y el análisis de residuos del
modelo corregido.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/ablacion_comparacion_metricas.png  (matriz de ablación)
           outputs/calibracion_residuos.png           (histograma de residuos)
           outputs/shap_summary.png                   (importancia de variables, SHAP)
           outputs/02_modelo_corregido.json           (todas las métricas)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\02_modelo_corregido.py
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
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana: las figuras se guardan directo en disco.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

import config as C
import lib as L

# Parámetros de la corrida.
BALANCEADO = True          # 50/50 evento/censura (False para proporción natural).
SAMPLE_N   = 1_000_000     # Submuestra para el modelamiento.

# --- Carga y submuestreo ----------------------------------------------------
# Se cargan los datos con rezagos porque la matriz de ablación necesita el caso
# "con rezago"; el modelo final, en cambio, no los utiliza.
d = L.load_clean(con_lags=True)
print(f"{len(d):,} filas | {d[C.ID_COL].nunique()} equipos | eventos {d['evt'].mean():.1%}")
res = {"n_filas": int(len(d))}
d = L.submuestrear(d, SAMPLE_N, balanceado=BALANCEADO)
res.update(n_filas_modelado=int(len(d)), balanceado=BALANCEADO)
print(f"submuestreo: {len(d):,} filas | {d[C.ID_COL].nunique()} equipos")

# --- Particiones ------------------------------------------------------------
# Se preparan dos tipos de partición para la matriz de ablación:
#   - aleatoria: filas al azar (puede dejar el mismo equipo en train y test).
#   - por equipo: ningún equipo aparece a la vez en entrenamiento y prueba.
idx = np.arange(len(d))
tr_r, te_r = train_test_split(idx, test_size=0.2, random_state=C.RS, stratify=d["evt"])
tr_g, te_g = L.gsplit(d, idx, 0.2, C.RS)


def cind(tr, te, feats):
    """Entrena XGBoost AFT con el conjunto de variables indicado y devuelve el C-index."""
    m = L.xgb_train(d, tr, feats=feats)
    return L.cidx(d, te, -L.xgb_predict(m, d, te, feats))


# --- Matriz de ablación 2x2 -------------------------------------------------
# Cruza dos factores: usar o no los rezagos, y particionar al azar o por equipo.
# El contraste clave es la caída del C-index al quitar los rezagos.
res["matriz_2x2_cindex"] = {
    "lag_random":   round(cind(tr_r, te_r, C.FEATS_LAG), 4),
    "lag_group":    round(cind(tr_g, te_g, C.FEATS_LAG), 4),
    "nolag_random": round(cind(tr_r, te_r, C.FEATS), 4),
    "nolag_group":  round(cind(tr_g, te_g, C.FEATS), 4),
}
res["n_equipos_test_group"] = int(d.loc[te_g, C.ID_COL].nunique())
print("2x2:", res["matriz_2x2_cindex"])

# --- Modelo corregido (sin rezago, partición por equipo) y residuos ---------
# Esta es la configuración válida: refleja el desempeño real del modelo.
m = L.xgb_train(d, tr_g, feats=C.FEATS)
pred = L.xgb_predict(m, d, te_g, C.FEATS)
yte = (d.loc[te_g, C.DURATION_COL] / C.SEC).values
ete = d.loc[te_g, "evt"].values
c_h = L.cidx(d, te_g, -pred)
# El residuo se evalúa solo sobre observaciones con evento (no censuradas).
resid = pred[ete] - yte[ete]
res["xgb_corregido"] = {"c_index": round(c_h, 4), "mae_h": round(float(np.mean(np.abs(resid))), 2),
                        "resid_media_h": round(float(np.mean(resid)), 2),
                        "resid_mediana_h": round(float(np.median(resid)), 2)}
print("corregido:", res["xgb_corregido"])

# --- Línea base: regresión de Cox -------------------------------------------
# Modelo de referencia clásico, sobre una submuestra y con variables
# estandarizadas. Sirve para contrastar el C-index del modelo basado en árboles.
try:
    trc = np.random.RandomState(C.RS).choice(tr_g, min(50000, len(tr_g)), replace=False)
    sc = StandardScaler().fit(d.loc[trc, C.FEATS])

    def mk(ix):
        X = pd.DataFrame(sc.transform(d.loc[ix, C.FEATS]), columns=C.FEATS)
        X["dur_h"] = (d.loc[ix, C.DURATION_COL] / C.SEC).values
        X["evt"]   = d.loc[ix, "evt"].astype(int).values
        return X

    cph = CoxPHFitter(penalizer=0.1).fit(mk(trc), "dur_h", "evt")
    res["cox_corregido"] = {"c_index": round(float(cph.score(mk(te_g), scoring_method="concordance_index")), 4)}
    print("cox:", res["cox_corregido"])
except Exception as e:
    res["cox_corregido"] = {"error": str(e)[:160]}

# --- Figura: matriz de ablación ---------------------------------------------
# Cuatro barras. Las dos primeras (con rezago) quedan cerca de 1; las dos
# últimas (sin rezago) caen al desempeño real. El hachurado y la línea de azar
# refuerzan la lectura sin depender solo del color (daltonismo).
labels = ["CON lag\n(aleatorio)", "CON lag\n(por equipo)", "SIN lag\n(aleatorio)", "SIN lag\n(por equipo)"]
vals = [res["matriz_2x2_cindex"][k] for k in ["lag_random", "lag_group", "nolag_random", "nolag_group"]]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, vals, color=[C.OI["gris"], C.OI["gris"], C.OI["celeste"], C.OI["azul"]], edgecolor=C.OI["negro"])
for b, h in zip(bars, ["xx", "xx", "", ""]):
    b.set_hatch(h)
ax.axhline(0.5, color=C.OI["negro"], ls=":", label="Azar (0,5)")
ax.set_ylim(0, 1.05)
ax.set_ylabel("C-index (Harrell)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}".replace(".", ","), ha="center", fontsize=9)
ax.legend(frameon=False)
# Sin título embebido: la descripción va en el pie de figura del documento.
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "ablacion_comparacion_metricas.png", dpi=150)
plt.close(fig)

# --- Figura: residuos del modelo corregido ----------------------------------
# El histograma muestra que los residuos se concentran a la derecha de cero: el
# modelo tiende a sobreestimar el tiempo hasta la falla.
fig, ax = plt.subplots(figsize=(6, 4))
r = resid[(resid > np.percentile(resid, 1)) & (resid < np.percentile(resid, 99))]
ax.hist(r, bins=60, color=C.OI["celeste"], edgecolor=C.OI["negro"])
ax.axvline(0, color=C.OI["negro"], ls="--", label="Cero (sin sesgo)")
ax.axvline(float(np.mean(resid)), color=C.OI["naranja"],
           label=f"Media = {np.mean(resid):.1f} h".replace(".", ","))
ax.set_xlabel("Residuo = predicho - real (horas)")
ax.set_ylabel("Frecuencia")
ax.legend(frameon=False)
# Sin título embebido: la descripción va en el pie de figura del documento.
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "calibracion_residuos.png", dpi=150)
plt.close(fig)

# --- Figura: importancia de variables del modelo corregido (SHAP) -----------
# SHAP sobre el modelo SIN rezagos, para mostrar los determinantes legitimos del
# tiempo hasta la falla. Se grafica la importancia media |SHAP| en barras (no el
# beeswarm rojo-azul de SHAP), respetando la accesibilidad daltonica (Okabe-Ito).
try:
    import shap, xgboost as xgb
    sh_idx = np.random.RandomState(C.RS).choice(te_g, min(3000, len(te_g)), replace=False)
    X_sh = d.loc[sh_idx, C.FEATS]
    sv = shap.TreeExplainer(m).shap_values(xgb.DMatrix(X_sh))
    imp = np.abs(sv).mean(axis=0)
    orden = np.argsort(imp)
    res["shap_importancia"] = {C.FEATS[i]: round(float(imp[i]), 4) for i in orden[::-1]}
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([C.FEATS[i] for i in orden], imp[orden], color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax.set_xlabel("Importancia media |SHAP| (impacto sobre log-TTF)")
    # Sin titulo embebido: la descripcion va en el pie de figura del documento.
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "shap_summary.png", dpi=150)
    plt.close(fig)
    print("shap:", res["shap_importancia"])
except Exception as e:
    res["shap_importancia"] = {"error": str(e)[:160]}
    print("shap error:", str(e)[:160])

(C.OUTPUTS_DIR / "02_modelo_corregido.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("Listo. Resultados en outputs/ablacion_comparacion_metricas.png, outputs/calibracion_residuos.png, outputs/shap_summary.png y outputs/02_modelo_corregido.json")
