"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Chile, 2026.

Script 03 - Comparación pareja entre XGBoost AFT y Random Survival Forest.

Compara ambos modelos en condiciones equivalentes: la misma submuestra, igual
presupuesto de optimización de hiperparámetros con Optuna, y separación por
equipo entre entrenamiento y prueba. Reporta capacidad discriminativa (C-index
de Harrell) y calibración (Integrated Brier Score). El propósito es mostrar que
ambos modelos empatan en desempeño, de modo que la elección de XGBoost se
justifica por su eficiencia y escalabilidad, no por una supuesta superioridad.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/comparacion_xgb_vs_rsf.png   (figura de dos paneles)
           outputs/03_comparacion_modelos.json  (métricas e hiperparámetros)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\03_comparacion_xgb_rsf.py
"""
import sys
from pathlib import Path

# Permite importar config.py y lib.py, que viven en esta misma carpeta (src/).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana: la figura se guarda directo en disco.
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import config as C
import lib as L

# Parámetros de la comparación. La submuestra es pequeña porque el Random
# Survival Forest es costoso en memoria y tiempo.
N_COMPARACION = 15000
N_TRIALS_XGB, N_TRIALS_RSF = 40, 20
RSF_TREES_TUNING, RSF_TREES_FINAL = 60, 200

# --- Carga, submuestreo y particiones ---------------------------------------
d = L.load_clean()
d = L.submuestrear(d, N_COMPARACION, balanceado=True)
print(f"{len(d):,} filas | {d[C.ID_COL].nunique()} equipos")
idx = np.arange(len(d))
trv, te = L.gsplit(d, idx, 0.2, C.RS)        # Prueba: equipos no vistos.
tr, val = L.gsplit(d, trv, 0.25, C.RS + 1)   # Validación para Optuna (equipos no vistos).
res = {"n_filas": int(len(d)), "n_equipos_test": int(d.loc[te, C.ID_COL].nunique())}


# --- Optimización de XGBoost AFT con Optuna ---------------------------------
def obj_xgb(t):
    params = dict(max_depth=t.suggest_int("max_depth", 3, 8),
                  learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                  subsample=t.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                  min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                  aft_loss_distribution_scale=t.suggest_float("scale", 0.5, 2.0))
    m = L.xgb_train(d, tr, params=params, nrounds=t.suggest_int("nrounds", 50, 300))
    return L.cidx(d, val, -L.xgb_predict(m, d, val))


s1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.RS))
s1.optimize(obj_xgb, n_trials=N_TRIALS_XGB)
bx = s1.best_params
print("XGB best:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bx.items()})


# --- Optimización de Random Survival Forest con el mismo presupuesto --------
def obj_rsf(t):
    r = L.rsf_train(d, tr, n_estimators=RSF_TREES_TUNING,
                    max_depth=t.suggest_int("max_depth", 4, 12),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 50),
                    max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
    return L.cidx(d, val, r.predict(d.loc[val, C.FEATS].values))


s2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.RS))
s2.optimize(obj_rsf, n_trials=N_TRIALS_RSF)
br = s2.best_params
print("RSF best:", br)

# --- Reentrenamiento final y evaluación en prueba ---------------------------
# Se reentrena cada modelo con sus mejores hiperparámetros sobre entrenamiento
# más validación, y se evalúa sobre el conjunto de prueba (equipos no vistos).
xparams = {k: bx[k] for k in ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"]}
xparams["aft_loss_distribution_scale"] = bx["scale"]
mxgb = L.xgb_train(d, trv, params=xparams, nrounds=bx["nrounds"])
pred_xgb = L.xgb_predict(mxgb, d, te)
c_xgb = L.cidx(d, te, -pred_xgb)
rsf = L.rsf_train(d, trv, n_estimators=RSF_TREES_FINAL, max_depth=br["max_depth"],
                  min_samples_leaf=br["min_samples_leaf"], max_features=br["max_features"])
c_rsf = L.cidx(d, te, rsf.predict(d.loc[te, C.FEATS].values))

# --- Integrated Brier Score (calibración) -----------------------------------
times = L.ibs_grid(d, trv, te)
ibs = {}
try:
    ibs["xgb"] = L.ibs_xgb(d, trv, te, pred_xgb, bx["scale"], times)
except Exception as e:
    ibs["xgb"] = None
    print("IBS XGB:", str(e)[:80])
try:
    ibs["rsf"] = L.ibs_rsf(d, trv, te, rsf, times)
except Exception as e:
    ibs["rsf"] = None
    print("IBS RSF:", str(e)[:80])

res["comparacion"] = {
    "xgb_aft": {"c_index": round(c_xgb, 4), "ibs": None if ibs["xgb"] is None else round(ibs["xgb"], 4)},
    "rsf":     {"c_index": round(c_rsf, 4), "ibs": None if ibs["rsf"] is None else round(ibs["rsf"], 4)}}
res["mejores_hiperparametros"] = {"xgb_aft": bx, "rsf": br}
print("[COMPARACION]", json.dumps(res["comparacion"], ensure_ascii=False))

# --- Figura: dos paneles (discriminación y calibración) ---------------------
# Los títulos de cada panel identifican qué muestra cada subgráfico (son
# necesarios); no se usa un título general de figura, ya que esa descripción
# corresponde al pie de figura del documento. El hachurado distingue las barras
# sin depender solo del color (daltonismo).
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.5))
a.bar(["XGB-AFT", "RSF"], [c_xgb, c_rsf], color=[C.OI["azul"], C.OI["naranja"]], edgecolor=C.OI["negro"], hatch=["", "//"])
a.axhline(0.5, color=C.OI["negro"], ls=":")
a.set_ylim(0, 1)
a.set_ylabel("C-index (Harrell)")
a.set_title("Discriminación")
for i, v in enumerate([c_xgb, c_rsf]):
    a.text(i, v + 0.02, f"{v:.3f}".replace(".", ","), ha="center")
vx = [ibs.get("xgb") or 0, ibs.get("rsf") or 0]
b.bar(["XGB-AFT", "RSF"], vx, color=[C.OI["azul"], C.OI["naranja"]], edgecolor=C.OI["negro"], hatch=["", "//"])
b.set_ylabel("IBS (menor es mejor)")
b.set_title("Calibración (IBS)")
for i, v in enumerate(vx):
    b.text(i, v + 0.005, f"{v:.3f}".replace(".", ","), ha="center")
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "comparacion_xgb_vs_rsf.png", dpi=150)
plt.close(fig)

(C.OUTPUTS_DIR / "03_comparacion_modelos.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("Listo. Resultados en outputs/comparacion_xgb_vs_rsf.png y outputs/03_comparacion_modelos.json")
