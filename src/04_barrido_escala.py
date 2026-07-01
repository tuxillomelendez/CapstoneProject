"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 04 - Barrido de escala: XGBoost AFT frente a Random Survival Forest.

Mide el C-index, el IBS y el tiempo de entrenamiento de ambos modelos a tamaños
de muestra crecientes, para evidenciar la diferencia de escalabilidad: el Random
Survival Forest agota la memoria al crecer la muestra, mientras que XGBoost AFT
escala sin dificultad. Guarda resultados parciales tras cada tamaño, de modo que
una falla por memoria no descarte el trabajo previo.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/04_barrido_escala.png    (figura de respaldo, no incluida en el documento)
           outputs/04_barrido_escala.json   (métricas y tiempos por tamaño)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\04_barrido_escala.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
import config as C, lib as L

SIZES = [15000, 30000, 60000, 100000]   # agrega mas si va sobrado (ojo: el RSF es lento y come RAM)
N_TRIALS_XGB, N_TRIALS_RSF = 30, 15
RSF_TREES_TUNING, RSF_TREES_FINAL = 50, 150

print("Cargando dataset (una sola vez) ...")
DF = L.load_clean()
print(f"  {len(DF):,} filas disponibles")
OUT = C.OUTPUTS_DIR / "04_barrido_escala.json"

def corrida(N):
    d = L.submuestrear(DF, N, balanceado=True)
    idx = np.arange(len(d))
    trv, te = L.gsplit(d, idx, 0.2, C.RS); tr, val = L.gsplit(d, trv, 0.25, C.RS + 1)

    def oxgb(t):
        p = dict(max_depth=t.suggest_int("max_depth", 3, 8),
                 learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                 subsample=t.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                 aft_loss_distribution_scale=t.suggest_float("scale", 0.5, 2.0))
        m = L.xgb_train(d, tr, params=p, nrounds=t.suggest_int("nrounds", 50, 300))
        return L.cidx(d, val, -L.xgb_predict(m, d, val))
    sx = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.RS))
    sx.optimize(oxgb, n_trials=N_TRIALS_XGB); bx = sx.best_params

    def orsf(t):
        r = L.rsf_train(d, tr, n_estimators=RSF_TREES_TUNING,
                        max_depth=t.suggest_int("max_depth", 4, 12),
                        min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 50),
                        max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
        return L.cidx(d, val, r.predict(d.loc[val, C.FEATS].values))
    sr = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.RS))
    sr.optimize(orsf, n_trials=N_TRIALS_RSF); br = sr.best_params

    # XGB: fit + pred con tiempos
    xp = {k: bx[k] for k in ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"]}
    xp["aft_loss_distribution_scale"] = bx["scale"]
    t0 = time.time(); mx = L.xgb_train(d, trv, params=xp, nrounds=bx["nrounds"]); xfit = time.time() - t0
    t0 = time.time(); px = L.xgb_predict(mx, d, te); xpred = time.time() - t0
    c_xgb = L.cidx(d, te, -px)

    # RSF: fit + pred con tiempos
    t0 = time.time()
    rsf = L.rsf_train(d, trv, n_estimators=RSF_TREES_FINAL, max_depth=br["max_depth"],
                      min_samples_leaf=br["min_samples_leaf"], max_features=br["max_features"])
    rfit = time.time() - t0
    t0 = time.time(); pr = rsf.predict(d.loc[te, C.FEATS].values); rpred = time.time() - t0
    c_rsf = L.cidx(d, te, pr)

    times = L.ibs_grid(d, trv, te); ibs = {}
    try: ibs["xgb"] = round(L.ibs_xgb(d, trv, te, px, bx["scale"], times), 4)
    except Exception: ibs["xgb"] = None
    try: ibs["rsf"] = round(L.ibs_rsf(d, trv, te, rsf, times), 4)
    except Exception: ibs["rsf"] = None

    return {"N": int(N), "n_equipos_test": int(d.loc[te, C.ID_COL].nunique()),
            "xgb": {"c_index": round(c_xgb, 4), "ibs": ibs["xgb"], "fit_s": round(xfit, 3), "pred_s": round(xpred, 4)},
            "rsf": {"c_index": round(c_rsf, 4), "ibs": ibs["rsf"], "fit_s": round(rfit, 3), "pred_s": round(rpred, 4)}}

results = []
for N in SIZES:
    print(f"\n=== N={N:,} ===")
    try:
        r = corrida(N); results.append(r)
        print(f"  XGB c={r['xgb']['c_index']} fit {r['xgb']['fit_s']}s | RSF c={r['rsf']['c_index']} fit {r['rsf']['fit_s']}s")
    except Exception as e:
        results.append({"N": int(N), "error": str(e)[:200]}); print(f"  ERROR: {str(e)[:140]}")
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))   # guardado incremental

ok = [r for r in results if "xgb" in r and "rsf" in r]
if ok:
    Ns = [r["N"] for r in ok]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.plot(Ns, [r["xgb"]["c_index"] for r in ok], color=C.OI["azul"], marker="o", ls="-", label="XGB-AFT")
    a.plot(Ns, [r["rsf"]["c_index"] for r in ok], color=C.OI["naranja"], marker="^", ls="--", label="RSF")
    a.set_xscale("log"); a.set_ylim(0.5, 0.8); a.set_xlabel("N"); a.set_ylabel("C-index"); a.set_title("Discriminacion"); a.legend(); a.grid(alpha=0.3, which="both")
    b.plot(Ns, [r["xgb"]["fit_s"] for r in ok], color=C.OI["azul"], marker="o", ls="-", label="XGB-AFT")
    b.plot(Ns, [r["rsf"]["fit_s"] for r in ok], color=C.OI["naranja"], marker="^", ls="--", label="RSF")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("N"); b.set_ylabel("Tiempo entrenamiento (s)"); b.set_title("Costo computacional"); b.legend(); b.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(C.OUTPUTS_DIR / "04_barrido_escala.png", dpi=150); plt.close(fig)
print("OK -> outputs/04_*")
