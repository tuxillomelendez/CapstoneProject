"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Módulo de funciones compartidas.

Reúne la lógica común a todos los scripts: carga y limpieza de datos,
submuestreo, partición por equipo, entrenamiento de XGBoost AFT y de Random
Survival Forest, y el cálculo de las métricas de evaluación con censura
(C-index de Harrell y de Uno, e Integrated Brier Score).
"""
import numpy as np, pandas as pd
import xgboost as xgb
from scipy.stats import norm
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            integrated_brier_score)
import config as C


# --------------------------- carga ---------------------------
def load_clean(path=None, con_lags=False):
    """Carga el parquet, limpia inf/nan en features y descarta duration<=0.
    Agrega columnas dur_h (horas) y evt (bool). Lee solo columnas presentes."""
    path = path or C.DATA_PATH
    feats = C.FEATS_LAG if con_lags else C.FEATS
    cols = list(dict.fromkeys(feats + [C.DURATION_COL, C.EVENT_COL, C.ID_COL]))
    if con_lags:
        cols.append(C.TARGET_COL)
    disponibles = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in dict.fromkeys(cols) if c in disponibles]
    d = pd.read_parquet(path, columns=cols)
    for c in feats:
        if c in d.columns and d[c].dtype.kind == "f":
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)
            d[c] = d[c].fillna(d[c].median())
    d = d[d[C.DURATION_COL] > 0].reset_index(drop=True)
    d["dur_h"] = d[C.DURATION_COL] / C.SEC
    d["evt"]   = d[C.EVENT_COL].astype(bool)
    return d


# --------------------------- submuestreo ---------------------------
def submuestrear(d, n, balanceado=False, rs=None):
    """n filas. balanceado=True => 50/50 evento/censura; False => proporcion natural."""
    rs = C.RS if rs is None else rs
    if n is None or n >= len(d):
        return d.reset_index(drop=True)
    if balanceado:
        k = n // 2
        ev = d[d["evt"]].sample(n=min(k, int(d["evt"].sum())), random_state=rs)
        ce = d[~d["evt"]].sample(n=min(k, int((~d["evt"]).sum())), random_state=rs)
        return pd.concat([ev, ce]).sample(frac=1, random_state=rs).reset_index(drop=True)
    return d.sample(n=n, random_state=rs).reset_index(drop=True)


# --------------------------- split por equipo ---------------------------
def gsplit(d, idx, test_size=0.2, seed=None):
    """Split agrupado por equipo: ningun Numpos en train y test a la vez."""
    seed = C.RS if seed is None else seed
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                  .split(idx, groups=d.loc[idx, C.ID_COL].values))
    return idx[tr], idx[te]


# --------------------------- helpers sksurv ---------------------------
def surv(d, ix):
    return Surv.from_arrays(d.loc[ix, C.EVENT_COL].astype(bool).values,
                            (d.loc[ix, C.DURATION_COL] / C.SEC).values)

def cidx(d, ix, risk):
    """C-index de Harrell. risk: mayor = mas riesgo (en AFT pasar -pred)."""
    return float(concordance_index_censored(
        d.loc[ix, C.EVENT_COL].astype(bool).values,
        (d.loc[ix, C.DURATION_COL] / C.SEC).values, risk)[0])

def cindex_uno(d, tr, te, risk_te, tau=None):
    """C-index de Uno (IPCW). risk_te: mayor = mas riesgo. Devuelve (cindex, tau)."""
    y_tr, y_te = surv(d, tr), surv(d, te)
    if tau is None:
        dur = (d.loc[te, C.DURATION_COL] / C.SEC).values
        evt = d.loc[te, C.EVENT_COL].astype(bool).values
        tau = float(np.percentile(dur[evt], 95))
    return float(concordance_index_ipcw(y_tr, y_te, risk_te, tau=tau)[0]), tau


# --------------------------- XGBoost-AFT ---------------------------
def xgb_train(d, tr, feats=None, params=None, nrounds=None, device="cpu"):
    """Entrena XGBoost-AFT (censura por la derecha). device='cuda' usa GPU."""
    feats = feats or C.FEATS
    y = (d.loc[tr, C.DURATION_COL] / C.SEC).values
    e = d.loc[tr, C.EVENT_COL].astype(bool).values
    dtr = xgb.DMatrix(d.loc[tr, feats].values, label=y)
    lo, hi = y.copy(), y.copy(); hi[~e] = np.inf
    dtr.set_float_info("label_lower_bound", lo)
    dtr.set_float_info("label_upper_bound", hi)
    p = dict(C.XGB_AFT_PARAMS)
    if params:
        p.update(params)
    if int(xgb.__version__.split(".")[0]) >= 2:      # XGBoost 2.x: device='cuda'
        p["tree_method"] = "hist"; p["device"] = device
    else:                                            # XGBoost 1.x: gpu_hist
        p["tree_method"] = "gpu_hist" if device == "cuda" else "hist"
    return xgb.train(p, dtr, num_boost_round=nrounds or C.NUM_BOOST_ROUND)

def xgb_predict(m, d, ix, feats=None):
    feats = feats or C.FEATS
    return m.predict(xgb.DMatrix(d.loc[ix, feats].values))


# --------------------------- Random Survival Forest ---------------------------
def rsf_train(d, tr, feats=None, n_estimators=200, max_depth=None,
              min_samples_leaf=10, max_features="sqrt"):
    feats = feats or C.FEATS
    return RandomSurvivalForest(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        n_jobs=-1, random_state=C.RS).fit(d.loc[tr, feats].values, surv(d, tr))


# --------------------------- IBS (Integrated Brier Score) ---------------------------
def ibs_grid(d, trv, te, cap=168.0, n=22):
    """Grilla de tiempos estrictamente DENTRO del seguimiento comun train/test."""
    y_trv = (d.loc[trv, C.DURATION_COL] / C.SEC).values
    yt    = (d.loc[te,  C.DURATION_COL] / C.SEC).values
    lo = max(y_trv.min(), yt.min()); hi = min(y_trv.max(), yt.max(), cap)
    return np.linspace(lo, hi, n)[1:-1]

def ibs_xgb(d, trv, te, pred_te, sigma, times):
    """IBS de XGBoost-AFT usando la sobrevivencia log-normal implicada por el AFT."""
    mu = np.log(np.clip(pred_te, 1e-6, None))
    pr = np.column_stack([1 - norm.cdf((np.log(t) - mu) / sigma) for t in times])
    return float(integrated_brier_score(surv(d, trv), surv(d, te), pr, times))

def ibs_rsf(d, trv, te, rsf, times, feats=None, cap_filas=4000):
    """IBS de RSF (predict_survival_function es caro: se acota el test)."""
    feats = feats or C.FEATS
    ix = te if len(te) <= cap_filas else np.random.RandomState(C.RS).choice(te, cap_filas, replace=False)
    sf = rsf.predict_survival_function(d.loc[ix, feats].values)
    pr = np.row_stack([[fn(t) for t in times] for fn in sf])
    return float(integrated_brier_score(surv(d, trv), surv(d, ix), pr, times))
