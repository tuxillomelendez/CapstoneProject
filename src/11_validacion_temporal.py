"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 11 - Validación temporal (walk-forward) sin rezagos.

Complementa la separación por equipo del script 02 con la segunda salvaguarda
contra la fuga: la temporal. Se ordena el período por fecha, se divide en folds
sucesivos y, para cada uno, se entrena con todo lo anterior y se valida en el
tramo siguiente. Nunca se usa el futuro para predecir el pasado.

Se emplea el conjunto de variables sin rezagos (modelo corregido). Si el
walk-forward arroja un C-index parecido al de la separación por equipo, queda
confirmado que la capacidad predictiva no dependía de mirar hacia adelante.

Entradas:  TESIS_DATA (o config_local / conjunto de juguete) -> parquet de
           modelamiento que conserve la columna 'fecha'.
Salidas:   outputs/validacion_temporal.png
           outputs/11_validacion_temporal.json

Uso:
    set TESIS_DATA=C:\\ruta\\dataset_modelamiento.parquet   (Windows)
    python src\\11_validacion_temporal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import lib as L


def coma(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


# Carga el parquet conservando la columna temporal, que load_clean no incluye
# por no ser una variable del modelo. Se aplica la misma limpieza (inf/nan en
# las features y descarte de duraciones no positivas) y se ordena por fecha.
def cargar_con_fecha():
    disp = set(pq.ParquetFile(C.DATA_PATH).schema.names)
    if "fecha" not in disp:
        raise SystemExit(
            "El parquet no expone la columna 'fecha', necesaria para el walk-forward. "
            "Usa el parquet de modelamiento que conserva la marca temporal."
        )
    cols = [c for c in dict.fromkeys(C.FEATS + [C.DURATION_COL, C.EVENT_COL, C.ID_COL, "fecha"])
            if c in disp]
    d = pd.read_parquet(C.DATA_PATH, columns=cols)
    for c in C.FEATS:
        if c in d.columns and d[c].dtype.kind == "f":
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)
            d[c] = d[c].fillna(d[c].median())
    d = d[d[C.DURATION_COL] > 0].copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d.sort_values("fecha").reset_index(drop=True)


def walk_forward(d, res, n_folds=5):
    fmin, fmax = d["fecha"].min(), d["fecha"].max()
    rango = (fmax - fmin).days
    fold_size = max(rango // n_folds, 1)
    filas = []
    rng = np.random.RandomState(C.RS)

    for fold in range(n_folds):
        ini = fmin + pd.Timedelta(days=fold * fold_size)
        fin = fmin + pd.Timedelta(days=(fold + 1) * fold_size)
        tr = d.index[d["fecha"] < ini].to_numpy()
        te = d.index[(d["fecha"] >= ini) & (d["fecha"] < fin)].to_numpy()
        if len(tr) < 1000 or len(te) < 1000:
            continue
        # Submuestreo por eficiencia (mismo criterio del experimento original).
        if len(tr) > 100000:
            tr = rng.choice(tr, 100000, replace=False)
        if len(te) > 50000:
            te = rng.choice(te, 50000, replace=False)

        m = L.xgb_train(d, tr, nrounds=100)           # variables sin rezagos (C.FEATS por defecto)
        pred = L.xgb_predict(m, d, te)
        c = L.cidx(d, te, -pred)                       # mayor riesgo = menor TTF predicho
        pred_tr = L.xgb_predict(m, d, tr)              # predicción sobre el propio entrenamiento
        c_tr = L.cidx(d, tr, -pred_tr)                 # C-index en train: control de memorización
        dur_h = (d.loc[te, C.DURATION_COL] / C.SEC).values
        evt = d.loc[te, C.EVENT_COL].astype(bool).values
        mae = float(np.mean(np.abs(pred[evt] - dur_h[evt]))) if evt.any() else float("nan")
        filas.append({"fold": fold + 1, "inicio": str(ini.date()), "fin": str(fin.date()),
                      "n_train": int(len(tr)), "n_test": int(len(te)),
                      "c_index_train": round(c_tr, 4),
                      "c_index": round(c, 4), "mae_h": round(mae, 2)})
        print(f"  fold {fold + 1}: {ini.date()} - {fin.date()} | C-index train {c_tr:.4f} / test {c:.4f} | MAE {mae:.1f}h")

    df = pd.DataFrame(filas)
    res["walk_forward"] = {
        "n_folds_validos": int(len(df)),
        "c_index_train_promedio": round(float(df["c_index_train"].mean()), 4),
        "c_index_promedio": round(float(df["c_index"].mean()), 4),
        "c_index_std": round(float(df["c_index"].std()), 4),
        "brecha_train_test_promedio": round(float((df["c_index_train"] - df["c_index"]).mean()), 4),
        "mae_promedio_h": round(float(df["mae_h"].mean()), 2),
        "por_fold": filas,
    }

    # Figura: C-index por fold (con línea de azar) y MAE por fold.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(df["fold"], df["c_index"], color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax1.axhline(0.5, color=C.OI["negro"], ls=":", lw=1.5, label="Azar (0,5)")
    for x, v in zip(df["fold"], df["c_index"]):
        ax1.text(x, v, coma(v), ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("Fold temporal")
    ax1.set_ylabel("C-index (Harrell)")
    ax1.set_title("C-index por fold (validación temporal)")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(df["fold"])
    ax1.legend(frameon=False)

    ax2.bar(df["fold"], df["mae_h"], color=C.OI["naranja"], edgecolor=C.OI["negro"], hatch="//")
    for x, v in zip(df["fold"], df["mae_h"]):
        ax2.text(x, v, coma(v, 0), ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Fold temporal")
    ax2.set_ylabel("MAE (horas)")
    ax2.set_title("MAE por fold (validación temporal)")
    ax2.set_xticks(df["fold"])

    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "validacion_temporal.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    res = {}
    d = cargar_con_fecha()
    print(f"parquet con fecha: {len(d):,} filas | {d['fecha'].min().date()} a {d['fecha'].max().date()}")
    walk_forward(d, res)
    print("walk-forward C-index promedio:", res.get("walk_forward", {}).get("c_index_promedio"))
    (C.OUTPUTS_DIR / "11_validacion_temporal.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("Listo. Figura validacion_temporal.png y outputs/11_validacion_temporal.json generados.")
