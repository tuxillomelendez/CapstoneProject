"""
crear_toy_publico.py — Genera el toy publico ANONIMO desde la data completa (privada).
Conserva solo features del modelo + objetivo/censura + lags (diagnostico); anonimiza
Numpos a EQ001.. y elimina identificadores/direcciones/timestamps reales.
Uso:  set TESIS_DATA=C:\\ruta\\dataset_modelamiento.parquet  &&  python crear_toy_publico.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import config as C

N_EQUIPOS = 30
OUT = C.DATA_DIR / "toy_publico_modelamiento.parquet"
rng = np.random.default_rng(C.RS)

SAFE_COLS = [C.ID_COL, C.TARGET_COL, "time_to_failure_lag1", "time_to_failure_lag2",
             C.DURATION_COL, C.EVENT_COL] + C.FEATS

df = pd.read_parquet(C.DATA_PATH)
sel = rng.choice(df[C.ID_COL].unique(), size=min(N_EQUIPOS, df[C.ID_COL].nunique()), replace=False)
d = df[df[C.ID_COL].isin(sel)].copy()
u = list(d[C.ID_COL].unique()); rng.shuffle(u)
d[C.ID_COL] = d[C.ID_COL].map({old: f"EQ{i+1:03d}" for i, old in enumerate(u)})

dropped = [c for c in d.columns if c not in SAFE_COLS]
d = d[[c for c in SAFE_COLS if c in d.columns]]
OUT.parent.mkdir(exist_ok=True)
d.to_parquet(OUT, index=False)
print(f"Toy publico: {len(d):,} filas | {d[C.ID_COL].nunique()} equipos -> {OUT}")
print(f"Conservadas: {list(d.columns)}")
print(f"Eliminadas (sensibles): {dropped}")
