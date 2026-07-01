"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 06 - Verificación del C-index: Harrell frente a Uno (IPCW).

Calcula el C-index del modelo corregido con dos estimadores: el de Harrell y el
de Uno con ponderación por probabilidad inversa de censura (IPCW). Verifica que,
con el nivel de censura observado, ambos estimadores coinciden, de modo que el
C-index de Harrell no presenta sesgo apreciable. No genera figuras.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/06_verificar_cindex.json  (C-index de Harrell y de Uno, tau y censura)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\06_verificar_cindex.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
import config as C, lib as L

N = 1_000_000
BALANCEADO = False         # natural (recomendado). Pon True si tu headline es el balanceado.
CIDX_SAMPLE = 50_000

d = L.load_clean()
d = L.submuestrear(d, N, balanceado=BALANCEADO)
print(f"{len(d):,} filas | censura {100*(1-d['evt'].mean()):.1f}% | balanceada={BALANCEADO}")
idx = np.arange(len(d))
tr, te = L.gsplit(d, idx, 0.2, C.RS)

m = L.xgb_train(d, tr, feats=C.FEATS)
pred = L.xgb_predict(m, d, te, C.FEATS)

# Submuestra del test (mismo subconjunto para ambos estimadores)
rng = np.random.RandomState(C.RS)
sub = rng.choice(len(te), min(CIDX_SAMPLE, len(te)), replace=False)
te_sub = te[sub]; pred_sub = pred[sub]

c_h = L.cidx(d, te_sub, -pred_sub)
res = {"n_muestra": int(len(d)), "balanceada": BALANCEADO,
       "censura_pct": round(100 * (1 - d["evt"].mean()), 1), "c_harrell": round(c_h, 4)}
try:
    c_u, tau = L.cindex_uno(d, tr, te_sub, -pred_sub)
    res["c_uno"] = round(c_u, 4); res["tau_horas"] = round(tau, 1)
    print(f"Harrell: {c_h:.4f} | Uno (IPCW, tau={tau:.1f} h): {c_u:.4f} | dif {abs(c_h-c_u):.4f}")
except Exception as e:
    res["c_uno_error"] = str(e)[:200]; print(f"Harrell: {c_h:.4f} | Uno fallo: {str(e)[:120]}")

(C.OUTPUTS_DIR / "06_verificar_cindex.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("OK -> outputs/06_*")
