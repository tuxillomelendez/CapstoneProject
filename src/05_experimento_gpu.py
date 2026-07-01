"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 05 - Escalabilidad de XGBoost AFT: CPU frente a GPU.

Compara el tiempo de entrenamiento en CPU y en GPU a tamaños crecientes,
verificando que el uso de GPU acelera el entrenamiento sin alterar la capacidad
discriminativa (el C-index es equivalente en ambos dispositivos). Resultado de
respaldo para la discusión sobre escalabilidad y trabajo futuro.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/05_experimento_gpu.png   (figura de respaldo, no incluida en el documento)
           outputs/05_experimento_gpu.json  (tiempos y C-index por dispositivo)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\05_experimento_gpu.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C, lib as L

SIZES = [100_000, 500_000, 1_000_000, 5_000_000]   # sube a 10M si la VRAM aguanta
CIDX_SAMPLE = 50_000

print("Cargando dataset (una sola vez) ...")
DF = L.load_clean()
print(f"  {len(DF):,} filas disponibles")
OUT = C.OUTPUTS_DIR / "05_experimento_gpu.json"

def evaluar(d, device, trv, te):
    t0 = time.time(); m = L.xgb_train(d, trv, feats=C.FEATS, device=device); fit_s = time.time() - t0
    t0 = time.time(); pred = L.xgb_predict(m, d, te, C.FEATS); pred_s = time.time() - t0
    sub = np.random.RandomState(C.RS).choice(len(te), min(CIDX_SAMPLE, len(te)), replace=False)
    c = L.cidx(d, te[sub], -pred[sub])
    return round(fit_s, 3), round(pred_s, 4), round(c, 4)

results = []
for N in SIZES:
    print(f"\n=== N={N:,} ===")
    try:
        d = L.submuestrear(DF, N, balanceado=False)
        idx = np.arange(len(d)); trv, te = L.gsplit(d, idx, 0.2, C.RS)
        row = {"N": int(N), "n_train": int(len(trv)), "n_test": int(len(te))}
        for dev in ["cpu", "cuda"]:
            try:
                fs, ps, c = evaluar(d, dev, trv, te)
                row[dev] = {"fit_s": fs, "pred_s": ps, "c_index": c}
                print(f"  {dev.upper():4} entrena {fs:8.2f}s | predice {ps:7.3f}s | c={c:.4f}")
            except Exception as e:
                row[dev] = {"error": str(e)[:200]}; print(f"  {dev.upper():4} ERROR: {str(e)[:120]}")
        if "fit_s" in row.get("cpu", {}) and "fit_s" in row.get("cuda", {}):
            row["speedup_fit"] = round(row["cpu"]["fit_s"] / max(row["cuda"]["fit_s"], 1e-9), 1)
            print(f"  --> speedup GPU: {row['speedup_fit']}x")
        results.append(row)
    except Exception as e:
        results.append({"N": int(N), "error": str(e)[:200]}); print(f"  ERROR N={N}: {str(e)[:140]}")
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

ok = [r for r in results if "fit_s" in r.get("cpu", {}) and "fit_s" in r.get("cuda", {})]
if ok:
    Ns = [r["N"] for r in ok]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.plot(Ns, [r["cpu"]["fit_s"] for r in ok], color=C.OI["azul"], marker="o", ls="-", label="CPU")
    a.plot(Ns, [r["cuda"]["fit_s"] for r in ok], color=C.OI["naranja"], marker="^", ls="--", label="GPU")
    for r in ok:
        if "speedup_fit" in r: a.annotate(f"{r['speedup_fit']}x", (r["N"], r["cuda"]["fit_s"]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=C.OI["naranja"])
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("N"); a.set_ylabel("Tiempo entrenamiento (s)"); a.set_title("Costo: CPU vs GPU"); a.legend(); a.grid(alpha=0.3, which="both")
    b.plot(Ns, [r["cpu"]["c_index"] for r in ok], color=C.OI["azul"], marker="o", ls="-", label="CPU")
    b.plot(Ns, [r["cuda"]["c_index"] for r in ok], color=C.OI["naranja"], marker="^", ls="--", label="GPU")
    b.set_xscale("log"); b.set_ylim(0.5, 0.8); b.set_xlabel("N"); b.set_ylabel("C-index"); b.set_title("Discriminacion identica"); b.legend(); b.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(C.OUTPUTS_DIR / "05_experimento_gpu.png", dpi=150); plt.close(fig)
print("OK -> outputs/05_*")
