"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Chile, 2026.

Script 01 - Diagnóstico de la fuga de información en las variables de rezago.

Este script demuestra empíricamente por qué las variables de rezago (lag) no
pueden usarse como predictores: el rezago de primer orden equivale, en la
práctica, a la variable objetivo desplazada en un intervalo de muestreo. Para
ello calcula la correlación entre el rezago y el objetivo, mide la diferencia
mediana entre ambos, y genera un gráfico de dispersión que evidencia la
relación casi perfecta a lo largo de la diagonal.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/diagnostico_fuga_lags.png    (gráfico de dispersión)
           outputs/01_diagnostico_leakage.json  (métricas del diagnóstico)

Uso:
    # Forma recomendada: definir la ruta de los datos por variable de entorno.
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\01_diagnostico_leakage.py
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

import config as C
import lib as L

# --- Carga de datos ---------------------------------------------------------
# Se cargan los datos incluyendo las variables de rezago, ya que el objetivo de
# este script es justamente analizarlas.
d = L.load_clean(con_lags=True)
print(f"{len(d):,} filas | {d[C.ID_COL].nunique()} equipos")

# --- Cuantificacion de la fuga ----------------------------------------------
# Si el rezago fuese el objetivo disfrazado, esperariamos dos cosas: una
# correlacion cercana a 1 entre lag1 y el objetivo, y una diferencia mediana
# entre ambos cercana al intervalo de muestreo (aproximadamente 0,25 horas).
corr   = float(d["time_to_failure_lag1"].corr(d[C.TARGET_COL]))
diff_h = float((d["time_to_failure_lag1"] - d[C.TARGET_COL]).median()) / C.SEC
print(f"corr(lag1, target) = {corr:.5f} | mediana(lag1 - target) = {diff_h:.3f} h")

# --- Grafico de dispersion: lag1 frente al objetivo -------------------------
# Se toma una muestra para que el grafico sea legible. La nube de puntos deberia
# alinearse sobre la diagonal y = x, lo que visualiza la fuga.
s = d.sample(min(5000, len(d)), random_state=C.RS)
x = s[C.TARGET_COL] / C.SEC               # Objetivo, en horas.
y = s["time_to_failure_lag1"] / C.SEC     # Rezago de primer orden, en horas.

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x, y, s=8, alpha=0.35, color=C.OI["azul"], marker="o", edgecolors="none",
           label="Observaciones")
# Linea diagonal de referencia: sobre ella, el rezago es identico al objetivo.
lim = float(np.percentile(np.r_[x, y], 99))
ax.plot([0, lim], [0, lim], color=C.OI["negro"], lw=1.2, ls="--",
        label=f"y = x  (r = {corr:.3f})")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Tiempo hasta la falla, objetivo (horas)")
ax.set_ylabel("Tiempo hasta la falla del registro anterior, lag1 (horas)")
ax.legend(loc="upper left", frameon=False)
# Nota: la figura no lleva titulo embebido; su descripcion va en el pie de
# figura (caption) del documento, segun la convencion editorial.
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "diagnostico_fuga_lags.png", dpi=150)
plt.close(fig)

# --- Persistencia de las metricas -------------------------------------------
# Se guardan los valores numericos para citarlos en el documento y para
# verificar la reproducibilidad de la corrida.
(C.OUTPUTS_DIR / "01_diagnostico_leakage.json").write_text(
    json.dumps(
        {"corr_lag1_target": round(corr, 5),
         "mediana_lag1_menos_target_h": round(diff_h, 4)},
        indent=2, ensure_ascii=False))
print("Listo. Resultados en outputs/diagnostico_fuga_lags.png y outputs/01_diagnostico_leakage.json")
