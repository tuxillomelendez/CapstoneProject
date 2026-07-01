"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 07 - Ejemplo ilustrativo de datos de supervivencia (swimmer plot).

Genera un diagrama de seguimiento individual para diez equipos, siguiendo la
convención clásica del análisis de supervivencia (Arribalzaga, 2007): cada
equipo es una barra horizontal cuyo largo es su tiempo de seguimiento, ordenadas
de menor a mayor. Un punto al final indica una falla observada (evento); una
flecha indica una observación censurada, es decir, un equipo que no falló dentro
del período y cuyo tiempo real hasta la falla es mayor al observado. El objetivo
es ilustrar, de forma concreta, qué es la censura en estos datos.

Entradas:  conjunto de datos definido en config.py (toy por defecto).
Salidas:   outputs/ejemplo_supervivencia.png  (swimmer plot de 10 equipos)

Uso:
    set TESIS_DATA=C:\\ruta\\al\\dataset_modelamiento.parquet   (Windows)
    python src\\07_ejemplo_supervivencia.py
"""
import sys
from pathlib import Path

# Permite importar config.py y lib.py, que viven en esta misma carpeta (src/).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana: la figura se guarda directo en disco.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config as C
import lib as L

N_EJEMPLO = 10        # Número de equipos a mostrar.
N_CENSURADOS = 4      # Cuántos de ellos deben ser censurados (para ilustrar ambos casos).

# --- Carga de datos ---------------------------------------------------------
d = L.load_clean()
d["dur_h"] = d[C.DURATION_COL] / C.SEC

# --- Selección de equipos para el ejemplo -----------------------------------
# Se eligen equipos distintos que ilustren los dos casos: algunos con falla
# observada y algunos censurados. Al ser un ejemplo didáctico, la selección
# busca deliberadamente mostrar ambas situaciones, como en la convención del
# artículo de referencia.
equipos_con_censura = d.loc[d[C.EVENT_COL] == 0, C.ID_COL].drop_duplicates()
equipos_con_evento  = d.loc[d[C.EVENT_COL] == 1, C.ID_COL].drop_duplicates()

n_cens = min(N_CENSURADOS, len(equipos_con_censura))
n_even = N_EJEMPLO - n_cens

sel_cens = equipos_con_censura.sample(n_cens, random_state=C.RS).tolist() if n_cens > 0 else []
# Equipos con evento, distintos de los ya elegidos como censurados.
candidatos_evento = [e for e in equipos_con_evento.tolist() if e not in sel_cens]
sel_even = pd.Series(candidatos_evento).sample(min(n_even, len(candidatos_evento)),
                                               random_state=C.RS).tolist()

# Para cada equipo seleccionado se toma un episodio del estado correspondiente.
filas = []
for eq in sel_cens:
    filas.append(d[(d[C.ID_COL] == eq) & (d[C.EVENT_COL] == 0)].iloc[0])
for eq in sel_even:
    filas.append(d[(d[C.ID_COL] == eq) & (d[C.EVENT_COL] == 1)].iloc[0])

ej = pd.DataFrame(filas).reset_index(drop=True)
# Se ordenan de menor a mayor tiempo de seguimiento, como en la Figura 2 del artículo.
ej = ej.sort_values("dur_h").reset_index(drop=True)

# --- Figura: swimmer plot ---------------------------------------------------
# Cada equipo es una barra horizontal. El punto marca la falla observada; la
# flecha marca la censura. Se usan color y forma a la vez (daltonismo).
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, row in ej.iterrows():
    dur = row["dur_h"]
    es_evento = row[C.EVENT_COL] == 1
    color = C.OI["azul"] if es_evento else C.OI["naranja"]
    ax.plot([0, dur], [i, i], color=color, lw=2.2, solid_capstyle="round")
    if es_evento:
        ax.plot(dur, i, marker="o", color=color, markersize=9)          # Falla observada.
    else:
        ax.plot(dur, i, marker=">", color=color, markersize=11)         # Censura (tiempo mayor al observado).

ax.set_yticks(range(len(ej)))
ax.set_yticklabels([f"Equipo {i + 1}" for i in range(len(ej))])
ax.set_xlabel("Tiempo de seguimiento (horas)")
ax.set_xlim(0, None)
ax.margins(y=0.05)

leyenda = [
    Line2D([0], [0], color=C.OI["azul"], marker="o", lw=2.2, markersize=9,
           label="Falla observada (evento)"),
    Line2D([0], [0], color=C.OI["naranja"], marker=">", lw=2.2, markersize=11,
           label="Sin falla en el período (censurado)"),
]
ax.legend(handles=leyenda, frameon=False, loc="lower right")
# Sin título embebido: la descripción va en el pie de figura del documento.
fig.tight_layout()
fig.savefig(C.OUTPUTS_DIR / "ejemplo_supervivencia.png", dpi=150)
plt.close(fig)

print(f"Equipos en el ejemplo: {len(ej)} | censurados: {int((ej[C.EVENT_COL] == 0).sum())} | con falla: {int((ej[C.EVENT_COL] == 1).sum())}")
print("Listo. Resultado en outputs/ejemplo_supervivencia.png")
