"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 10 - Caracterización del cluster crítico (análisis geográfico).

Toma la asignación de clusters producida por el script 08 y describe cómo se
reparten los equipos del cluster crítico por comuna, marca de módem, operador y
tipo de equipo. Esto ayuda a orientar las acciones operativas (reemplazo,
mantenimiento correctivo generalizado, priorización en el plan anual).

El cluster crítico se lee del resultado del script 08 (no se fija a mano), de
modo que el análisis sigue siendo válido aunque cambie la numeración interna del
agrupamiento.

Entradas:  TESIS_MANTENEDOR -> excel del mantenedor de equipos.
           outputs/eda6_asignacion_clusters.xlsx (producido por el script 08).
           outputs/08_eda.json (para el número del cluster crítico).
Salidas:   outputs/cluster3_distribucion.png
           outputs/10_geografico.json

Uso:
    python src\\08_eda_caracterizacion.py   (debe correrse antes)
    python src\\10_geografico.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def coma(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def cargar():
    if not C.MANTENEDOR_PATH:
        raise SystemExit("Define TESIS_MANTENEDOR (excel del mantenedor).")
    asign_path = C.OUTPUTS_DIR / "eda6_asignacion_clusters.xlsx"
    if not asign_path.exists():
        raise SystemExit("Falta outputs/eda6_asignacion_clusters.xlsx. Corre antes el script 08.")

    man = pd.read_excel(C.MANTENEDOR_PATH, header=1)
    man["Numpos"] = pd.to_numeric(man["Numpos"], errors="coerce")
    man = man.dropna(subset=["Numpos"])
    man["Numpos"] = man["Numpos"].astype(int)

    asign = pd.read_excel(asign_path)

    # Número del cluster crítico: se toma del script 08; si no, el de mayor nivel medio de fallas.
    try:
        j = json.loads((C.OUTPUTS_DIR / "08_eda.json").read_text())
        critico = int(j["clustering"]["cluster_critico"])
    except Exception:
        critico = int(asign.groupby("cluster")["n_fallas"].mean().idxmax())
    return man, asign, critico


def figura_cluster_critico(man, asign, critico, res):
    equipos = asign.loc[asign["cluster"] == critico, "Numpos"].tolist()
    mc = man[man["Numpos"].isin(equipos)].copy()
    ttf_med = float(asign.loc[asign["cluster"] == critico, "ttf_median_h"].median())
    res["cluster_critico"] = {"numero": int(critico), "n_equipos": int(len(equipos)),
                              "ttf_mediano_h": round(ttf_med, 2)}

    # Figura 2x2: comuna, marca, operador y tipo. Barras en azul Okabe-Ito; las
    # categorías se leen en los ejes, no por color (accesibilidad).
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    com = mc["Comuna"].value_counts().head(10)
    ax.barh(range(len(com)), com.values, color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax.set_yticks(range(len(com)))
    ax.set_yticklabels(com.index)
    ax.invert_yaxis()
    ax.set_xlabel("Nº de equipos")
    ax.set_title("Distribución por comuna (top 10)")
    for i, v in enumerate(com.values):
        ax.text(v + 0.3, i, f"{v}", va="center", fontsize=9)

    ax = axes[0, 1]
    mar = mc["Marca Modem"].value_counts()
    ax.bar(range(len(mar)), mar.values, color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax.set_xticks(range(len(mar)))
    ax.set_xticklabels(mar.index, rotation=20, ha="right")
    ax.set_ylabel("Nº de equipos")
    ax.set_title("Distribución por marca de módem")
    for i, v in enumerate(mar.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    op = mc["Operador"].value_counts()
    tot = op.sum()
    ax.bar(range(len(op)), op.values, color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax.set_xticks(range(len(op)))
    ax.set_xticklabels(op.index)
    ax.set_ylabel("Nº de equipos")
    ax.set_title("Distribución por operador")
    for i, v in enumerate(op.values):
        ax.text(i, v, f"{v}\n({coma(100 * v / tot)} %)", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    tp = mc["Tipo de Equipo"].value_counts()
    ax.bar(range(len(tp)), tp.values, color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax.set_xticks(range(len(tp)))
    ax.set_xticklabels(tp.index, rotation=20, ha="right")
    ax.set_ylabel("Nº de equipos")
    ax.set_title("Distribución por tipo de equipo")
    for i, v in enumerate(tp.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Caracterización del cluster crítico ({len(equipos)} equipos)",
                 fontweight="bold", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "cluster3_distribucion.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    res = {}
    man, asign, critico = cargar()
    print(f"cluster crítico: {critico} | equipos: {(asign['cluster'] == critico).sum()}")
    figura_cluster_critico(man, asign, critico, res)
    print("geográfico:", res.get("cluster_critico"))
    (C.OUTPUTS_DIR / "10_geografico.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("Listo. Figura cluster3_distribucion.png y outputs/10_geografico.json generados.")
