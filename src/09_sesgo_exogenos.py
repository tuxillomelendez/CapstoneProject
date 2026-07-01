"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 09 - Análisis de sesgo por eventos exógenos.

Cuantifica cuántas fallas provienen de fenómenos que el modelo no puede
anticipar con datos endógenos ---mantenimientos programados (tickets) y eventos
masivos externos (temporales, blackout)--- y compara las métricas del parque
con y sin esos eventos. El objetivo es justificar el filtrado de fallas no
predecibles antes del modelamiento.

Las ventanas de los eventos exógenos corresponden a los picos específicos
detectados; difieren a propósito de las ventanas más amplias que el script 08
usa solo para señalizar las bandas en la serie temporal.

Entradas:  TESIS_TELEMETRIA -> parquet de telemetría cruda.
           TESIS_TICKETS    -> csv de tickets de mantenimiento.
Salidas:   outputs/analisis_sesgo_eventos_exogenos.png
           outputs/09_sesgo.json   (cuantificación con/sin eventos)

Uso:
    set TESIS_TELEMETRIA=C:\\ruta\\telemetria_cruda.parquet   (Windows)
    set TESIS_TICKETS=C:\\ruta\\maestro_tickets.csv
    python src\\09_sesgo_exogenos.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

# Ventanas de los eventos exógenos a filtrar (picos específicos del período).
EVENTOS = [
    {"nombre": "Temporal de viento (agosto 2024)", "inicio": "2024-08-01", "fin": "2024-08-10"},
    {"nombre": "Pico de febrero 2025",             "inicio": "2025-02-24", "fin": "2025-02-26"},
    {"nombre": "Pico de abril 2025",               "inicio": "2025-04-13", "fin": "2025-04-15"},
]


def cargar():
    if not C.TELEMETRIA_CRUDA_PATH or not C.TICKETS_PATH:
        raise SystemExit(
            "Faltan rutas. Define TESIS_TELEMETRIA (parquet de telemetría) y "
            "TESIS_TICKETS (csv de tickets de mantenimiento)."
        )
    df = pd.read_parquet(C.TELEMETRIA_CRUDA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    tk = pd.read_csv(C.TICKETS_PATH)
    tk["Fecha_Inicio"] = pd.to_datetime(tk["Fecha_Inicio"])
    tk["Fecha_fin"] = pd.to_datetime(tk["Fecha_fin"])
    return df, tk


def mascara_tickets(df_fallas, tk):
    """Marca las fallas que caen dentro de la ventana de algún ticket del equipo."""
    df_temp = df_fallas[["Numpos", "fecha"]].copy()
    df_temp["idx"] = df_temp.index
    m = df_temp.merge(tk[["Numpos", "Fecha_Inicio", "Fecha_fin"]], on="Numpos", how="inner")
    dentro = (m["fecha"] >= m["Fecha_Inicio"]) & (m["fecha"] <= m["Fecha_fin"])
    idx_tk = m.loc[dentro, "idx"].unique()
    return df_fallas.index.isin(idx_tk)


def mascara_eventos(fechas):
    """Marca las filas cuya fecha cae dentro de algún evento exógeno masivo."""
    m = pd.Series(False, index=fechas.index)
    for ev in EVENTOS:
        m |= (fechas >= ev["inicio"]) & (fechas <= ev["fin"])
    return m


def eda_sesgo(df, tk, res):
    fallas = df[df["Connection_Status"] == 1].copy()
    res["fallas_totales"] = int(len(fallas))

    # Filtro 1: fallas durante mantenimiento (tickets).
    m_tk = mascara_tickets(fallas, tk)
    fallas = fallas[~m_tk].copy()  # se trabaja sobre las fallas fuera de mantenimiento
    res["fallas_en_mantenimiento"] = int(m_tk.sum())

    # Filtro 2: eventos exógenos masivos.
    m_ev = mascara_eventos(fallas["fecha"])
    con = fallas.copy()
    sin = fallas[~m_ev].copy()

    equipos_con = con.groupby("Numpos").size()
    equipos_sin = sin.groupby("Numpos").size()
    dias_total = max((fallas["fecha"].max() - fallas["fecha"].min()).days, 1)
    cv_con = equipos_con.std() / equipos_con.mean()
    cv_sin = equipos_sin.std() / equipos_sin.mean()

    # Coincidencia del top-20 de equipos problemáticos con y sin eventos.
    top_con, top_sin = set(equipos_con.nlargest(20).index), set(equipos_sin.nlargest(20).index)
    coincidencia = 100 * len(top_con & top_sin) / 20

    res["sesgo"] = {
        "fallas_con_eventos": int(len(con)),
        "fallas_sin_eventos": int(len(sin)),
        "pct_fallas_por_eventos": round(100 * (len(con) - len(sin)) / len(con), 1),
        "media_por_equipo_con": round(float(equipos_con.mean()), 1),
        "media_por_equipo_sin": round(float(equipos_sin.mean()), 1),
        "cv_con": round(float(cv_con), 3),
        "cv_sin": round(float(cv_sin), 3),
        "coincidencia_top20_pct": round(coincidencia, 0),
    }

    # Figura 2x2 (paneles A-D). CON eventos en naranja, SIN eventos en azul; se
    # refuerza con textura en las barras para no depender solo del color.
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.hist(equipos_con.clip(0, 10000), bins=50, alpha=0.6, color=C.OI["naranja"],
            label=f"CON eventos (n = {len(equipos_con):,})".replace(",", "."), density=True)
    ax.hist(equipos_sin.clip(0, 10000), bins=50, alpha=0.6, color=C.OI["azul"],
            label=f"SIN eventos (n = {len(equipos_sin):,})".replace(",", "."), density=True)
    ax.axvline(equipos_con.mean(), color=C.OI["naranja"], ls="--", lw=2)
    ax.axvline(equipos_sin.mean(), color=C.OI["azul"], ls="--", lw=2)
    ax.set_xlabel("Fallas por equipo")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de fallas por equipo")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(-0.10, 1.05, "A", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    ax = axes[0, 1]
    bp = ax.boxplot([equipos_con.clip(0, 15000), equipos_sin.clip(0, 15000)],
                    labels=["CON eventos", "SIN eventos"], patch_artist=True)
    bp["boxes"][0].set_facecolor(C.OI["naranja"]); bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(C.OI["azul"]); bp["boxes"][1].set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color(C.OI["negro"])
    ax.set_ylabel("Fallas por equipo")
    ax.set_title("Comparación de distribución")
    ax.text(-0.10, 1.05, "B", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    ax = axes[1, 0]
    diarias = fallas.groupby(fallas["fecha"].dt.date).size()
    ax.fill_between(diarias.index, diarias.values, alpha=0.3, color=C.OI["azul"])
    ax.plot(diarias.index, diarias.values, color=C.OI["azul"], lw=1)
    for ev in EVENTOS:
        ini, fin = pd.Timestamp(ev["inicio"]).date(), pd.Timestamp(ev["fin"]).date()
        mask = [(d >= ini) & (d <= fin) for d in diarias.index]
        ax.fill_between(diarias.index, diarias.values, where=mask, alpha=0.6, color=C.OI["naranja"])
    prom_sin = len(sin) / dias_total
    ax.axhline(prom_sin, color=C.OI["negro"], ls="--", lw=2,
               label=f"Promedio sin eventos: {prom_sin:,.0f}".replace(",", "."))
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Fallas por día")
    ax.set_title("Serie temporal (naranja = eventos exógenos)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.text(-0.10, 1.05, "C", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    ax = axes[1, 1]
    metricas = ["Total", "Media", "Mediana", "Desv.", "CV"]
    val_con = [len(con) / 1e6, equipos_con.mean() / 1000, equipos_con.median() / 1000,
               equipos_con.std() / 1000, cv_con]
    val_sin = [len(sin) / 1e6, equipos_sin.mean() / 1000, equipos_sin.median() / 1000,
               equipos_sin.std() / 1000, cv_sin]
    x = np.arange(len(metricas))
    w = 0.35
    b1 = ax.bar(x - w / 2, val_con, w, label="CON eventos", color=C.OI["naranja"],
                edgecolor=C.OI["negro"], hatch="//")
    b2 = ax.bar(x + w / 2, val_sin, w, label="SIN eventos", color=C.OI["azul"], edgecolor=C.OI["negro"])
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)
    ax.set_ylabel("Valor (Total en millones; resto en miles)")
    ax.set_title("Comparación de métricas clave")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(-0.10, 1.05, "D", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "analisis_sesgo_eventos_exogenos.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    res = {}
    df, tk = cargar()
    print(f"telemetría: {len(df):,} filas | tickets: {len(tk):,}")
    eda_sesgo(df, tk, res)
    print("sesgo:", res.get("sesgo"))
    (C.OUTPUTS_DIR / "09_sesgo.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("Listo. Figura analisis_sesgo_eventos_exogenos.png y outputs/09_sesgo.json generados.")
