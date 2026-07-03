"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Chile, 2026.

Script 08 - Análisis exploratorio (EDA) y caracterización del parque.

Parte de la telemetría cruda y del mantenedor (no del parquet de modelamiento)
para construir métricas por equipo y producir las figuras de caracterización:
concentración de fallas (Pareto y Lorenz), evolución temporal, distribución por
cuartiles, mortalidad infantil, fallas y TTF por categoría, y agrupamiento.

Todas las figuras usan la paleta Okabe-Ito y refuerzan la lectura con formas o
estilos de línea (no solo color), por accesibilidad. Los rótulos numéricos van
con coma decimal, en coherencia con el documento.

Entradas:  TESIS_TELEMETRIA -> parquet de telemetría cruda.
           TESIS_MANTENEDOR -> excel del mantenedor de equipos.
Salidas:   outputs/eda1_pareto_fallas.png          (concentración: Pareto)
           outputs/eda1_curva_lorenz.png           (concentración: Lorenz)
           outputs/eda1_fallas_tiempo.png          (evolución temporal y eventos)
           outputs/eda2_distribucion_cuartiles.png (cuartiles de fallas)
           outputs/eda3_fallas_por_grupo_edad.png  (mortalidad infantil + test)
           outputs/eda4_fallas_por_categoria.png   (fallas por operador/tipo/marca/comuna)
           outputs/eda4_heatmap_marca_operador.png (mapa de calor marca x operador)
           outputs/eda5_ttf_por_categoria.png      (TTF por categoría)
           outputs/eda6_clusters_caracteristicas.png (agrupamiento de equipos)
           outputs/eda6_asignacion_clusters.xlsx   (insumo del análisis geográfico)
           outputs/08_eda.json                     (estadísticas para citar en el texto)

Uso:
    set TESIS_TELEMETRIA=C:\\ruta\\telemetria_cruda.parquet   (Windows)
    set TESIS_MANTENEDOR=C:\\ruta\\Mantenedor.xlsx
    python src\\08_eda_caracterizacion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana: las figuras se guardan directo en disco.
import matplotlib.pyplot as plt
import seaborn as sns

import config as C


def coma(x, dec=1):
    """Formatea un número con coma decimal (convención del documento)."""
    return f"{x:.{dec}f}".replace(".", ",")


# --- Carga de datos crudos --------------------------------------------------
# La telemetría aporta el estado de conexión y el tiempo hasta la falla; el
# mantenedor aporta los atributos de cada equipo (comuna, operador, marca,
# tipo) y la fecha de instalación, de la que se deriva la antigüedad.
def cargar():
    if not C.TELEMETRIA_CRUDA_PATH or not C.MANTENEDOR_PATH:
        raise SystemExit(
            "Faltan rutas de datos crudos. Define las variables de entorno:\n"
            "  TESIS_TELEMETRIA -> parquet de telemetría cruda\n"
            "  TESIS_MANTENEDOR -> excel del mantenedor de equipos"
        )
    tel = pd.read_parquet(C.TELEMETRIA_CRUDA_PATH)
    if "fecha" in tel.columns:
        tel["fecha"] = pd.to_datetime(tel["fecha"])

    man = pd.read_excel(C.MANTENEDOR_PATH, header=1)
    man["Numpos"] = pd.to_numeric(man["Numpos"], errors="coerce")
    man = man.dropna(subset=["Numpos"])
    man["Numpos"] = man["Numpos"].astype(int)

    # Fecha de instalación (formato DD-MM-YYYY) y antigüedad en años.
    def parse_fecha(s):
        if pd.isna(s) or s in ["00-00-0000", "NaN", ""]:
            return pd.NaT
        return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")

    man["fecha_instalacion"] = man["Fecha PS Telecontrol"].apply(parse_fecha)
    ref = datetime(2024, 12, 1)  # Fecha de referencia para calcular la antigüedad.
    man["edad_anios"] = ((ref - man["fecha_instalacion"]).dt.days / 365.25).clip(lower=0)
    return tel, man


# --- Métricas por equipo ----------------------------------------------------
# Se agrega la telemetría a nivel de equipo: número de fallas (Connection_Status
# igual a 1 marca desconexión), total de registros, TTF mediano, días de
# operación y tasa de fallas; luego se cruza con los atributos del mantenedor.
def metricas_por_equipo(tel, man):
    g = tel.groupby("Numpos")
    fallas = tel[tel["Connection_Status"] == 1].groupby("Numpos").size().rename("n_fallas")
    regs = g.size().rename("n_registros")

    # TTF mediano por equipo (horas). Si la telemetria no trae 'time_to_failure'
    # (caso de la telemetria cruda), se calcula el tiempo hasta la proxima
    # desconexion, replicando el procedimiento del analisis exploratorio original.
    # Se trabaja solo con tres columnas para no duplicar en memoria toda la tabla.
    if "time_to_failure" in tel.columns:
        ttf = (g["time_to_failure"].median() / 3600.0).rename("ttf_median_h")
    else:
        t = tel[["Numpos", "fecha", "Connection_Status"]].copy()
        t["helper_time"] = pd.NaT
        off = t["Connection_Status"] == 1
        t.loc[off, "helper_time"] = t.loc[off, "fecha"]
        # Ordenando por fecha descendente, ffill propaga hacia los registros
        # anteriores la fecha de la PROXIMA desconexion de cada equipo.
        t = t.sort_values(["Numpos", "fecha"], ascending=[True, False])
        t["next_offline"] = t.groupby("Numpos")["helper_time"].ffill()
        t["ttf_seconds"] = (t["next_offline"] - t["fecha"]).dt.total_seconds()
        ttf = (t.groupby("Numpos")["ttf_seconds"].median() / 3600.0).rename("ttf_median_h")
        del t

    fmin, fmax = g["fecha"].min(), g["fecha"].max()
    dias = (fmax - fmin).dt.days.rename("dias_operacion")

    df = pd.concat([fallas, regs, ttf, dias], axis=1)
    df.index.name = "Numpos"          # el concat puede perder el nombre del indice
    df = df.reset_index()
    df["tasa_fallas_dia"] = df["n_fallas"] / df["dias_operacion"].replace(0, 1)

    cols = ["Numpos", "Comuna", "Operador", "Marca Modem", "Modelo Modem",
            "Tipo de Equipo", "Antena", "edad_anios", "fecha_instalacion"]
    df = df.merge(man[cols], on="Numpos", how="left")
    df["n_fallas"] = df["n_fallas"].fillna(0)
    return df


# --- EDA 1: concentración de fallas (Pareto, Lorenz y evolución temporal) ---
def eda1_concentracion(dfm, tel, res):
    s = dfm[dfm["n_fallas"] > 0].sort_values("n_fallas", ascending=False).reset_index(drop=True)
    total = s["n_fallas"].sum()
    s["pct_fallas_acum"] = 100 * s["n_fallas"].cumsum() / total
    s["pct_equipos"] = 100 * (s.index + 1) / len(s)

    pct_eq_80 = len(s[s["pct_fallas_acum"] <= 80]) / len(s) * 100
    pct_top10 = s.head(10)["n_fallas"].sum() / total * 100
    n20 = int(len(s) * 0.20)
    pct_top20 = s.head(n20)["n_fallas"].sum() / total * 100
    res["concentracion"] = {
        "equipos_con_fallas": int(len(s)),
        "total_fallas": int(total),
        "fallas_media_por_equipo": round(float(s["n_fallas"].mean()), 1),
        "fallas_mediana_por_equipo": round(float(s["n_fallas"].median()), 1),
        "pct_equipos_que_concentra_80": round(pct_eq_80, 1),
        "pct_fallas_top10": round(pct_top10, 1),
        "pct_fallas_top20pct": round(pct_top20, 1),
    }

    # Figura: Pareto (barras de fallas + curva de % acumulado, doble eje).
    n = min(100, len(s))
    x = range(n)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x, s["n_fallas"].head(n), color=C.OI["azul"], label="Fallas por equipo")
    ax1.set_xlabel("Equipos (ordenados por nº de fallas)")
    ax1.set_ylabel("Nº de fallas")
    ax2 = ax1.twinx()
    ax2.plot(x, s["pct_fallas_acum"].head(n), color=C.OI["naranja"], lw=2, label="% acumulado")
    ax2.axhline(80, color=C.OI["negro"], ls=":", lw=1.5, label="80 %")
    ax2.set_ylabel("% acumulado de fallas")
    ax2.set_ylim(0, 105)
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax2.legend(l1 + l2, la1 + la2, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda1_pareto_fallas.png", dpi=150)
    plt.close(fig)

    # Figura: curva de Lorenz (concentración frente a la igualdad perfecta).
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(s["pct_equipos"], s["pct_fallas_acum"], color=C.OI["azul"], lw=2, label="Distribución real")
    ax.plot([0, 100], [0, 100], ls="--", color=C.OI["gris"], lw=1.5, label="Igualdad perfecta")
    ax.fill_between(s["pct_equipos"], s["pct_fallas_acum"], s["pct_equipos"], alpha=0.25, color=C.OI["celeste"])
    ax.axhline(80, color=C.OI["naranja"], ls=":", alpha=0.7)
    ax.axvline(pct_eq_80, color=C.OI["naranja"], ls=":", alpha=0.7)
    ax.scatter([pct_eq_80], [80], color=C.OI["naranja"], marker="D", s=90, zorder=5, edgecolor=C.OI["negro"])
    ax.annotate(f"{coma(pct_eq_80)} % de equipos\n= 80 % de fallas",
                xy=(pct_eq_80, 80), xytext=(min(pct_eq_80 + 10, 60), 66),
                fontsize=10, arrowprops=dict(arrowstyle="->", color=C.OI["negro"]))
    ax.set_xlabel("% de equipos (acumulado)")
    ax.set_ylabel("% de fallas (acumulado)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda1_curva_lorenz.png", dpi=150)
    plt.close(fig)

    # Figura: fallas diarias en el tiempo, con eventos exógenos y picos.
    ft = tel[tel["Connection_Status"] == 1].copy()
    ft["dia"] = ft["fecha"].dt.date
    diarias = ft.groupby("dia").size()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(diarias.index, diarias.values, color=C.OI["celeste"], lw=0.8, label="Fallas diarias")
    ax.plot(diarias.index, diarias.rolling(7).mean().values, color=C.OI["azul"], lw=2, label="Media móvil 7 días")
    for i, (ev, (fi, ff)) in enumerate(C.EVENTOS_EXOGENOS.items()):
        ax.axvspan(pd.to_datetime(fi), pd.to_datetime(ff), alpha=0.25, color=C.OI["amarillo"],
                   label="Eventos exógenos" if i == 0 else None)
    umbral = diarias.mean() + 3 * diarias.std()
    picos = diarias[diarias > umbral]
    if len(picos) > 0:
        ax.scatter(picos.index, picos.values, color=C.OI["naranja"], marker="^", s=55, zorder=5,
                   edgecolor=C.OI["negro"], label=f"Picos (> {umbral:.0f})".replace(".", ","))
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nº de fallas")
    ax.legend(frameon=False)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda1_fallas_tiempo.png", dpi=150)
    plt.close(fig)
    return s


# --- EDA 2: cuartiles de fallas y valores atípicos -------------------------
def eda2_cuartiles(dfm, res):
    f = dfm[dfm["n_fallas"] > 0].copy()
    q1, q2, q3 = f["n_fallas"].quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    lim_sup = q3 + 1.5 * iqr
    f["cuartil"] = pd.qcut(f["n_fallas"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    n_out = int((f["n_fallas"] > lim_sup).sum())
    res["cuartiles_fallas"] = {
        "q1": round(float(q1), 1), "q2_mediana": round(float(q2), 1), "q3": round(float(q3), 1),
        "iqr": round(float(iqr), 1), "limite_outliers": round(float(lim_sup), 1),
        "n_outliers": n_out, "pct_outliers": round(100 * n_out / len(f), 1),
    }

    # Figura: histograma con cuartiles marcados + conteo por cuartil. Las líneas
    # se distinguen por color Okabe-Ito y por estilo (continua/discontinua).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(f["n_fallas"], bins=50, color=C.OI["celeste"], edgecolor=C.OI["negro"])
    for v, lab, ls, col in [(q1, "Q1", "--", C.OI["azul"]),
                            (q2, "Q2 (mediana)", "-", C.OI["naranja"]),
                            (q3, "Q3", "--", C.OI["azul"]),
                            (lim_sup, "Límite de atípicos", ":", C.OI["negro"])]:
        axes[0].axvline(v, color=col, ls=ls, lw=2, label=f"{lab} = {v:.0f}".replace(".", ","))
    axes[0].set_xlabel("Nº de fallas")
    axes[0].set_ylabel("Frecuencia (equipos)")
    axes[0].legend(frameon=False)

    cc = f["cuartil"].value_counts().sort_index()
    bars = axes[1].bar(cc.index.astype(str), cc.values, color=C.OI["azul"], edgecolor=C.OI["negro"])
    for b, v in zip(bars, cc.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(cc.values) * 0.01, f"{v}",
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_xlabel("Cuartil de fallas")
    axes[1].set_ylabel("Nº de equipos")
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda2_distribucion_cuartiles.png", dpi=150)
    plt.close(fig)


# --- EDA 3: mortalidad infantil (antigüedad frente a fallas) ----------------
# Corrección pedida en la revisión: en vez del promedio de fallas crudo por
# grupo ---que no controla ni por el número de equipos del grupo ni por el
# tiempo que cada uno lleva operando--- se usa la tasa de fallas por día de
# operación (normaliza la exposición), se reporta el n de cada grupo y se aplica
# un test de Kruskal-Wallis para contrastar si las diferencias son significativas
# (no se asume normalidad, dado lo sesgado de la distribución de fallas).
def eda3_mortalidad_infantil(dfm, res):
    d = dfm[(dfm["edad_anios"].notna()) & (dfm["edad_anios"] > 0) & (dfm["edad_anios"] < 30)].copy()
    bins = [0, 2, 5, 10, 15, 30]
    labels = ["0-2", "2-5", "5-10", "10-15", "15+"]
    d["grupo_edad"] = pd.cut(d["edad_anios"], bins=bins, labels=labels)

    g = d.groupby("grupo_edad", observed=True)
    resumen = g.agg(n_equipos=("Numpos", "count"),
                    tasa_media=("tasa_fallas_dia", "mean"),
                    tasa_mediana=("tasa_fallas_dia", "median")).reset_index()

    # Test de Kruskal-Wallis sobre la tasa entre los grupos de antigüedad.
    muestras = [d.loc[d["grupo_edad"] == l, "tasa_fallas_dia"].dropna().values
                for l in labels if (d["grupo_edad"] == l).any()]
    H, p = stats.kruskal(*muestras)
    res["mortalidad_infantil"] = {
        "n_por_grupo": {str(r.grupo_edad): int(r.n_equipos) for r in resumen.itertuples()},
        "tasa_mediana_por_grupo": {str(r.grupo_edad): round(float(r.tasa_mediana), 4) for r in resumen.itertuples()},
        "kruskal_H": round(float(H), 2),
        "kruskal_p": float(p),
        "diferencia_significativa": bool(p < 0.05),
    }

    # Figura: tasa mediana de fallas por grupo de antigüedad, con el n anotado.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(resumen["grupo_edad"].astype(str), resumen["tasa_mediana"],
                  color=C.OI["azul"], edgecolor=C.OI["negro"])
    for b, n in zip(bars, resumen["n_equipos"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n = {n}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Grupo de antigüedad (años)")
    ax.set_ylabel("Tasa mediana de fallas (fallas por día de operación)")
    sig = "significativa" if p < 0.05 else "no significativa"
    ax.text(0.98, 0.96, f"Kruskal-Wallis: H = {H:.1f}".replace(".", ",") + f"; p = {p:.1e}\nDiferencia {sig}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=C.OI["gris"]))
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda3_fallas_por_grupo_edad.png", dpi=150)
    plt.close(fig)


# --- EDA 4: fallas por categoría (operador, tipo, marca, comuna) ------------
def eda4_fallas_por_categoria(dfm, res):
    cats = {"Operador": "Operador", "Tipo de Equipo": "Tipo de Equipo",
            "Marca Modem": "Marca Modem", "Comuna": "Comuna"}
    resumenes = {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for idx, (nombre, col) in enumerate(cats.items()):
        if col not in dfm.columns:
            continue
        dc = dfm[dfm[col].notna() & (dfm[col] != "SIN_DATO")]
        r = dc.groupby(col).agg(n_equipos=("Numpos", "count"),
                                fallas_total=("n_fallas", "sum"),
                                fallas_media=("n_fallas", "mean")).sort_values("fallas_total", ascending=False)
        resumenes[nombre] = r
        n = len(r)
        bars = axes[idx].barh(range(n), r["fallas_total"], color=C.OI["azul"], edgecolor=C.OI["negro"])
        axes[idx].set_yticks(range(n))
        axes[idx].set_yticklabels(r.index)
        axes[idx].set_xlabel("Total de fallas")
        axes[idx].set_title(f"Fallas por {nombre} (n = {n})")
        axes[idx].invert_yaxis()
        for i, v in enumerate(r["fallas_total"]):
            axes[idx].text(v + r["fallas_total"].max() * 0.01, i, f"{v:,.0f}".replace(",", "."),
                           va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda4_fallas_por_categoria.png", dpi=150)
    plt.close(fig)
    res["fallas_por_categoria_top"] = {
        nombre: {str(k): int(v) for k, v in resumenes[nombre]["fallas_total"].head(8).items()}
        for nombre in resumenes
    }

    # Heatmap marca de módem x operador (promedio de fallas). Escala secuencial
    # azul (segura para daltonismo) con los valores anotados.
    dx = dfm[(dfm["Operador"].notna()) & (dfm["Marca Modem"].notna()) &
             (dfm["Operador"] != "SIN_DATO") & (dfm["Marca Modem"] != "SIN_DATO")]
    pivot = pd.pivot_table(dx, values="n_fallas", index="Marca Modem", columns="Operador",
                           aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Promedio de fallas"})
    ax.set_xlabel("Operador")
    ax.set_ylabel("Marca de módem")
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda4_heatmap_marca_operador.png", dpi=150)
    plt.close(fig)


# --- EDA 5: tiempo entre fallas (TTF) por categoría -------------------------
def eda5_ttf_por_categoria(dfm, res):
    dttf = dfm[(dfm["ttf_median_h"].notna()) & (dfm["ttf_median_h"] > 0)]
    cats = {"Operador": "Operador", "Tipo de Equipo": "Tipo de Equipo",
            "Marca Modem": "Marca Modem", "Comuna": "Comuna"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (nombre, col) in enumerate(cats.items()):
        if col not in dttf.columns:
            continue
        dc = dttf[(dttf[col].notna()) & (dttf[col] != "SIN_DATO")]
        orden = dc.groupby(col)["ttf_median_h"].median().sort_values().index
        if len(orden) > 10:
            orden = orden[:10]
        data = [dc[dc[col] == cat]["ttf_median_h"].values for cat in orden]
        bp = axes[idx].boxplot(data, labels=list(orden), patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(C.OI["celeste"])
        for med in bp["medians"]:
            med.set_color(C.OI["negro"])
        axes[idx].set_xlabel(nombre)
        axes[idx].set_ylabel("TTF (horas)")
        axes[idx].set_title(f"Distribución de TTF por {nombre} (menor = más crítico)")
        axes[idx].set_yscale("log")
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda5_ttf_por_categoria.png", dpi=150)
    plt.close(fig)


# --- EDA 6: agrupamiento de equipos por similitud ---------------------------
# Agrupa los equipos según su comportamiento (fallas, tasa, TTF, antigüedad de
# operación y atributos) mediante similitud coseno y clustering jerárquico de
# Ward. El objetivo operativo (respuesta a la revisión, #19) es identificar el
# grupo de equipos crónicamente problemáticos: candidatos a reemplazo o
# mantenimiento correctivo generalizado, a priorizar en el plan anual y a
# revisar de forma inmediata.
def eda6_clustering(dfm, res, n_sample=500):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    feats = ["n_fallas", "tasa_fallas_dia", "ttf_median_h", "dias_operacion"]
    d = dfm.copy()
    for col, pref in [("Operador", "op"), ("Tipo de Equipo", "tipo"), ("Marca Modem", "marca")]:
        if col in d.columns:
            du = pd.get_dummies(d[col], prefix=pref)
            d = pd.concat([d, du], axis=1)
            feats += du.columns.tolist()
    d = d.dropna(subset=["n_fallas", "ttf_median_h"])

    # Muestreo estratificado: se asegura incluir a los 100 equipos con más fallas.
    if len(d) > n_sample:
        top = d.nlargest(100, "n_fallas")
        rest = d.drop(top.index).sample(n=n_sample - 100, random_state=C.RS)
        d = pd.concat([top, rest])

    X = StandardScaler().fit_transform(d[feats].fillna(0).values)
    sim = cosine_similarity(X)
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    n_clusters = 5
    d["cluster"] = fcluster(Z, n_clusters, criterion="maxclust")

    rc = d.groupby("cluster").agg(
        n_equipos=("Numpos", "count"), fallas_mean=("n_fallas", "mean"),
        fallas_median=("n_fallas", "median"), fallas_total=("n_fallas", "sum"),
        ttf_median=("ttf_median_h", "median"), tasa_fallas=("tasa_fallas_dia", "mean")).round(2)
    critico = int(rc["fallas_mean"].idxmax())  # grupo crítico = mayor nivel medio de fallas
    res["clustering"] = {
        "n_clusters": n_clusters,
        "n_equipos_analizados": int(len(d)),
        "cluster_critico": critico,
        "resumen": {int(c): {"n_equipos": int(r.n_equipos),
                             "fallas_media": float(r.fallas_mean),
                             "ttf_mediana_h": float(r.ttf_median),
                             "tasa_fallas_dia": float(r.tasa_fallas)} for c, r in rc.iterrows()},
    }

    # Figura: caracterización de los clusters. Colores Okabe-Ito consistentes y,
    # en el scatter, un marcador distinto por cluster (no solo color).
    cl_col = [C.OI["azul"], C.OI["naranja"], C.OI["verde"], C.OI["rosa"], C.OI["celeste"]]
    cl_mk = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(rc.index.astype(str), rc["n_equipos"], color=cl_col[:len(rc)], edgecolor=C.OI["negro"])
    axes[0, 0].set_xlabel("Cluster"); axes[0, 0].set_ylabel("Nº de equipos"); axes[0, 0].set_title("Equipos por cluster")
    axes[0, 1].bar(rc.index.astype(str), rc["fallas_mean"], color=cl_col[:len(rc)], edgecolor=C.OI["negro"])
    axes[0, 1].set_xlabel("Cluster"); axes[0, 1].set_ylabel("Fallas promedio"); axes[0, 1].set_title("Fallas promedio por cluster")
    axes[1, 0].bar(rc.index.astype(str), rc["ttf_median"], color=cl_col[:len(rc)], edgecolor=C.OI["negro"])
    axes[1, 0].set_xlabel("Cluster"); axes[1, 0].set_ylabel("TTF mediana (horas)"); axes[1, 0].set_title("TTF por cluster")
    for i, c in enumerate(sorted(d["cluster"].unique())):
        dc = d[d["cluster"] == c]
        axes[1, 1].scatter(dc["n_fallas"], dc["ttf_median_h"], label=f"Cluster {c}",
                           color=cl_col[i % len(cl_col)], marker=cl_mk[i % len(cl_mk)],
                           alpha=0.6, s=30, edgecolor=C.OI["negro"], linewidth=0.3)
    axes[1, 1].set_xlabel("Nº de fallas"); axes[1, 1].set_ylabel("TTF mediana (horas)")
    axes[1, 1].set_title("Fallas frente a TTF por cluster")
    axes[1, 1].set_yscale("log"); axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "eda6_clusters_caracteristicas.png", dpi=150)
    plt.close(fig)

    # Asignación de clusters: la consume el análisis geográfico (script 10).
    d[["Numpos", "cluster", "n_fallas", "ttf_median_h"]].to_excel(
        C.OUTPUTS_DIR / "eda6_asignacion_clusters.xlsx", index=False)
    return d


# --- Orquestación -----------------------------------------------------------
if __name__ == "__main__":
    res = {}
    tel, man = cargar()
    print(f"telemetría: {len(tel):,} filas | {tel['Numpos'].nunique()} equipos")
    dfm = metricas_por_equipo(tel, man)
    res["n_equipos"] = int(len(dfm))
    res["equipos_con_alguna_falla"] = int((dfm["n_fallas"] > 0).sum())
    print(f"métricas por equipo: {len(dfm):,} equipos")

    eda1_concentracion(dfm, tel, res)
    print("eda1:", res.get("concentracion"))

    eda2_cuartiles(dfm, res)
    print("eda2:", res.get("cuartiles_fallas"))

    eda3_mortalidad_infantil(dfm, res)
    print("eda3:", res.get("mortalidad_infantil"))

    eda4_fallas_por_categoria(dfm, res)
    print("eda4: figuras de fallas por categoría y heatmap generadas")

    eda5_ttf_por_categoria(dfm, res)
    print("eda5: figura de TTF por categoría generada")

    eda6_clustering(dfm, res)
    print("eda6:", res.get("clustering", {}).get("cluster_critico"))

    (C.OUTPUTS_DIR / "08_eda.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("Listo. Figuras eda1-eda6 y outputs/08_eda.json generados.")
