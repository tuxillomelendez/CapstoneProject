"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Santiago de Chile, 2026.

Script 00 - Ingeniería de características (Feature Engineering).

Transforma la telemetría cruda en el conjunto de modelamiento: filtra las fallas
ocurridas durante mantenimientos (tickets), integra los atributos del mantenedor
y los codifica, calcula el tiempo hasta la falla (TTF), define la censura a siete
días (event y duration) y construye las variables del modelo (rezagos, variables
horarias, razón RX/TX, conteo de desconexiones, tiempo desde la última caída).
El resultado se guarda como dataset_modelamiento.parquet, insumo de los scripts
de modelamiento (01 a 07 y 11).

Este script conserva la lógica del flujo de trabajo original; solo se ajustaron
las rutas para que se resuelvan por variables de entorno y se retiró el estado
propio del entorno interactivo.

Entradas:  TESIS_TELEMETRIA -> parquet de telemetría cruda.
           TESIS_TICKETS    -> csv de tickets de mantenimiento.
           TESIS_MANTENEDOR -> excel del mantenedor de equipos.
Salida:    el parquet de modelamiento (ruta TESIS_DATA, o junto a la telemetría
           con nombre dataset_modelamiento.parquet) y outputs/fe_distribucion_ttf.png.

Uso:
    set TESIS_TELEMETRIA=C:\\ruta\\telemetria_cruda.parquet   (Windows)
    set TESIS_TICKETS=C:\\ruta\\maestro_tickets.csv
    set TESIS_MANTENEDOR=C:\\ruta\\Mantenedor.xlsx
    set TESIS_DATA=C:\\ruta\\dataset_modelamiento.parquet
    python src\\00_feature_engineering.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: la figura se guarda en disco
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import LabelEncoder

# --- Rutas (se leen desde config.py / config_local.py; sin variables de entorno) ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

DATA_PATH = C.TELEMETRIA_CRUDA_PATH   # telemetria cruda: ENTRADA del feature engineering
TICKETS_PATH = C.TICKETS_PATH or Path("__sin_definir__")   # tickets (None -> ruta inexistente)
OUTPUT_PATH = C.DATA_PATH             # parquet de modelamiento: SALIDA (lo leen 01-07 y 11)
OUTPUTS_DIR = C.OUTPUTS_DIR

if DATA_PATH is None:
    raise SystemExit("Falta la ruta de la telemetria cruda. Definela en config_local.py "
                     "(variable TELEMETRIA_PATH_LOCAL).")

# --- Parámetros del proceso -------------------------------------------------
DEVICE_ID_COL = C.ID_COL            # identificador de equipo
TARGET_COL = C.TARGET_COL           # variable objetivo (TTF), en segundos
CENSOR_LIMIT = 7 * 24 * 3600        # límite de censura: 7 días en segundos


# ==============================================================================
# CARGAR DATOS CRUDOS
# ==============================================================================
print("Cargando datos de telemetría...")
print(f"  Archivo: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)

print(f"\n✓ Datos cargados exitosamente")
print(f"  - Registros totales: {len(df):,}")
print(f"  - Columnas: {len(df.columns)}")
print(f"  - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# Veamos cuántos equipos únicos tenemos
n_equipos = df[DEVICE_ID_COL].nunique()
print(f"\n✓ Equipos únicos: {n_equipos:,}")

# Veamos el rango de fechas
fecha_min = df['fecha'].min()
fecha_max = df['fecha'].max()
dias_datos = (fecha_max - fecha_min).days
print(f"✓ Rango temporal: {fecha_min.date()} a {fecha_max.date()} ({dias_datos} días)")

# Veamos la distribución de Connection_Status
print(f"\n✓ Distribución de Connection_Status:")
print(df['Connection_Status'].value_counts())
print(f"\n  - 0 = ONLINE (equipo conectado)")
print(f"  - 1 = OFFLINE (equipo desconectado = FALLA)")


# Veamos cuántos equipos únicos tenemos
n_equipos = df[DEVICE_ID_COL].nunique()
print(f"\n✓ Equipos únicos: {n_equipos:,}")

# Veamos el rango de fechas
fecha_min = df['fecha'].min()
fecha_max = df['fecha'].max()
dias_datos = (fecha_max - fecha_min).days
print(f"✓ Rango temporal: {fecha_min.date()} a {fecha_max.date()} ({dias_datos} días)")

# Veamos la distribución de Connection_Status
print(f"\n✓ Distribución de Connection_Status:")
print(df['Connection_Status'].value_counts())
print(f"\n  - 0 = ONLINE (equipo conectado)")
print(f"  - 1 = OFFLINE (equipo desconectado = FALLA)")


# ==============================================================================
# PASO 1.5: FILTRADO DE TICKETS - CÁLCULO DE POSITIVO NETO (CSV + PERFORMANTE)
# ==============================================================================
print("Paso 1.5: Filtrando registros con tickets (fallas exógenas)...")

if TICKETS_PATH.exists():
    # Cargar tickets (CSV)
    df_tickets = pd.read_csv(
        TICKETS_PATH,
        sep=",",
        encoding="utf-8",  # si falla, prueba "latin1"
        low_memory=False
    )
    print(f"  ✓ Tickets cargados: {len(df_tickets):,}")

    # Limpiar tickets con valores nulos
    df_tickets = df_tickets.dropna(subset=['Numpos', 'Fecha_Inicio', 'Fecha_fin'])
    print(f"  ✓ Tickets válidos (sin nulos): {len(df_tickets):,}")

    # Convertir fechas y Numpos
    df_tickets['Fecha_Inicio'] = pd.to_datetime(df_tickets['Fecha_Inicio'], errors="coerce")
    df_tickets['Fecha_fin'] = pd.to_datetime(df_tickets['Fecha_fin'], errors="coerce")
    df_tickets['Numpos'] = pd.to_numeric(df_tickets['Numpos'], errors="coerce").astype("Int64")

    df_tickets = df_tickets.dropna(subset=['Numpos', 'Fecha_Inicio', 'Fecha_fin'])
    df_tickets['Numpos'] = df_tickets['Numpos'].astype(int)

    # Corregir tickets invertidos
    bad = df_tickets['Fecha_Inicio'] > df_tickets['Fecha_fin']
    if bad.any():
        print(f"  ⚠ Tickets con Fecha_Inicio > Fecha_fin: {bad.sum():,} (se corrigen)")
        df_tickets.loc[bad, ['Fecha_Inicio', 'Fecha_fin']] = df_tickets.loc[bad, ['Fecha_fin', 'Fecha_Inicio']].values

    # Mostrar rango de tickets
    print(f"  ✓ Rango de tickets: {df_tickets['Fecha_Inicio'].min().date()} a {df_tickets['Fecha_fin'].max().date()}")

    # Registros iniciales
    registros_iniciales = len(df)
    print(f"  ✓ Registros totales a evaluar: {registros_iniciales:,}")

    # -------------------------------------------------------------------------
    # 🚀 FILTRADO PERFORMANTE:
    # 1) Fusionar (union) intervalos por Numpos para evitar casos anidados/solapados
    # 2) Para cada Numpos, marcar filas con searchsorted (vectorizado)
    # -------------------------------------------------------------------------
    print("  → Preparando datos para filtrado eficiente...")

    # Tipos en df principal
    df['fecha'] = pd.to_datetime(df['fecha'], errors="coerce")
    df = df.dropna(subset=['fecha'])

    df[DEVICE_ID_COL] = pd.to_numeric(df[DEVICE_ID_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[DEVICE_ID_COL])
    df[DEVICE_ID_COL] = df[DEVICE_ID_COL].astype(int)

    print(f"  ✓ Registros válidos con fecha/Numpos: {len(df):,}")

    # --- 1) Construir intervalos fusionados por Numpos ---
    print("  → Fusionando intervalos de tickets por equipo...")

    tickets_min = df_tickets[['Numpos', 'Fecha_Inicio', 'Fecha_fin']].copy()
    tickets_min.sort_values(['Numpos', 'Fecha_Inicio'], kind='mergesort', inplace=True)

    # dict: numpos -> (starts_np, ends_np) ya fusionados y NO solapados
    intervals_by_numpos = {}

    for numpos, g in tickets_min.groupby('Numpos', sort=False):
        starts = g['Fecha_Inicio'].to_numpy(dtype='datetime64[ns]')
        ends   = g['Fecha_fin'].to_numpy(dtype='datetime64[ns]')

        # Fusionar intervalos solapados / adyacentes
        merged_starts = []
        merged_ends = []

        cur_s = starts[0]
        cur_e = ends[0]

        for s, e in zip(starts[1:], ends[1:]):
            if s <= cur_e:  # solapa (si quieres considerar adyacente, usa: if s <= cur_e + np.timedelta64(0,'ns')
                if e > cur_e:
                    cur_e = e
            else:
                merged_starts.append(cur_s)
                merged_ends.append(cur_e)
                cur_s, cur_e = s, e

        merged_starts.append(cur_s)
        merged_ends.append(cur_e)

        intervals_by_numpos[numpos] = (
            np.array(merged_starts, dtype='datetime64[ns]'),
            np.array(merged_ends, dtype='datetime64[ns]')
        )

    print(f"  ✓ Equipos con tickets: {len(intervals_by_numpos):,}")

    # --- 2) Marcar registros con tickets usando searchsorted por Numpos ---
    print("  → Marcando registros con tickets (vectorizado por equipo)...")

    # DataFrame mínimo para procesar rápido + mapear al df original
    df_min = df[[DEVICE_ID_COL, 'fecha']].copy()
    df_min['_row_id'] = np.arange(len(df_min), dtype=np.int64)

    # Ordenar por Numpos y fecha para recorrer por tramos (rápido)
    df_min.sort_values([DEVICE_ID_COL, 'fecha'], kind='mergesort', inplace=True, ignore_index=True)

    numpos_arr = df_min[DEVICE_ID_COL].to_numpy(dtype=np.int64)
    fecha_arr = df_min['fecha'].to_numpy(dtype='datetime64[ns]')
    rowid_arr = df_min['_row_id'].to_numpy(dtype=np.int64)

    # Máscara final (orden original)
    mask_tickets = np.zeros(len(df), dtype=bool)

    # Encontrar cortes de Numpos (run-length)
    cuts = np.flatnonzero(numpos_arr[1:] != numpos_arr[:-1]) + 1
    starts_idx = np.r_[0, cuts]
    ends_idx = np.r_[cuts, len(df_min)]

    # Recorrer por equipo (loop por equipos, NO por 100M filas ni por tickets)
    for a, b in zip(starts_idx, ends_idx):
        n = int(numpos_arr[a])
        intervals = intervals_by_numpos.get(n)
        if intervals is None:
            continue

        s_arr, e_arr = intervals
        f = fecha_arr[a:b]
        rid = rowid_arr[a:b]

        # Buscar el último intervalo cuyo start <= fecha
        idx = np.searchsorted(s_arr, f, side='right') - 1
        valid = (idx >= 0)

        if valid.any():
            idxv = idx[valid]
            fv = f[valid]
            # dentro del intervalo si fecha <= end del intervalo encontrado
            inside = fv <= e_arr[idxv]
            if inside.any():
                mask_tickets[rid[valid][inside]] = True

    # Contar registros con ticket
    registros_con_ticket = int(mask_tickets.sum())
    porcentaje_tickets = 100 * registros_con_ticket / len(df)

    print(f"\n  === RESULTADOS DEL FILTRADO ===")
    print(f"  - Registros TOTALES: {len(df):,}")
    print(f"  - Registros CON ticket (exógenos): {registros_con_ticket:,} ({porcentaje_tickets:.2f}%)")

    # Filtrar para mantener solo POSITIVO NETO
    df = df.loc[~mask_tickets].copy()

    print(f"  - Registros SIN ticket (POSITIVO NETO): {len(df):,} ({100-porcentaje_tickets:.2f}%)")
    print(f"\n  ✓ POSITIVO NETO calculado exitosamente")
    print(f"    → Solo se usarán fallas endógenas para el modelo")

else:
    print(f"  ⚠ No se encontró archivo de tickets: {TICKETS_PATH}")
    print(f"    → Continuando sin filtrar (se usarán TODAS las fallas)")
    print(f"    → Registros: {len(df):,}")


# ==============================================================================
# PASO 1.6: INTEGRACIÓN DE VARIABLES DEL MANTENEDOR
# ==============================================================================
print("\n" + "="*70)
print("PASO 1.6: INTEGRACIÓN DE VARIABLES DEL MANTENEDOR")
print("="*70)

# Ruta al archivo del Mantenedor
MANTENEDOR_PATH = C.MANTENEDOR_PATH or Path("__sin_definir__")

if MANTENEDOR_PATH.exists():
    print(f"\n✓ Cargando Mantenedor: {MANTENEDOR_PATH.name}")
    
    # Cargar mantenedor (header en fila 2)
    df_mantenedor = pd.read_excel(MANTENEDOR_PATH, header=1)
    
    # Limpiar Numpos
    df_mantenedor['Numpos'] = pd.to_numeric(df_mantenedor['Numpos'], errors='coerce')
    df_mantenedor = df_mantenedor.dropna(subset=['Numpos'])
    df_mantenedor['Numpos'] = df_mantenedor['Numpos'].astype(int)
    
    print(f"  ✓ Mantenedor cargado: {len(df_mantenedor):,} equipos")
    
    # === EXPLORAR VARIABLES CATEGÓRICAS ===
    print("\n  Variables categóricas disponibles:")
    COLS_CATEGORICAS = ['Operador', 'Tipo de Equipo', 'Marca Modem', 'Modelo Modem', 'Comuna', 'Antena']
    
    for col in COLS_CATEGORICAS:
        if col in df_mantenedor.columns:
            n_unique = df_mantenedor[col].nunique()
            print(f"    - {col}: {n_unique} valores únicos")
else:
    print(f"⚠ No se encontró el archivo: {MANTENEDOR_PATH}")
    df_mantenedor = None


# ==============================================================================
# PREPARAR Y MERGE CON DATOS DE TELEMETRÍA
# ==============================================================================
if df_mantenedor is not None:
    print("\nPreparando datos del Mantenedor...")
    
    # Mapeo de columnas
    COLS_MANTENEDOR = {
        'Numpos': 'Numpos',
        'Operador': 'Operador',
        'Tipo de Equipo': 'Tipo_Equipo',
        'Marca Modem': 'Marca_Modem',
        'Modelo Modem': 'Modelo_Modem',
        'Comuna': 'Comuna',
        'Antena': 'Antena',
        'Fecha PS Telecontrol': 'Fecha_Instalacion'
    }
    
    # Seleccionar columnas disponibles
    cols_disponibles = [c for c in COLS_MANTENEDOR.keys() if c in df_mantenedor.columns]
    df_mant_clean = df_mantenedor[cols_disponibles].copy()
    
    # Renombrar
    rename_dict = {k: v for k, v in COLS_MANTENEDOR.items() if k in cols_disponibles}
    df_mant_clean = df_mant_clean.rename(columns=rename_dict)
    
    # === CALCULAR EDAD DEL EQUIPO ===
    def parse_fecha_instalacion(fecha_str):
        if pd.isna(fecha_str) or str(fecha_str) in ['00-00-0000', 'NaN', '', 'nan']:
            return pd.NaT
        try:
            return pd.to_datetime(fecha_str, format='%d-%m-%Y', errors='coerce')
        except:
            return pd.to_datetime(fecha_str, errors='coerce')
    
    if 'Fecha_Instalacion' in df_mant_clean.columns:
        df_mant_clean['Fecha_Instalacion_parsed'] = df_mant_clean['Fecha_Instalacion'].apply(parse_fecha_instalacion)
        FECHA_REFERENCIA = df['fecha'].max()
        df_mant_clean['Edad_Equipo_Dias'] = (FECHA_REFERENCIA - df_mant_clean['Fecha_Instalacion_parsed']).dt.days
        df_mant_clean['Edad_Equipo_Dias'] = df_mant_clean['Edad_Equipo_Dias'].clip(lower=0).fillna(0).astype(int)
        df_mant_clean = df_mant_clean.drop(columns=['Fecha_Instalacion', 'Fecha_Instalacion_parsed'], errors='ignore')
        print(f"  ✓ Edad de equipos calculada (ref: {FECHA_REFERENCIA.date()})")
    else:
        df_mant_clean['Edad_Equipo_Dias'] = 0
    
    # === MERGE ===
    df_mant_clean = df_mant_clean.drop_duplicates(subset=['Numpos'])
    n_antes = len(df)
    df = df.merge(df_mant_clean, on='Numpos', how='left')
    
    print(f"  ✓ Merge completado: {n_antes:,} registros")
    
    # Verificar cobertura
    print("\n  Cobertura de variables:")
    for col in df_mant_clean.columns:
        if col != 'Numpos' and col in df.columns:
            cobertura = df[col].notna().mean() * 100
            print(f"    - {col}: {cobertura:.1f}%")


# ==============================================================================
# ENCODING DE VARIABLES CATEGÓRICAS
# ==============================================================================
if df_mantenedor is not None:
    print("\nAplicando Label Encoding...")
    
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    # Variables a encodear
    CATEGORICAS = ['Operador', 'Tipo_Equipo', 'Marca_Modem', 'Modelo_Modem', 'Comuna', 'Antena']
    CATEGORICAS = [c for c in CATEGORICAS if c in df.columns]
    
    encoders_mantenedor = {}
    
    for col in CATEGORICAS:
        # Rellenar NaN
        df[col] = df[col].fillna('DESCONOCIDO').astype(str)
        
        # Crear encoder
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders_mantenedor[col] = le
        
        n_clases = df[f'{col}_encoded'].nunique()
        print(f"  ✓ {col}_encoded: {n_clases} clases")
    
    # Guardar encoders
    ENCODERS_DIR = OUTPUTS_DIR / "encoders"
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    
    for col, encoder in encoders_mantenedor.items():
        joblib.dump(encoder, ENCODERS_DIR / f"{col}_encoder.joblib")
    
    # Guardar lista de categóricas
    with open(ENCODERS_DIR / "categoricas_list.txt", 'w') as f:
        for col in CATEGORICAS:
            f.write(col + '\n')
    
    print(f"\n  ✓ Encoders guardados en: {ENCODERS_DIR}")
    
    # Eliminar columnas string (solo mantener encoded)
    df = df.drop(columns=CATEGORICAS, errors='ignore')
    
    # === RESUMEN ===
    print("\n" + "-"*50)
    print("NUEVAS FEATURES DEL MANTENEDOR:")
    print("-"*50)
    nuevas_features = [f'{c}_encoded' for c in CATEGORICAS] + ['Edad_Equipo_Dias']
    for feat in nuevas_features:
        if feat in df.columns:
            print(f"  ✓ {feat}: min={df[feat].min()}, max={df[feat].max()}")
    print("-"*50)


# ==============================================================================
# PASO 2.1: PREPARACIÓN - Eliminar registros sin fecha
# ==============================================================================
# Los registros sin fecha no pueden calcular TTF, así que los eliminamos
print("Paso 2.1: Limpieza de fechas nulas...")

registros_iniciales = len(df)
df = df.dropna(subset=['fecha'])
registros_eliminados = registros_iniciales - len(df)

if registros_eliminados > 0:
    print(f"  ⚠ Eliminados {registros_eliminados:,} registros sin fecha")
else:
    print(f"  ✓ No hay registros con fecha nula")

print(f"  - Registros restantes: {len(df):,}")


# ==============================================================================
# PASO 2.2: Ordenar datos por equipo y fecha (ASCENDENTE)
# ==============================================================================
# El orden es CRÍTICO para el cálculo correcto del TTF
# Ordenamos primero por Numpos (equipo) y luego por fecha
print("Paso 2.2: Ordenando datos por equipo y fecha...")

df = df.sort_values(by=[DEVICE_ID_COL, 'fecha'], ascending=[True, True])
df = df.reset_index(drop=True)

print(f"  ✓ Datos ordenados")
print(f"  - Primer registro: {df['fecha'].iloc[0]}")
print(f"  - Último registro: {df['fecha'].iloc[-1]}")

# Mostramos ejemplo de un equipo
equipo_ejemplo = df[DEVICE_ID_COL].iloc[0]
print(f"\n  Ejemplo - Equipo {equipo_ejemplo} (primeros 5 registros):")
print(df[df[DEVICE_ID_COL] == equipo_ejemplo][['fecha', 'Connection_Status']].head())


# ==============================================================================
# PASO 2.3: Crear columna auxiliar para marcar eventos OFFLINE
# ==============================================================================
# Creamos una columna 'helper_time' que solo tiene valor cuando hay una falla
# Los registros ONLINE tendrán NaT (Not a Time)
print("Paso 2.3: Marcando eventos de falla (OFFLINE)...")

# Inicializar columna auxiliar con NaT (valor nulo de fecha)
df['helper_time'] = pd.NaT

# Identificar registros OFFLINE (Connection_Status == 1)
# Estos son los eventos de FALLA
offline_mask = df['Connection_Status'] == 1
n_fallas = offline_mask.sum()

# Copiar la fecha solo en los registros de falla
df.loc[offline_mask, 'helper_time'] = df.loc[offline_mask, 'fecha']

print(f"  ✓ Eventos de falla marcados")
print(f"  - Total fallas en dataset: {n_fallas:,}")
print(f"  - Tasa de falla: {100*n_fallas/len(df):.2f}%")

# Veamos un ejemplo
print(f"\n  Ejemplo de helper_time (primeros 10 registros con falla):")
print(df[offline_mask][['fecha', 'Connection_Status', 'helper_time']].head(10))


# ==============================================================================
# PASO 2.4: Ordenar DESCENDENTE para propagar fallas hacia atrás
# ==============================================================================
# Este es el TRUCO clave del algoritmo:
# Al ordenar en orden DESCENDENTE y usar forward-fill, propagamos la fecha de
# la próxima falla hacia todos los registros anteriores.
print("Paso 2.4: Ordenando descendente para propagación...")

df = df.sort_values(by=[DEVICE_ID_COL, 'fecha'], ascending=[True, False])

print(f"  ✓ Datos reordenados en forma descendente")
print(f"  - Ahora el registro más reciente está primero (por equipo)")


# ==============================================================================
# PASO 2.5: Forward-fill para propagar próxima falla
# ==============================================================================
# El forward-fill (ffill) copia el valor hacia adelante hasta encontrar otro valor.
# Como ordenamos de forma descendente, esto propaga la fecha de falla "hacia atrás" 
# en el tiempo (hacia los registros más antiguos).
print("Paso 2.5: Propagando fecha de próxima falla (forward-fill)...")

df['next_offline_time'] = df.groupby(DEVICE_ID_COL)['helper_time'].ffill()

# Contar cuántos registros tienen próxima falla asignada
registros_con_ttf = df['next_offline_time'].notna().sum()
print(f"  ✓ Propagación completada")
print(f"  - Registros con próxima falla asignada: {registros_con_ttf:,} ({100*registros_con_ttf/len(df):.1f}%)")
print(f"  - Registros sin próxima falla (censurados): {len(df) - registros_con_ttf:,}")


# ==============================================================================
# PASO 2.6: Re-ordenar ASCENDENTE para calcular diferencia de tiempo
# ==============================================================================
print("Paso 2.6: Re-ordenando ascendente para cálculo final...")

df = df.sort_values(by=[DEVICE_ID_COL, 'fecha'], ascending=[True, True])

print(f"  ✓ Datos reordenados cronológicamente")


# ==============================================================================
# PASO 2.7: Calcular Time-to-Failure (diferencia en segundos)
# ==============================================================================
# TTF = fecha_proxima_falla - fecha_actual
# El resultado está en segundos
print("Paso 2.7: Calculando Time-to-Failure...")

df[TARGET_COL] = (df['next_offline_time'] - df['fecha']).dt.total_seconds()

# Estadísticas del TTF calculado
ttf_stats = df[TARGET_COL].describe()
print(f"  ✓ Time-to-Failure calculado")
print(f"\n  Estadísticas de TTF (en segundos):")
print(f"  - Media:   {ttf_stats['mean']:,.0f} seg ({ttf_stats['mean']/3600:.1f} horas)")
print(f"  - Mediana: {ttf_stats['50%']:,.0f} seg ({ttf_stats['50%']/3600:.1f} horas)")
print(f"  - Mínimo:  {ttf_stats['min']:,.0f} seg")
print(f"  - Máximo:  {ttf_stats['max']:,.0f} seg ({ttf_stats['max']/3600/24:.1f} días)")
print(f"  - Valores NaN: {df[TARGET_COL].isna().sum():,}")


# ==============================================================================
# PASO 2.8: Limpiar columnas auxiliares
# ==============================================================================
print("Paso 2.8: Eliminando columnas auxiliares...")

df = df.drop(columns=['helper_time', 'next_offline_time'])

print(f"  ✓ Columnas auxiliares eliminadas")
print(f"  - Columnas restantes: {len(df.columns)}")


OUTPUT_DIR = OUTPUTS_DIR


# ==============================================================================
# DISTRIBUCIÓN DE TTF - CORREGIDO
# ==============================================================================
# Cambios:
# 1. Formato de eje Y con separador de miles en lugar de notación científica
# 2. Mejor legibilidad de números grandes
# ==============================================================================

from matplotlib.ticker import FuncFormatter

# Función para formatear eje Y con separador de miles
def formato_miles(x, pos):
    """Formatea números grandes con K (miles) o M (millones)"""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'

# Visualización del TTF calculado
print("\n" + "="*60)
print("VISUALIZACIÓN: Distribución de Time-to-Failure")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Histograma de TTF en horas
ttf_horas = df[TARGET_COL].dropna() / 3600
ax1 = axes[0]
ax1.hist(ttf_horas[ttf_horas < 168], bins=50, color='#0072B2', edgecolor='white', alpha=0.8)
ax1.axvline(ttf_horas.median(), color='#D55E00', linestyle='--', linewidth=2, label=f'Mediana: {ttf_horas.median():.1f}h')
ax1.set_xlabel('Time-to-Failure (horas)', fontsize=11)
ax1.set_ylabel('Frecuencia', fontsize=11)
ax1.set_title('Distribución de TTF (< 7 días)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.yaxis.set_major_formatter(FuncFormatter(formato_miles))  # CAMBIO: formato legible
ax1.grid(True, alpha=0.3, axis='y')

# Gráfico 2: TTF en escala logarítmica
ax2 = axes[1]
ax2.hist(np.log1p(ttf_horas), bins=50, color='#009E73', edgecolor='white', alpha=0.8)
ax2.set_xlabel('log(1 + TTF en horas)', fontsize=11)
ax2.set_ylabel('Frecuencia', fontsize=11)
ax2.set_title('Distribución de log(TTF)', fontsize=12, fontweight='bold')
ax2.yaxis.set_major_formatter(FuncFormatter(formato_miles))  # CAMBIO: formato legible
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fe_distribucion_ttf.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n✓ Gráfico guardado: fe_distribucion_ttf.png")
print("  ✓ La distribución es asimétrica, por eso usaremos log-transform en el modelado")


# ==============================================================================
# PASO 3.1: Crear columna 'event' (indicador de falla observada)
# ==============================================================================
# Un evento está "observado" si el TTF es menor o igual al límite de censura
# Si el TTF es mayor a 7 días, consideramos que el dato está "censurado"
print("Paso 3.1: Creando columna 'event' (indicador de falla)...")

df['event'] = (df[TARGET_COL] <= CENSOR_LIMIT).astype(int)

n_eventos = df['event'].sum()
n_censurados = len(df) - n_eventos
print(f"  ✓ Columna 'event' creada")
print(f"  - Eventos observados (TTF ≤ 7 días): {n_eventos:,} ({100*n_eventos/len(df):.1f}%)")
print(f"  - Datos censurados (TTF > 7 días): {n_censurados:,} ({100*n_censurados/len(df):.1f}%)")


# ==============================================================================
# PASO 3.2: Crear columna 'duration' (duración censurada)
# ==============================================================================
# La duración se limita al máximo de 7 días para análisis de supervivencia
print("Paso 3.2: Creando columna 'duration' (duración censurada)...")

df['duration'] = np.clip(df[TARGET_COL], 0, CENSOR_LIMIT)

print(f"  ✓ Columna 'duration' creada")
print(f"  - Mínimo: {df['duration'].min():,.0f} seg")
print(f"  - Máximo: {df['duration'].max():,.0f} seg ({df['duration'].max()/3600:.0f} horas)")


# ==============================================================================
# PASO 3.3: Rellenar valores NaN del TTF
# ==============================================================================
# Los registros sin próxima falla (censurados al final de la observación)
# se rellenan con la mediana del TTF para poder entrenar el modelo de regresión
print("Paso 3.3: Rellenando valores NaN del TTF...")

n_nan_antes = df[TARGET_COL].isna().sum()
ttf_mediana = df[TARGET_COL].median()

df[TARGET_COL] = df[TARGET_COL].fillna(ttf_mediana)
df[TARGET_COL] = df[TARGET_COL].clip(lower=1)  # Mínimo 1 segundo para evitar log(0)

print(f"  ✓ Valores NaN rellenados con mediana")
print(f"  - NaN antes: {n_nan_antes:,}")
print(f"  - Mediana usada: {ttf_mediana:,.0f} seg ({ttf_mediana/3600:.1f} horas)")

# Lo mismo para duration
duration_mediana = df['duration'].median()
df['duration'] = df['duration'].fillna(duration_mediana)
df['duration'] = df['duration'].clip(lower=1)

print(f"  - Duration NaN rellenados con: {duration_mediana:,.0f} seg")


# ==============================================================================
# PASO 4.1: Features de LAG (valores anteriores del TTF)
# ==============================================================================
# Los valores pasados del TTF ayudan a capturar tendencias
# Si el TTF viene disminuyendo, es señal de que una falla se aproxima
print("Paso 4.1: Creando features de LAG (valores anteriores)...")

# LAG 1: TTF del registro anterior (del mismo equipo)
df[f'{TARGET_COL}_lag1'] = df.groupby(DEVICE_ID_COL)[TARGET_COL].shift(1)
# Rellenar NaN al inicio de cada equipo con el valor actual
df[f'{TARGET_COL}_lag1'] = df[f'{TARGET_COL}_lag1'].bfill()

# LAG 2: TTF de hace 2 registros
df[f'{TARGET_COL}_lag2'] = df.groupby(DEVICE_ID_COL)[TARGET_COL].shift(2)
df[f'{TARGET_COL}_lag2'] = df[f'{TARGET_COL}_lag2'].bfill()

print(f"  ✓ Features de LAG creadas")
print(f"  - {TARGET_COL}_lag1: TTF del registro anterior")
print(f"  - {TARGET_COL}_lag2: TTF de hace 2 registros")


# ==============================================================================
# PASO 4.2: Día de la semana y fin de semana
# ==============================================================================
# Los patrones de falla pueden variar según el día
# Por ejemplo: menos monitoreo los fines de semana
print("Paso 4.2: Creando features de día de semana...")

df['day_of_week'] = df['fecha'].dt.dayofweek  # 0=Lunes, 6=Domingo

# Indicador binario de fin de semana
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print(f"  ✓ Features de día creadas")
print(f"  - Distribución por día:")
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
for i, dia in enumerate(dias):
    count = (df['day_of_week'] == i).sum()
    pct = 100 * count / len(df)
    print(f"    {dia}: {count:,} ({pct:.1f}%)")


# ==============================================================================
# PASO 4.3: Hora del día (encoding cíclico)
# ==============================================================================
# La hora del día tiene un patrón cíclico: la hora 23 está cerca de la hora 0
# Por eso usamos transformación seno/coseno en lugar de one-hot encoding
# Esto preserva la continuidad cíclica
print("Paso 4.3: Creando features de hora (encoding cíclico)...")

df['hour'] = df['fecha'].dt.hour

# Transformación cíclica usando seno y coseno
# seno y coseno de 2π * hora / 24
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

print(f"  ✓ Features de hora creadas")
print(f"  - hour_sin: Componente seno de la hora")
print(f"  - hour_cos: Componente coseno de la hora")
print(f"  - Ejemplo: hora 0 → (sin=0, cos=1), hora 6 → (sin=1, cos=0), hora 12 → (sin=0, cos=-1)")

# Visualización del encoding cíclico
fig, ax = plt.subplots(figsize=(8, 8))
horas = np.arange(24)
x = np.cos(2 * np.pi * horas / 24)
y = np.sin(2 * np.pi * horas / 24)
ax.scatter(x, y, c=horas, cmap='hsv', s=200, edgecolors='black')
for i, h in enumerate(horas):
    ax.annotate(str(h), (x[i], y[i]), ha='center', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('cos(2π × hora / 24)', fontsize=11)
ax.set_ylabel('sin(2π × hora / 24)', fontsize=11)
ax.set_title('Encoding Cíclico de Hora del Día', fontsize=12, fontweight='bold')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_aspect('equal')
print("  ✓ Las horas cercanas (ej: 23 y 0) quedan cerca en el espacio seno-coseno")


# ==============================================================================
# PASO 5.1: Tiempo desde último evento
# ==============================================================================
# Calculamos cuántos segundos han pasado desde el registro anterior del mismo equipo
print("Paso 5.1: Calculando tiempo desde último evento...")

df['last_offline_time'] = df.groupby(DEVICE_ID_COL)['fecha'].diff().dt.total_seconds()
df['last_offline_time'] = df['last_offline_time'].fillna(0)

print(f"  ✓ Feature 'last_offline_time' creada")
print(f"  - Promedio: {df['last_offline_time'].mean():,.0f} segundos ({df['last_offline_time'].mean()/60:.1f} minutos)")
print(f"  - Mediana: {df['last_offline_time'].median():,.0f} segundos")


# ==============================================================================
# PASO 5.2: Promedios móviles de tráfico (RX y TX)
# ==============================================================================
# El promedio móvil suaviza el ruido y muestra tendencias
# Usamos ventana de 4 registros
print("Paso 5.2: Calculando promedios móviles de tráfico...")

# Promedio móvil de RX_bytes (bytes recibidos)
df['RX_moving_avg'] = df.groupby(DEVICE_ID_COL)['RX_bytes'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)

# Promedio móvil de TX_bytes (bytes transmitidos)
df['TX_moving_avg'] = df.groupby(DEVICE_ID_COL)['TX_bytes'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)

print(f"  ✓ Promedios móviles calculados (ventana=4)")
print(f"  - RX_moving_avg: Media={df['RX_moving_avg'].mean():,.0f}")
print(f"  - TX_moving_avg: Media={df['TX_moving_avg'].mean():,.0f}")


# ==============================================================================
# PASO 5.3: Ratio RX/TX
# ==============================================================================
# El ratio entre bytes recibidos y transmitidos puede indicar anomalías
# Un cambio brusco en este ratio puede preceder una falla
print("Paso 5.3: Calculando ratio RX/TX...")

# Sumamos 1 al denominador para evitar división por cero
df['RX_TX_ratio'] = df['RX_bytes'] / (df['TX_bytes'] + 1)

print(f"  ✓ Feature 'RX_TX_ratio' creada")
print(f"  - Media: {df['RX_TX_ratio'].mean():.2f}")
print(f"  - Mediana: {df['RX_TX_ratio'].median():.2f}")


# ==============================================================================
# PASO 5.4: Detección de caídas bruscas en tráfico
# ==============================================================================
# Una caída del tráfico a menos del 50% del promedio móvil puede indicar problemas
print("Paso 5.4: Detectando caídas bruscas de tráfico...")

# RX_drop = 1 si el tráfico RX actual es menor al 50% del promedio móvil
df['RX_drop'] = (df['RX_bytes'] < df['RX_moving_avg'] * 0.5).astype(int)

# TX_drop = 1 si el tráfico TX actual es menor al 50% del promedio móvil
df['TX_drop'] = (df['TX_bytes'] < df['TX_moving_avg'] * 0.5).astype(int)

n_rx_drops = df['RX_drop'].sum()
n_tx_drops = df['TX_drop'].sum()

print(f"  ✓ Features de caídas creadas")
print(f"  - Caídas en RX detectadas: {n_rx_drops:,} ({100*n_rx_drops/len(df):.2f}%)")
print(f"  - Caídas en TX detectadas: {n_tx_drops:,} ({100*n_tx_drops/len(df):.2f}%)")


# ==============================================================================
# PASO 5.5: Patrón de causa de desconexión repetida
# ==============================================================================
# Si la causa de desconexión es la misma que la anterior, puede indicar un problema persistente
print("Paso 5.5: Detectando patrones de causa repetida...")

df['same_cause_last_time'] = (
    df['Cause_of_last_disconnection'] == 
    df.groupby(DEVICE_ID_COL)['Cause_of_last_disconnection'].shift(1)
).astype(int)

n_same_cause = df['same_cause_last_time'].sum()
print(f"  ✓ Feature 'same_cause_last_time' creada")
print(f"  - Registros con misma causa que anterior: {n_same_cause:,} ({100*n_same_cause/len(df):.2f}%)")


# ==============================================================================
# PASO 6.1: Label Encoding de Connection_Status
# ==============================================================================
print("Paso 6.1: Encoding de Connection_Status...")

print(f"  Valores únicos antes: {df['Connection_Status'].unique()}")

le_status = LabelEncoder()
df['Connection_Status'] = le_status.fit_transform(df['Connection_Status'].astype(str))

print(f"  ✓ Connection_Status encoded")
print(f"  - Mapping: {dict(zip(le_status.classes_, le_status.transform(le_status.classes_)))}")


# ==============================================================================
# PASO 6.2: Label Encoding de Cause_of_last_disconnection
# ==============================================================================
print("Paso 6.2: Encoding de Cause_of_last_disconnection...")

n_causas = df['Cause_of_last_disconnection'].nunique()
print(f"  Causas únicas: {n_causas}")

le_cause = LabelEncoder()
df['Cause_of_last_disconnection'] = le_cause.fit_transform(df['Cause_of_last_disconnection'].astype(str))

print(f"  ✓ Cause_of_last_disconnection encoded")


# ==============================================================================
# PASO 6.3: Encoding del identificador de equipo (Numpos)
# ==============================================================================
# Codificamos el Numpos para que el modelo pueda aprender patrones específicos por equipo
print("Paso 6.3: Encoding de Numpos (identificador de equipo)...")

le_numpos = LabelEncoder()
df['Numpos_encoded'] = le_numpos.fit_transform(df[DEVICE_ID_COL])

print(f"  ✓ Numpos_encoded creado")
print(f"  - Equipos codificados: {len(le_numpos.classes_):,}")


# ==============================================================================
# PASO 7.1: Eliminar columnas innecesarias para el modelo
# ==============================================================================
# Algunas columnas intermedias o redundantes deben eliminarse
print("Paso 7.1: Eliminando columnas innecesarias...")

# Columnas a eliminar (intermedias o que causaron problemas en experimentos previos)
columns_to_drop = ['day_of_week', 'hour', 'RX_moving_avg', 'TX_moving_avg']

for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=col)
        print(f"  - Eliminada: {col}")

print(f"  ✓ Limpieza completada")


# ==============================================================================
# PASO 7.2: Verificar el dataset final
# ==============================================================================
print("\n" + "="*60)
print("RESUMEN DEL DATASET PROCESADO")
print("="*60)

print(f"\n✓ Dimensiones finales: {df.shape[0]:,} registros × {df.shape[1]} columnas")
print(f"✓ Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\n✓ Columnas del dataset:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    na_count = df[col].isna().sum()
    print(f"  {i:2}. {col:<35} | {str(dtype):<15} | NaN: {na_count:,}")


# ==============================================================================
# PASO 7.3: Estadísticas de la variable objetivo
# ==============================================================================
print("\n" + "="*60)
print("ESTADÍSTICAS DE LA VARIABLE OBJETIVO (TTF)")
print("="*60)

ttf_horas = df[TARGET_COL] / 3600

print(f"\n  Time-to-Failure en HORAS:")
print(f"  - Media:     {ttf_horas.mean():.2f} horas")
print(f"  - Mediana:   {ttf_horas.median():.2f} horas")
print(f"  - Std:       {ttf_horas.std():.2f} horas")
print(f"  - Mínimo:    {ttf_horas.min():.4f} horas")
print(f"  - Máximo:    {ttf_horas.max():.2f} horas ({ttf_horas.max()/24:.1f} días)")

# Percentiles
print(f"\n  Percentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    val = np.percentile(ttf_horas, p)
    print(f"  - P{p}: {val:.2f} horas")


# ==============================================================================
# EXPORTAR DATASET
# ==============================================================================
print("Paso 8: Exportando dataset procesado...")

# Asegurar que el directorio existe
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Guardar en formato Parquet (eficiente en espacio y rápido de leer)
df.to_parquet(OUTPUT_PATH, index=False)

# Verificar que se guardó correctamente
file_size = OUTPUT_PATH.stat().st_size / 1024**2

print(f"\n✓ Dataset exportado exitosamente")
print(f"  - Ruta: {OUTPUT_PATH}")
print(f"  - Tamaño: {file_size:.1f} MB")
print(f"  - Registros: {len(df):,}")
print(f"  - Columnas: {len(df.columns)}")


# ==============================================================================
# VERIFICACIÓN FINAL
# ==============================================================================
print("\n" + "="*60)
print("VERIFICACIÓN: Cargando archivo guardado")
print("="*60)

df_verify = pd.read_parquet(OUTPUT_PATH)
print(f"  ✓ Archivo leído correctamente")
print(f"  - Shape: {df_verify.shape}")
print(f"  - Columnas: {list(df_verify.columns)}")

# Verificar que TTF tiene valores válidos
assert df_verify[TARGET_COL].isna().sum() == 0, "ERROR: Hay valores NaN en TTF"
assert (df_verify[TARGET_COL] > 0).all(), "ERROR: Hay valores TTF <= 0"
print(f"  ✓ Variable objetivo validada (sin NaN, todos > 0)")

print("\n" + "="*60)
print("✓ FEATURE ENGINEERING COMPLETADO")
print("="*60)
print(f"\nEl archivo '{OUTPUT_PATH.name}' está listo para el notebook de modelamiento.")
