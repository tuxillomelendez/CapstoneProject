"""
Predicción de Fallas en Equipos de Telecontrol usando Análisis de Sobrevivencia
Tesis para optar al grado de Magíster en Ciencia de Datos.

Autor:             Jorge Eduardo Meléndez Bastías
Profesor guía:     Dr. Christian Pieringer Baeza
Profesor co-guía:  Dr. Ronal Manuel Coronado
Profesor revisor:  Francisco Pérez Galarce
Universidad de Las Américas (UDLA), Chile, 2026.

Configuración central del proyecto.

Este archivo concentra las rutas, los nombres de columnas, el conjunto de
variables predictivas y los hiperparámetros del modelo. Todos los scripts de
la carpeta src/ leen su configuración desde aquí, de modo que no es necesario
modificarlos para cambiar rutas o parámetros: basta con editar este archivo.
"""
import os
from pathlib import Path

# --- Rutas base -------------------------------------------------------------
# Este archivo vive en src/; la raiz del repositorio es el directorio padre.
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- Ruta del conjunto de datos ---------------------------------------------
# La ruta a los datos se resuelve en tres niveles, por orden de prioridad:
#
#   1. Variable de entorno TESIS_DATA, si esta definida (forma recomendada).
#   2. Ruta privada declarada en config_local.py, si ese archivo existe.
#   3. Conjunto de juguete anonimizado incluido en el repositorio (por defecto).
#
# El conjunto de datos real es privado y no forma parte del repositorio. Este
# esquema permite reproducir el analisis sobre la data real sin escribir rutas
# privadas en el codigo que se publica: la ruta personal vive en config_local.py,
# que esta excluido del control de versiones mediante .gitignore.
_TOY_POR_DEFECTO = DATA_DIR / "toy_publico_modelamiento.parquet"


def _resolver_ruta_datos():
    """Devuelve la ruta al conjunto de datos segun el orden de prioridad descrito."""
    # Nivel 1: variable de entorno.
    ruta_entorno = os.environ.get("TESIS_DATA")
    if ruta_entorno:
        return Path(ruta_entorno)
    # Nivel 2: configuracion local no versionada (archivo opcional).
    try:
        from config_local import DATA_PATH_LOCAL
        return Path(DATA_PATH_LOCAL)
    except ImportError:
        pass
    # Nivel 3: conjunto de juguete incluido en el repositorio.
    return _TOY_POR_DEFECTO


DATA_PATH = _resolver_ruta_datos()

# --- Rutas de datos crudos (solo para el EDA / caracterizacion) -------------
# El EDA parte de la telemetria cruda y del mantenedor (no del parquet de
# modelamiento), porque necesita columnas en su forma original (marca, operador,
# comuna, edad). Estas rutas se resuelven por variable de entorno; si no estan
# definidas quedan en None, y los scripts de modelamiento no se ven afectados.
def _resolver_cruda(nombre_local, var_entorno):
    """Resuelve una ruta de datos crudos: primero config_local.py, luego la
    variable de entorno. Si no aparece en ninguno, devuelve None (los scripts de
    modelamiento 01-07 y 11 no usan estas rutas, asi que no se ven afectados)."""
    try:
        import config_local
        valor = getattr(config_local, nombre_local, None)
        if valor:
            return Path(valor)
    except ImportError:
        pass
    valor = os.environ.get(var_entorno)
    return Path(valor) if valor else None


TELEMETRIA_CRUDA_PATH = _resolver_cruda("TELEMETRIA_PATH_LOCAL", "TESIS_TELEMETRIA")  # parquet de telemetria cruda
MANTENEDOR_PATH       = _resolver_cruda("MANTENEDOR_PATH_LOCAL", "TESIS_MANTENEDOR")  # excel del mantenedor de equipos
TICKETS_PATH          = _resolver_cruda("TICKETS_PATH_LOCAL", "TESIS_TICKETS")        # csv de tickets de mantenimiento

# Ventanas de eventos exogenos conocidos, para marcarlos y/o excluirlos en el EDA.
EVENTOS_EXOGENOS = {
    "Temporal de vientos (agosto 2024)": ("2024-08-01", "2024-08-15"),
    "Blackout (febrero 2025)":           ("2025-02-04", "2025-02-26"),
    "Caida de sistema de gateways":      ("2025-04-25", "2025-04-25"),
}

# --- Nombres de columnas clave ----------------------------------------------
ID_COL       = "Numpos"            # Identificador de equipo (anonimizado a EQ001.. en el toy).
DURATION_COL = "duration"          # Tiempo hasta el evento, en segundos.
EVENT_COL    = "event"             # 1 = falla observada, 0 = observacion censurada.
TARGET_COL   = "time_to_failure"   # TTF en segundos (se usa solo en el diagnostico de la fuga).
LAG_COLS     = ["time_to_failure_lag1", "time_to_failure_lag2"]  # Variables de rezago: SOLO para
                                                                  # el diagnostico de la fuga, NUNCA
                                                                  # para el modelo final.

# --- Variables predictivas --------------------------------------------------
# Conjunto de variables del modelo CORREGIDO (sin los rezagos, que filtran el objetivo).
FEATS = ["Operador_encoded", "Tipo_Equipo_encoded", "Marca_Modem_encoded", "Comuna_encoded",
         "RX_TX_ratio", "N_of_disconnections", "is_weekend", "hour_sin", "hour_cos", "last_offline_time"]

# Conjunto CON rezagos: reproduce el modelo con fuga. Se usa unicamente para el
# diagnostico y la matriz de ablacion, con fines comparativos.
FEATS_LAG = LAG_COLS + FEATS

# --- Hiperparametros base de XGBoost AFT ------------------------------------
# Valores de partida para el modelo de tiempo de fallo acelerado (AFT). Los
# scripts que optimizan con Optuna parten de esta base y ajustan sobre ella.
XGB_AFT_PARAMS = dict(objective="survival:aft", eval_metric="aft-nloglik",
                      aft_loss_distribution="normal", aft_loss_distribution_scale=1.0,
                      tree_method="hist", max_depth=6, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, seed=42, verbosity=0)
NUM_BOOST_ROUND = 200

RS  = 42       # Semilla aleatoria global, para reproducibilidad.
SEC = 3600.0   # Segundos por hora (para convertir el TTF de segundos a horas).

# --- Paleta de colores Okabe-Ito --------------------------------------------
# Paleta segura para daltonismo. Regla del proyecto: no usar rojo y verde como
# unica distincion; acompanar siempre el color con un glifo, forma o textura.
OI = dict(azul="#0072B2", naranja="#D55E00", celeste="#56B4E9", verde="#009E73",
          negro="#000000", gris="#999999", amarillo="#F0E442", rosa="#CC79A7")
