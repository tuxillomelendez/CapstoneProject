# Predicción de Time-to-Failure de comunicaciones SCADA — código de reproducción

Análisis de supervivencia (XGBoost-AFT) para anticipar la pérdida del enlace de
comunicación de equipos telecontrolados. Este repositorio contiene el código que
reproduce los experimentos de la tesis. **La data completa es privada** (Enel,
infraestructura crítica); se incluye un **toy público anónimo** para reproducir el
pipeline de punta a punta sin exponer información sensible.

## Estructura
```
src/        código fuente (motor + scripts numerados)
  config.py        rutas, features, hiperparámetros, paleta  (editar aquí)
  lib.py           motor: carga, split por equipo, AFT, RSF, métricas con censura
  01_diagnostico_leakage.py     diagnóstico del data leakage de los lags
  02_modelo_corregido.py          matriz 2x2 + modelo corregido + Cox + residuos
  03_comparacion_xgb_rsf.py     comparación justa XGB-AFT vs RSF (Optuna parejo) + IBS
  04_barrido_escala.py          barrido de escala: C-index, IBS y tiempo por tamaño
  05_experimento_gpu.py         escalabilidad CPU vs GPU (Discusión / Trabajo Futuro)
  06_verificar_cindex.py        C-index de Harrell vs Uno (IPCW)
  crear_toy_publico.py          genera el toy anónimo desde la data privada
data/       toy público anónimo + nota sobre la data privada
outputs/    resultados (JSON + figuras) que generan los scripts
```

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir
Por defecto, todos los scripts corren sobre el **toy público** (`data/toy_publico_modelamiento.parquet`):
```bash
python src/01_diagnostico_leakage.py
python src/02_modelo_corregido.py
python src/03_comparacion_xgb_rsf.py
python src/06_verificar_cindex.py
```
Para reproducir sobre la **data completa** (privada), apunta la variable de entorno a tu parquet:
```bash
# Windows
set TESIS_DATA=C:\ruta\dataset_modelamiento.parquet
# Linux / Mac
export TESIS_DATA=/ruta/dataset_modelamiento.parquet
```
Los resultados (JSON + figuras Okabe-Ito) quedan en `outputs/`.

## Notas
- **Modelo corregido:** el modelo NO usa las variables `lag` (eran fuga de datos: `lag1 ≈ target`).
  El script 01 documenta la fuga; el 02 la cuantifica con la matriz 2x2.
- **Daltonismo:** todas las figuras usan la paleta Okabe-Ito + glifos/hachurado, nunca solo color
  ni rojo-verde.
- **GPU (script 05):** requiere drivers NVIDIA/CUDA. La GPU solo acelera; el C-index es idéntico al de CPU.
