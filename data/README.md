# Datos

## `toy_publico_modelamiento.parquet`  (incluido, público)
Muestra **anónima** de equipos para reproducir el pipeline. Generada con
`src/crear_toy_publico.py`. Conserva únicamente:
- `Numpos` anonimizado a `EQ001..` (sin relación con el identificador real)
- features del modelo (los `*_encoded`, `RX_TX_ratio`, `N_of_disconnections`, etc.)
- `duration`, `event`, y `time_to_failure` + lags (solo para reproducir el diagnóstico de fuga)

**No contiene** identificadores reales, direcciones de comunicación, ni timestamps.

## Data completa (privada — NO incluida)
El dataset real (datos de telemetría de equipos telecontrolados de una empresa de distribución
eléctrica) es **privado** por confidencialidad y seguridad de infraestructura crítica. Para reproducir sobre él,
apunta la variable de entorno `TESIS_DATA` a tu archivo `dataset_modelamiento.parquet`
(ver README principal).
