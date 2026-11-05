# EDA adicional: concentración y balance por estación

Módulo: `mibici/eda_station_balance.py` · Equipo 5 · Fase 1.

## Qué mide

1. **Concentración de salidas:** porcentaje de los viajes iniciados en el **10 % de estaciones de origen más activas**. El denominador es la cantidad de estaciones que presentan al menos una salida en el período; la cantidad seleccionada se redondea hacia arriba.
2. **Balance neto por estación:** cantidad de **salidas menos llegadas** en una ventana horaria. Una salida se asigna según `inicio` y `origen_id`; una llegada según `fin` y `destino_id`. Se analizan lunes a viernes en 07:00–09:59 y 17:00–19:59.

Balances positivos indican predominio de salidas, y negativos, de llegadas. **No son medidas de ocupación real:** no se conocen inventarios iniciales, reubicaciones operativas ni intentos de viajes no realizados. Se presentan balances acumulados en el período, no una estimación puntual de bicicletas disponibles.

## Cómo reproducir

Desde la raíz del repositorio, después de ejecutar la descarga y limpieza de datos:

```bash
python -m mibici.eda_station_balance
python -m pytest tests/test_eda_station_balance.py -q
```

El comando utiliza `data/interim/trips_v1.parquet` y el catálogo de estaciones ya descargado. Produce:

- `reports/station_balance_summary.json` — métricas de concentración, criterio de cálculo, período y estaciones con balances más extremos.
- `reports/station_balance_matutino.csv` y `reports/station_balance_vespertino.csv` — tabla de salidas, llegadas y balance neto por estación.
- `reports/figures/07_balance_neto_matutino.png` y `reports/figures/07_balance_neto_vespertino.png` — gráficos de las 10 estaciones con mayor balance positivo y las 10 con mayor balance negativo de cada ventana.

Las salidas se regeneran localmente, **no se incorporan cifras supuestas al código**. Para el PDF de la Fase 1, copiar la cuota exacta del JSON e incluir la figura correspondiente, indicando que representa un balance histórico de viajes, no inventario observado.

## Validación

Las pruebas sintéticas cubren proporciones, redondeo, ventanas horarias, exclusión de fines de semana, contabilización independiente de inicio/fin y generación de reportes. Verificar también el pipeline original (`python -m pytest tests -q`) antes de fusionar el PR.
