# Indicadores prudenciales comparados: Perú, Chile, Colombia y México

- **Autora:** Imasumajj García Beltrán
- **Código de matrícula:** 2024200500F
- **Curso:** Finanzas I (055D) · Universidad Nacional del Centro del Perú · 2026-II · Unidad I
- **Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México**
- **Repositorio:** https://github.com/ImasumajjGarciaBeltran/indicadores-prudenciales-alianza-pacifico

## Modelo
Panel mensual de 4 países (Perú, Chile, Colombia, México):

`MOR_it = a_i + b1·CAPR_it + b2·ROA_it + b3·CRED_it + e_it`

| Variable | Descripción | Fuente |
|---|---|---|
| MOR | Morosidad (NPL): préstamos improductivos / cartera bruta (%) | FMI – Financial Soundness Indicators (API) |
| CAPR | Capital regulatorio / activos ponderados por riesgo (%) | FMI – Financial Soundness Indicators (API) |
| ROA | Rentabilidad sobre activos (%) | FMI – Financial Soundness Indicators (API) |
| CRED | Crédito interno al sector privado (% del PBI) | Banco Mundial – WDI, FS.AST.PRVT.GD.ZS (API) |

## Fuentes y endpoints
_Se completa en la Fase 4._

## Fecha de corte
_FECHA_INICIO y FECHA_CORTE se completan en la Fase 3._

## Orden de ejecución
1. `codigo/01_extraccion_api.py`
2. `codigo/02_scraping_web.py` (opcional)
3. `codigo/03_limpieza_datos.py`
4. `codigo/04_analisis.py`

## Versión del lenguaje y librerías
Ver `requirements.txt`.

## Hash SHA-256 de datos_procesados_2024200500F.csv
_Se completa en la Fase 6._
