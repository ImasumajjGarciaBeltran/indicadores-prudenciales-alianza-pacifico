# Indicadores prudenciales comparados: Perú, Chile, Colombia y México

**Determinantes de la morosidad bancaria en la Alianza del Pacífico, 2005-2025: un análisis de datos de panel**

| | |
|---|---|
| Autora | Imasumajj García Beltrán |
| Código de matrícula | 2024200500F |
| Curso | Finanzas I (055D) · Facultad de Economía · Universidad Nacional del Centro del Perú · 2026-II · Unidad I |
| Tema del temario | Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México |
| Repositorio | https://github.com/ImasumajjGarciaBeltran/indicadores-prudenciales-alianza-pacifico |
| Fecha de extracción | 2026-09-27 |
| Periodo (FECHA_INICIO a FECHA_CORTE) | 2005-01-01 a 2025-12-31 |

## Modelo
Panel mensual de 4 países × 252 meses = 1008 observaciones. Modelo principal log-log con efectos fijos por país
y errores estándar de Driscoll-Kraay:

`Log_MOR_it = a_i + b1·Log_CAPR_it + b2·Log_ROA_it + b3·Log_CRED_it + e_it`

| Variable | Rol | Definición | Fuente |
|---|---|---|---|
| MOR | Y | Préstamos morosos / préstamos brutos (%) | FMI FSIC `AQ12_CFSI_PT` |
| CAPR | X1 | Capital regulatorio / activos ponderados por riesgo (%) | FMI FSIC `FSI688_CFSI_PT` |
| ROA | X2 | Rentabilidad sobre activos (%) | FMI FSIC `ROA_CFSI_PT` |
| CRED | X3 | Crédito interno al sector privado (% del PBI) | Banco Mundial WDI `FS.AST.PRVT.GD.ZS` |

Perú 2005-2010 (no cubierto por el FMI): Banco Mundial GFDD `GFDD.SI.02`, `GFDD.SI.05`, `GFDD.EI.05`, empalmados a la escala del FMI.
Definiciones completas en `diccionario_variables.xlsx`.

## Fuentes y endpoints (Vía 1: API; no requieren clave)
- FMI, Financial Soundness Indicators (SDMX 3.0):
  `https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/FSIC/~/PER+CHL+COL+MEX.S12CFSI.AQ12_CFSI_PT+FSI688_CFSI_PT+ROA_CFSI_PT.M+Q+A`
- Banco Mundial, API v2:
  `https://api.worldbank.org/v2/country/PER;CHL;COL;MEX/indicator/{codigo}?format=json&source={2|32}&date=2004:2025`
- Vía 2 (scraping): no utilizada en la Unidad I (opcional, numeral 2.4.1).

## Estructura
```
codigo/            01_extraccion_api.py · 02_scraping_web.py · 03_limpieza_datos.py · 04_analisis.py
datos_crudos/      datos_crudos_2024200500F.csv (+ respuestas originales _fmi.csv y _bm.json)
datos_procesados/  datos_procesados_2024200500F.csv
salidas/           tablas (.csv y .tex), salidas de regresión (.txt) y figuras (.png)
diccionario_variables.xlsx · requirements.txt · .env.example · log_ejecucion.txt
```

## Orden de ejecución (desde esta carpeta)
```
pip install -r requirements.txt
python codigo/01_extraccion_api.py    # descarga los datos crudos (API)
python codigo/02_scraping_web.py      # vía 2: no aplica en la Unidad I
python codigo/03_limpieza_datos.py    # panel mensual, logaritmos, atípicos, hash
python codigo/04_analisis.py          # tablas, figuras y modelos
```
Las claves de API se declaran como variables de entorno (ver `.env.example`); estas APIs no piden clave.

## Versión del lenguaje y librerías
```
# Python 3.13.15 (Google Colab)
# Librerías usadas por codigo/01, 02, 03 y 04 · instalar con: pip install -r requirements.txt
requests==2.32.4
pandas==2.2.3
numpy==2.1.3
scipy==1.16.3
matplotlib==3.10.0
openpyxl==3.1.5
```

## Hash SHA-256 de datos_procesados_2024200500F.csv
`cf071587222c83647e56c0d37055bb00526edd4d026677ae123fa0984400e6b3`  (1008 filas; verificado el 2026-09-27 04:01)

## Citas de los datos (APA 7)
- Fondo Monetario Internacional. (2026). *Financial Soundness Indicators (FSIC)* [Base de datos]. Consultado el 2026-09-27, de https://data.imf.org/en/datasets/IMF.STA:FSIC
- Banco Mundial. (2026). *World Development Indicators* [Base de datos]. Consultado el 2026-09-27, de https://api.worldbank.org/v2/
- Banco Mundial. (2026). *Global Financial Development Database* [Base de datos]. Consultado el 2026-09-27, de https://api.worldbank.org/v2/
