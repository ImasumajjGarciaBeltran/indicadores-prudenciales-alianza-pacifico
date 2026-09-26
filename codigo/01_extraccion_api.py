# Autora: Imasumajj García Beltrán
# Código de matrícula: 2024200500F
# Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México
# Fecha de extracción: AAAA-MM-DD (se completa al extraer)

"""Vía 1 (API). Descarga los datos crudos:
  - FMI, Financial Soundness Indicators (FSIC): morosidad, capital regulatorio/APR, ROA (trimestral).
  - Banco Mundial, WDI: crédito privado/PBI, FS.AST.PRVT.GD.ZS (anual).
Salida: datos_crudos/datos_crudos_2024200500F.csv y log_ejecucion.txt
"""
# Parámetros congelados (numeral 2.4.5): se fijan en la Fase 3
FECHA_INICIO = "2005-01-01"
FECHA_CORTE  = "2025-12-31"
PAISES = ["PER", "CHL", "COL", "MEX"]
