# Autora: Imasumajj García Beltrán
# Código de matrícula: 2024200500F
# Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México
# Fecha de extracción: 2026-09-27
"""
02_scraping_web.py  ·  Vía 2 (web scraping o descarga programática)  ·  NO APLICA EN LA UNIDAD I
----------------------------------------------------------------------------------------------
Según el numeral 2.4.1 de la consigna, en la Unidad I solo es obligatoria una vía automatizada
(preferentemente API). Esta base se construyó íntegramente con la Vía 1 (01_extraccion_api.py):
  - FMI, Financial Soundness Indicators (API SDMX 3.0)
  - Banco Mundial, WDI y GFDD (API v2)

La Vía 2 (por ejemplo, descarga programática de los boletines de la SBS para contrastar la serie
peruana) queda planteada como extensión para la Unidad II. Este script se incluye para respetar la
estructura de entrega; al ejecutarse solo registra en el log que la vía no se utilizó.
"""

from datetime import datetime
from pathlib import Path

ARCHIVO_LOG = Path("log_ejecucion.txt")


def main():
    linea = (f"{datetime.now():%Y-%m-%d %H:%M:%S} · 02_scraping_web · Vía 2 no utilizada en la Unidad I "
             f"(opcional según numeral 2.4.1); la base proviene de 01_extraccion_api.py")
    print(linea)
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as log:
        log.write(linea + "\n")


if __name__ == "__main__":
    main()
