# Autora: Imasumajj García Beltrán
# Código de matrícula: 2024200500F
# Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México
# Fecha de extracción: 2026-09-27
"""
01_extraccion_api.py  ·  Vía 1 (API)  ·  Fase 4: datos crudos
-------------------------------------------------------------
Descarga, SIN MODIFICAR ningún valor, los datos de dos APIs oficiales:

  1) FMI · Financial Soundness Indicators (FSIC), API SDMX 3.0
     - AQ12_CFSI_PT   Morosidad: préstamos morosos / préstamos brutos (%)       -> MOR
     - FSI688_CFSI_PT Capital regulatorio / activos ponderados por riesgo (%)   -> CAPR
     - ROA_CFSI_PT    Rentabilidad sobre activos (%)                            -> ROA
     Sector S12CFSI (entidades captadoras de depósitos), frecuencias M, Q y A.

  2) Banco Mundial · API v2 (WDI y GFDD)
     - FS.AST.PRVT.GD.ZS  Crédito interno al sector privado (% del PBI)         -> CRED
     - GFDD.SI.02 / GFDD.SI.05 / GFDD.EI.05  Morosidad, capital/APR y ROA anuales:
       respaldo para Perú 2005-2010, años que el FMI no cubre.

Salidas (carpeta datos_crudos/):
  - datos_crudos_2024200500F_fmi.csv   respuesta del FMI, byte por byte
  - datos_crudos_2024200500F_bm.json   respuestas del Banco Mundial, tal cual
  - datos_crudos_2024200500F.csv       ambas fuentes juntas en formato largo (mismos valores)
  - log_ejecucion.txt                  fecha, hora, código HTTP y filas de cada consulta
Ejecutar desde la Carpeta 3:  python codigo/01_extraccion_api.py
"""

import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── 1. Parámetros congelados (numeral 2.4.5) ──────────────────────────────
FECHA_INICIO = "2005-01-01"      # inicio del panel
FECHA_CORTE = "2025-12-31"       # fin del panel
ANIO_DESCARGA_DESDE = 2004       # 1 año antes: permite interpolar ene-feb 2005 en la limpieza
PAISES = ["PER", "CHL", "COL", "MEX"]
CODIGO = "2024200500F"

FMI_BASE = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/FSIC/~/"
FMI_SECTOR = "S12CFSI"
FMI_INDICADORES = {"AQ12_CFSI_PT": "MOR", "FSI688_CFSI_PT": "CAPR", "ROA_CFSI_PT": "ROA"}
FMI_FRECUENCIAS = ["M", "Q", "A"]

BM_BASE = "https://api.worldbank.org/v2/country/{paises}/indicator/{indicador}"
BM_INDICADORES = {                      # código: (variable, id de la base en el Banco Mundial)
    "FS.AST.PRVT.GD.ZS": ("CRED", "2"),       # 2  = World Development Indicators
    "GFDD.SI.02": ("MOR_GFDD", "32"),         # 32 = Global Financial Development
    "GFDD.SI.05": ("CAPR_GFDD", "32"),
    "GFDD.EI.05": ("ROA_GFDD", "32"),
}

# Rutas RELATIVAS a la Carpeta 3 (numeral 2.4.4)
CARPETA_CRUDOS = Path("datos_crudos")
ARCHIVO_LOG = Path("log_ejecucion.txt")
CORREO = os.getenv("CORREO_CONTACTO", "e_2024200500F@uncp.edu.pe")
CABECERAS = {"User-Agent": f"Investigacion-academica-UNCP/{CODIGO} ({CORREO})"}
PAUSA_SEG = 1                            # pausa mínima entre solicitudes (numeral 2.4.8)


# ── 2. Funciones de apoyo ─────────────────────────────────────────────────
def registrar(mensaje):
    """Escribe una línea con fecha y hora en el log y en pantalla."""
    linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} · 01_extraccion_api · {mensaje}"
    print(linea)
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as log:
        log.write(linea + "\n")


def pedir(url, params=None, accept=None, intentos=3):
    """GET con reintentos: si la API falla, espera y vuelve a intentar."""
    cabeceras = dict(CABECERAS)
    if accept:
        cabeceras["Accept"] = accept
    for intento in range(1, intentos + 1):
        try:
            r = requests.get(url, params=params, headers=cabeceras, timeout=180)
            if r.status_code == 200:
                return r
            registrar(f"HTTP {r.status_code} en intento {intento}: {r.url[:120]}")
        except requests.RequestException as error:
            registrar(f"Error de conexión en intento {intento}: {error}")
        time.sleep(PAUSA_SEG * 3 * intento)
    raise RuntimeError(f"No se pudo descargar tras {intentos} intentos: {url}")


# ── 3. FMI: indicadores de solidez financiera ─────────────────────────────
def descargar_fmi():
    clave = ".".join([
        "+".join(PAISES),                 # COUNTRY
        FMI_SECTOR,                       # SECTOR
        "+".join(FMI_INDICADORES),        # INDICATOR
        "+".join(FMI_FRECUENCIAS),        # FREQUENCY
    ])
    r = pedir(FMI_BASE + clave, accept="text/csv")
    # Se guarda la respuesta exacta, sin tocar un solo carácter
    (CARPETA_CRUDOS / f"datos_crudos_{CODIGO}_fmi.csv").write_text(r.text, encoding="utf-8")
    fmi = pd.read_csv(io.StringIO(r.text))
    registrar(f"FMI FSIC · HTTP {r.status_code} · {len(fmi)} filas · endpoint {FMI_BASE}{clave}")
    largo = pd.DataFrame({
        "fuente": "FMI_FSIC",
        "codigo_serie": fmi["INDICATOR"],
        "variable": fmi["INDICATOR"].map(FMI_INDICADORES),
        "pais": fmi["COUNTRY"],
        "frecuencia": fmi["FREQUENCY"],
        "periodo": fmi["TIME_PERIOD"].astype(str),
        "valor": fmi["OBS_VALUE"],
    })
    return largo


# ── 4. Banco Mundial: WDI y GFDD ──────────────────────────────────────────
def descargar_bm():
    respuestas, filas = {}, []
    for indicador, (variable, base) in BM_INDICADORES.items():
        url = BM_BASE.format(paises=";".join(PAISES), indicador=indicador)
        params = {"format": "json", "source": base, "per_page": 20000,
                  "date": f"{ANIO_DESCARGA_DESDE}:{FECHA_CORTE[:4]}"}
        r = pedir(url, params=params)
        contenido = r.json()
        respuestas[indicador] = contenido        # respuesta completa, tal cual
        datos = contenido[1] if len(contenido) > 1 and contenido[1] else []
        registrar(f"Banco Mundial {indicador} · HTTP {r.status_code} · {len(datos)} filas · endpoint {r.url}")
        for d in datos:
            filas.append({
                "fuente": "BM_WDI" if base == "2" else "BM_GFDD",
                "codigo_serie": indicador,
                "variable": variable,
                "pais": d["countryiso3code"],
                "frecuencia": "A",
                "periodo": str(d["date"]),
                "valor": d["value"],
            })
        time.sleep(PAUSA_SEG)
    with open(CARPETA_CRUDOS / f"datos_crudos_{CODIGO}_bm.json", "w", encoding="utf-8") as f:
        json.dump(respuestas, f, ensure_ascii=False, indent=1)
    return pd.DataFrame(filas)


# ── 5. Programa principal ─────────────────────────────────────────────────
def main():
    CARPETA_CRUDOS.mkdir(exist_ok=True)
    registrar(f"Inicio · ventana {FECHA_INICIO} a {FECHA_CORTE} (descarga desde {ANIO_DESCARGA_DESDE})")

    fmi = descargar_fmi()
    time.sleep(PAUSA_SEG)
    bm = descargar_bm()

    crudos = pd.concat([fmi, bm], ignore_index=True)
    # Solo se recorta a la ventana de extracción declarada; los valores NO se modifican
    anio = crudos["periodo"].str[:4].astype(int)
    crudos = crudos[(anio >= ANIO_DESCARGA_DESDE) & (anio <= int(FECHA_CORTE[:4]))]
    crudos = crudos.sort_values(["fuente", "variable", "pais", "frecuencia", "periodo"])
    crudos["fecha_extraccion"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"

    salida = CARPETA_CRUDOS / f"datos_crudos_{CODIGO}.csv"
    crudos.to_csv(salida, index=False, encoding="utf-8")
    registrar(f"Fin · {len(crudos)} filas guardadas en {salida}")

    # Resumen para revisar en pantalla
    resumen = (crudos.dropna(subset=["valor"])
                     .groupby(["variable", "frecuencia", "pais"])["periodo"]
                     .agg(["min", "max", "count"]))
    print("\nResumen de lo descargado (desde, hasta, n.º de datos):")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
