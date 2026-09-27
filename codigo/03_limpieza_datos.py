# Autora: Imasumajj García Beltrán
# Código de matrícula: 2024200500F
# Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México
# Fecha de extracción: 2026-09-27
"""
03_limpieza_datos.py  ·  Fase 6: de datos crudos a datos procesados
--------------------------------------------------------------------
Entrada : datos_crudos/datos_crudos_2024200500F.csv   (salida de 01_extraccion_api.py)
Salidas : datos_procesados/datos_procesados_2024200500F.csv   panel mensual 2005-01 a 2025-12
          diccionario_variables.xlsx
          salidas/tabla_empalme_peru.csv, salidas/tabla_origen_datos.csv, salidas/tabla_atipicos.csv
          README.md (hash SHA-256) y log_ejecucion.txt

Reglas de limpieza (en este orden):
  1. Tipificación: cada periodo se convierte a un mes.
     Mensual -> ese mes · Trimestral -> último mes del trimestre · Anual -> diciembre.
  2. Prioridad de fuentes por país y variable:
     a) FMI mensual observado; b) FMI trimestral observado; c) solo Perú antes de 2010-12:
        Banco Mundial GFDD anual, empalmado con el FMI (retropolación por cociente).
  3. Interpolación lineal de los meses que quedan entre dos datos observados:
        V_mes = V_anterior + (V_actual - V_anterior) × (k / n)
     donde n = 12 si los datos son anuales (ancla en diciembre) o n = 3 si son
     trimestrales (ancla en el último mes del trimestre), y k = meses transcurridos.
     Ej.: PBI 2022 = 200 000 y 2023 = 220 000 -> enero 2023 = 200 000 + 20 000 × 1/12.
  4. Imputación por media para los meses de los extremos sin dato a ambos lados:
     media de los datos observados del mismo país y año.
     CRED de Perú 2025 (el Banco Mundial aún no lo publica): media de los últimos 3 años.
  5. Unión por la llave país + año + mes.
  6. Atípicos: regla de Tukey (Q1 - 1,5·RIC; Q3 + 1,5·RIC) dentro de cada país; se marcan en la
     columna "atipico" (1 = sí) y se conservan; su efecto se evalúa con modelos log-log y winsorizado.
  7. Transformación logarítmica: Log_MOR, Log_CAPR, Log_ROA y Log_CRED (log base 10).
  8. Diccionario de variables y hash SHA-256.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 1. Parámetros ─────────────────────────────────────────────────────────
FECHA_INICIO = "2005-01"
FECHA_CORTE = "2025-12"
PAISES = ["PER", "CHL", "COL", "MEX"]
CODIGO = "2024200500F"
VARIABLES_FMI = ["MOR", "CAPR", "ROA"]
INICIO_FMI_PERU = pd.Period("2010-12", "M")      # primer dato trimestral de Perú en el FMI (2010-Q4)
ANIOS_MEDIA_CRED = 3                              # años usados en la imputación de CRED

ENTRADA = Path(f"datos_crudos/datos_crudos_{CODIGO}.csv")
SALIDA = Path(f"datos_procesados/datos_procesados_{CODIGO}.csv")
ARCHIVO_LOG = Path("log_ejecucion.txt")
MESES = pd.period_range(FECHA_INICIO, FECHA_CORTE, freq="M")
# Nombres en español para la base final
NOMBRE_PAIS = {"PER": "Perú", "CHL": "Chile", "COL": "Colombia", "MEX": "México"}
NOMBRE_MES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio", 7: "Julio",
              8: "Agosto", 9: "Setiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}


def registrar(mensaje):
    linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} · 03_limpieza_datos · {mensaje}"
    print(linea)
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as log:
        log.write(linea + "\n")


# ── 2. Tipificación: texto del periodo -> mes ─────────────────────────────
def periodo_a_mes(texto):
    """'2005-M03' -> 2005-03 · '2005-Q1' -> 2005-03 · '2005' -> 2005-12"""
    texto = str(texto)
    if "-M" in texto:
        return pd.Period(texto.replace("-M", "-"), "M")
    if "-Q" in texto:
        return pd.Period(texto.replace("-", ""), "Q").asfreq("M", "end")
    return pd.Period(f"{texto[:4]}-12", "M")


def serie(crudos, variable, pais, frecuencia, fuente=None):
    """Extrae una serie como pd.Series indexada por mes."""
    filtro = (crudos.variable == variable) & (crudos.pais == pais) & (crudos.frecuencia == frecuencia)
    if fuente:
        filtro &= crudos.fuente == fuente
    datos = crudos[filtro].dropna(subset=["valor"])
    return pd.Series(datos.valor.values, index=datos.mes.values, dtype=float).sort_index()


# ── 3. Empalme de Perú (Banco Mundial GFDD -> escala FMI) ─────────────────
def factores_empalme(crudos):
    """Cociente medio FMI/GFDD en los años donde ambas fuentes tienen dato para Perú."""
    filas = []
    for var in VARIABLES_FMI:
        fmi_a = serie(crudos, var, "PER", "A")
        gfdd = serie(crudos, f"{var}_GFDD", "PER", "A")
        comunes = fmi_a.index.intersection(gfdd.index)
        factor = fmi_a[comunes].mean() / gfdd[comunes].mean()
        filas.append({"variable": var, "anios_comunes": f"{comunes.min().year}-{comunes.max().year}",
                      "n_anios": len(comunes), "media_FMI": round(fmi_a[comunes].mean(), 4),
                      "media_GFDD": round(gfdd[comunes].mean(), 4), "factor": round(factor, 6)})
    return pd.DataFrame(filas).set_index("variable")


# ── 4. Construcción de cada serie mensual ─────────────────────────────────
def construir_fmi(crudos, var, pais, empalme):
    valor = pd.Series(np.nan, index=MESES)
    origen = pd.Series("", index=MESES, dtype=object)
    # índice ampliado un año atrás para poder interpolar enero-noviembre 2005
    ampliado = pd.period_range(MESES[0] - 12, MESES[-1], freq="M")
    ancla = pd.Series(np.nan, index=ampliado)
    tipo = pd.Series("", index=ampliado, dtype=object)

    def poner(s, etiqueta):
        s = s[s.index.isin(ampliado)]
        vacios = ancla[s.index].isna()
        idx = s.index[vacios.values]
        ancla[idx] = s[idx]
        tipo[idx] = etiqueta

    poner(serie(crudos, var, pais, "M"), "FMI_mensual_observado")
    poner(serie(crudos, var, pais, "Q"), "FMI_trimestral_observado")
    if pais == "PER":
        gfdd = serie(crudos, f"{var}_GFDD", "PER", "A")
        gfdd = gfdd[gfdd.index < INICIO_FMI_PERU] * empalme.loc[var, "factor"]
        poner(gfdd, "BM_GFDD_anual_empalmado")

    observados = ancla.notna()
    interpolada = ancla.interpolate(method="linear", limit_area="inside")
    tipo[interpolada.notna() & ~observados] = "interpolado_lineal"

    # extremos sin dato a ambos lados -> media de los observados del mismo país y año
    for mes in interpolada.index[interpolada.isna()]:
        mismo_anio = ancla[(ancla.index.year == mes.year) & observados]
        interpolada[mes] = mismo_anio.mean()
        tipo[mes] = "imputado_media"

    return interpolada[MESES], tipo[MESES]


def construir_cred(crudos, pais):
    anual = serie(crudos, "CRED", pais, "A")
    ampliado = pd.period_range(MESES[0] - 12, MESES[-1], freq="M")
    ancla = pd.Series(np.nan, index=ampliado)
    tipo = pd.Series("", index=ampliado, dtype=object)
    anual = anual[anual.index.isin(ampliado)]
    ancla[anual.index] = anual.values
    tipo[anual.index] = "BM_WDI_anual_observado"
    # años faltantes al final (p. ej. Perú 2025): media de los últimos años observados
    ultimo = anual.index.max()
    for anio in range(ultimo.year + 1, MESES[-1].year + 1):
        mes = pd.Period(f"{anio}-12", "M")
        ancla[mes] = anual.iloc[-ANIOS_MEDIA_CRED:].mean()
        tipo[mes] = "imputado_media"
        registrar(f"CRED {pais} {anio}: imputado con la media de "
                  f"{ANIOS_MEDIA_CRED} años ({anual.index[-ANIOS_MEDIA_CRED].year}-{ultimo.year}) = {ancla[mes]:.4f}")
    observados = ancla.notna()
    interpolada = ancla.interpolate(method="linear", limit_area="inside")
    tipo[interpolada.notna() & ~observados] = "interpolado_lineal"
    return interpolada[MESES], tipo[MESES]


# ── 5. Programa principal ─────────────────────────────────────────────────
def main():
    registrar(f"Inicio · entrada {ENTRADA}")
    crudos = pd.read_csv(ENTRADA, dtype={"periodo": str})
    crudos["valor"] = pd.to_numeric(crudos["valor"], errors="coerce")
    crudos["mes"] = crudos["periodo"].apply(periodo_a_mes)

    empalme = factores_empalme(crudos)
    Path("salidas").mkdir(exist_ok=True)
    empalme.to_csv("salidas/tabla_empalme_peru.csv", encoding="utf-8")
    registrar("Factores de empalme Perú (FMI/GFDD): " +
              ", ".join(f"{v}={f:.4f}" for v, f in empalme["factor"].items()))

    bloques = []
    for pais in PAISES:
        tabla = pd.DataFrame(index=MESES)
        for var in VARIABLES_FMI:
            tabla[var], tabla[f"origen_{var}"] = construir_fmi(crudos, var, pais, empalme)
        tabla["CRED"], tabla["origen_CRED"] = construir_cred(crudos, pais)
        tabla.insert(0, "pais", pais)
        bloques.append(tabla)

    panel = pd.concat(bloques)
    panel["pais"] = panel["pais"].map(NOMBRE_PAIS)
    panel.insert(1, "año", panel.index.year)
    panel.insert(2, "mes", panel.index.month.map(NOMBRE_MES))
    # Orden: id, país, año, mes, Y (MOR), X1 (CAPR), X2 (ROA), X3 (CRED) y el origen de cada dato
    columnas = ["pais", "año", "mes", "MOR", "CAPR", "ROA", "CRED",
                "origen_MOR", "origen_CAPR", "origen_ROA", "origen_CRED"]
    panel = panel[columnas].reset_index(drop=True)
    panel.insert(0, "id", range(1, len(panel) + 1))
    for var in ["MOR", "CAPR", "ROA", "CRED"]:
        panel[var] = panel[var].round(6)
        # Transformación logarítmica (base 10) al lado de cada variable: Log_MOR, Log_CAPR, ...
        # Reduce el peso de los valores atípicos y permite leer los coeficientes como elasticidades.
        if (panel[var] <= 0).any():
            raise ValueError(f"{var} tiene valores <= 0: no se puede aplicar logaritmo")
        panel.insert(panel.columns.get_loc(var) + 1, f"Log_{var}", np.log10(panel[var]).round(6))

    # Controles de calidad
    assert len(panel) == len(MESES) * len(PAISES), "El panel no tiene el número de filas esperado"
    assert not panel[["MOR", "CAPR", "ROA", "CRED"]].isna().any().any(), "Quedan valores vacíos"
    assert not panel.duplicated(["pais", "año", "mes"]).any(), "Hay filas duplicadas país-año-mes"

    # Tratamiento de atípicos (regla de Tukey DENTRO de cada país): se identifican y se marcan
    # en la columna "atipico"; NO se eliminan ni se modifican porque corresponden a episodios
    # económicos reales. Su influencia se evalúa en 04_analisis.py (modelo log-log y winsorizado).
    filas = []
    marca = pd.Series(0, index=panel.index)
    for var in ["MOR", "CAPR", "ROA", "CRED"]:
        for pais, g in panel.groupby("pais"):
            q1, q3 = g[var].quantile([0.25, 0.75])
            ric = q3 - q1
            inf, sup = q1 - 1.5 * ric, q3 + 1.5 * ric
            fuera = g[(g[var] < inf) | (g[var] > sup)]
            marca[fuera.index] = 1
            filas.append({"variable": var, "pais": pais, "limite_inf": round(inf, 4),
                          "limite_sup": round(sup, 4), "n_atipicos": len(fuera)})
    panel["atipico"] = marca.astype(int)
    atipicos = pd.DataFrame(filas)
    atipicos.to_csv("salidas/tabla_atipicos.csv", index=False, encoding="utf-8")
    registrar(f"Atípicos identificados dentro de cada país: {int(panel['atipico'].sum())} filas marcadas (no se eliminan)")

    # Origen de los datos: cuántos meses son observados, interpolados o imputados
    origen = pd.concat([panel[f"origen_{v}"].value_counts().rename(v) for v in ["MOR", "CAPR", "ROA", "CRED"]],
                       axis=1).fillna(0).astype(int)
    origen.to_csv("salidas/tabla_origen_datos.csv", encoding="utf-8")

    SALIDA.parent.mkdir(exist_ok=True)
    panel.to_csv(SALIDA, index=False, encoding="utf-8")
    huella = hashlib.sha256(SALIDA.read_bytes()).hexdigest()
    registrar(f"Fin · {len(panel)} filas × {panel.shape[1]} columnas (incluye columnas Log_) · SHA-256 {huella}")

    crear_diccionario()
    actualizar_readme(huella, len(panel))

    print("\nOrigen de cada dato (meses):")
    print(origen.to_string())
    print("\nAtípicos por país (se marcan en la columna 'atipico', no se eliminan):")
    print(atipicos.to_string(index=False))
    print("\nFactores de empalme de Perú:")
    print(empalme.to_string())


# ── 6. Diccionario de variables y README ──────────────────────────────────
def crear_diccionario():
    fmi = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/FSIC/~/"
    bm = "https://api.worldbank.org/v2/country/PER;CHL;COL;MEX/indicator/"
    filas = [
        ("id", "Número correlativo de la observación (1 a 1008)", "número", "—", "—", "—"),
        ("pais", "País: Perú, Chile, Colombia o México; con año y mes forma la llave del panel", "texto", "—", "—", "—"),
        ("año", "Año (2005 a 2025)", "número", "mensual", "—", "—"),
        ("mes", "Mes en español (Enero a Diciembre)", "texto", "mensual", "—", "—"),
        ("MOR", "Y (endógena) · Morosidad: préstamos morosos / préstamos brutos totales", "%", "mensual",
         "FMI FSIC AQ12_CFSI_PT (sector S12CFSI); Perú 2005-2010: Banco Mundial GFDD.SI.02 empalmado",
         fmi + "PER+CHL+COL+MEX.S12CFSI.AQ12_CFSI_PT.M+Q+A"),
        ("CAPR", "X1 · Capital regulatorio / activos ponderados por riesgo", "%", "mensual",
         "FMI FSIC FSI688_CFSI_PT (sector S12CFSI); Perú 2005-2010: Banco Mundial GFDD.SI.05 empalmado",
         fmi + "PER+CHL+COL+MEX.S12CFSI.FSI688_CFSI_PT.M+Q+A"),
        ("ROA", "X2 · Rentabilidad sobre activos (utilidad antes de impuestos / activos promedio)", "%", "mensual",
         "FMI FSIC ROA_CFSI_PT (sector S12CFSI); Perú 2005-2010: Banco Mundial GFDD.EI.05 empalmado",
         fmi + "PER+CHL+COL+MEX.S12CFSI.ROA_CFSI_PT.M+Q+A"),
        ("CRED", "X3 · Crédito interno al sector privado", "% del PBI", "anual, mensualizado",
         "Banco Mundial WDI FS.AST.PRVT.GD.ZS", bm + "FS.AST.PRVT.GD.ZS?format=json&source=2"),
        ("Log_MOR / Log_CAPR / Log_ROA / Log_CRED",
         "Logaritmo en base 10 de cada variable; en el modelo log-log los coeficientes son elasticidades",
         "log10(%)", "mensual", "03_limpieza_datos.py", "—"),
        ("origen_MOR / origen_CAPR / origen_ROA / origen_CRED",
         "Cómo se obtuvo cada dato: FMI_mensual_observado, FMI_trimestral_observado, "
         "BM_GFDD_anual_empalmado, BM_WDI_anual_observado, interpolado_lineal o imputado_media",
         "texto", "mensual", "03_limpieza_datos.py", "—"),
        ("atipico", "1 si alguna variable de la fila es atípica dentro de su país (regla de Tukey); 0 si no",
         "0/1", "mensual", "03_limpieza_datos.py", "—"),
    ]
    dic = pd.DataFrame(filas, columns=["variable", "definicion", "unidad", "frecuencia",
                                       "fuente", "endpoint"])
    dic["fecha_corte"] = FECHA_CORTE
    dic.to_excel("diccionario_variables.xlsx", index=False)
    registrar("diccionario_variables.xlsx actualizado")


def actualizar_readme(huella, filas):
    readme = Path("README.md")
    if not readme.exists():
        return
    texto = readme.read_text(encoding="utf-8")
    bloque = (f"## Hash SHA-256 de datos_procesados_{CODIGO}.csv\n"
              f"`{huella}`  ({filas} filas; generado el {datetime.now():%Y-%m-%d %H:%M})\n")
    patron = r"## Hash SHA-256 de datos_procesados_[^\n]*\n(?:(?!\n## ).)*"
    if re.search(patron, texto, flags=re.S):
        texto = re.sub(patron, bloque.rstrip("\n"), texto, flags=re.S)
    else:
        texto += "\n" + bloque
    readme.write_text(texto, encoding="utf-8")
    registrar("README.md: hash SHA-256 actualizado")


if __name__ == "__main__":
    main()
