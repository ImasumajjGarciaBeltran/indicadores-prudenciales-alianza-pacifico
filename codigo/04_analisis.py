# Autora: Imasumajj García Beltrán
# Código de matrícula: 2024200500F
# Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México
# Fecha de extracción: 2026-09-27
"""
04_analisis.py  ·  Fase 7: estimaciones, tablas y figuras del artículo
----------------------------------------------------------------------
Entrada : datos_procesados/datos_procesados_2024200500F.csv  (salida de 03_limpieza_datos.py)
Salidas : carpeta salidas/  (tablas .csv y .tex, figuras .png)

Modelo (panel de 4 países × 252 meses, efectos fijos por país):
    MOR_it = a_i + b1·CAPR_it + b2·ROA_it + b3·CRED_it + e_it
    Y = MOR (morosidad) · X1 = CAPR (capital/APR) · X2 = ROA · X3 = CRED (crédito privado/PBI)

Hipótesis:  H1: b1 < 0 (riesgo moral) · H2: b2 < 0 (mala gestión) · H3: b3 > 0 (auge crediticio)

Todo se calcula con numpy/scipy (fórmulas matriciales visibles), sin cajas negras:
  - MCO agrupado y Efectos Fijos (transformación within)
  - Errores estándar de Driscoll-Kraay: robustos a heterocedasticidad, autocorrelación
    (la interpolación la genera) y dependencia entre países.
  - Pruebas: F de efectos fijos, VIF, Breusch-Pagan, Pesaran CD, autocorrelación AR(1)
  - Robustez: X rezagadas 12 meses; solo Chile-Colombia-México; solo meses observados
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── 1. Parámetros ─────────────────────────────────────────────────────────
CODIGO = "2024200500F"
ENTRADA = Path(f"datos_procesados/datos_procesados_{CODIGO}.csv")
SALIDAS = Path("salidas")
ARCHIVO_LOG = Path("log_ejecucion.txt")
Y = "MOR"
XS = ["CAPR", "ROA", "CRED"]
ETIQUETAS = {
    "MOR": "Y: Morosidad (%)",
    "CAPR": "X1: Capital regulatorio / APR (%)",
    "ROA": "X2: Rentabilidad sobre activos, ROA (%)",
    "CRED": "X3: Crédito privado / PBI (%)",
}
SIGNO_ESPERADO = {"CAPR": -1, "ROA": -1, "CRED": +1}
HIPOTESIS = {"CAPR": "H1 riesgo moral", "ROA": "H2 mala gestión", "CRED": "H3 auge crediticio"}
COLORES = {"Perú": "#D62728", "Chile": "#1F77B4", "Colombia": "#F2B701", "México": "#2CA02C"}
MESES_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
            "Setiembre", "Octubre", "Noviembre", "Diciembre"]
ALFA = 0.05


def registrar(mensaje):
    linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} · 04_analisis · {mensaje}"
    print(linea)
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as log:
        log.write(linea + "\n")


def guardar_tabla(df, nombre, titulo):
    """Guarda cada tabla en CSV (para revisar) y en LaTeX (para el artículo)."""
    df.to_csv(SALIDAS / f"{nombre}.csv", encoding="utf-8")
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lineas = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{titulo}}}",
              "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline", (" & ".join(cols) + " \\\\").replace("%", "\\%").replace("_", "\\_"), "\\hline"]
    for idx, fila in df.iterrows():
        celdas = [str(idx)] + [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in fila]
        lineas.append(" & ".join(celdas).replace("%", "\\%").replace("_", "\\_") + " \\\\")
    lineas += ["\\hline", "\\end{tabular}", f"\\label{{tab:{nombre}}}", "\\end{table}"]
    (SALIDAS / f"{nombre}.tex").write_text("\n".join(lineas), encoding="utf-8")


# ── 2. Estimadores (álgebra matricial) ────────────────────────────────────
def ols(y, X):
    """b = (X'X)^-1 X'y ; devuelve coeficientes, residuos e inversa de X'X."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    return b, y - X @ b, XtX_inv


def driscoll_kraay(X, e, tiempos, XtX_inv):
    """Varianza de Driscoll-Kraay (1998): HAC de Newey-West sobre h_t = Σ_i x_it·e_it."""
    T_unicos = np.unique(tiempos)
    T = len(T_unicos)
    h = np.vstack([(X[tiempos == t] * e[tiempos == t, None]).sum(axis=0) for t in T_unicos])
    rezagos = int(np.floor(4 * (T / 100) ** (2 / 9)))
    S = h.T @ h
    for l in range(1, rezagos + 1):
        w = 1 - l / (rezagos + 1)
        G = h[l:].T @ h[:-l]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv, rezagos, T


def within(df, columnas, grupo="pais"):
    """Transformación within: resta la media de cada país (elimina el efecto fijo a_i)."""
    return df[columnas] - df.groupby(grupo)[columnas].transform("mean")


def estimar(df, y=Y, xs=XS, efectos_fijos=True):
    n, k = len(df), len(xs)
    N = df["pais"].nunique()
    if efectos_fijos:
        dem = within(df, [y] + xs)
        yv, X = dem[y].to_numpy(), dem[xs].to_numpy()
        gl = n - N - k
    else:
        yv = df[y].to_numpy()
        X = np.column_stack([np.ones(n), df[xs].to_numpy()])
        gl = n - k - 1
    b, e, XtX_inv = ols(yv, X)
    sigma2 = (e @ e) / gl
    V_clasica = sigma2 * XtX_inv
    V_dk, L, T = driscoll_kraay(X, e, df["t"].to_numpy(), XtX_inv)
    nombres = xs if efectos_fijos else ["constante"] + xs
    se_dk = np.sqrt(np.diag(V_dk))
    t_dk = b / se_dk
    p_dk = 2 * stats.t.sf(np.abs(t_dk), df=T - 1)
    ssr = e @ e
    sst = ((yv - yv.mean()) ** 2).sum()
    return {
        "coef": pd.Series(b, index=nombres), "se_clasico": pd.Series(np.sqrt(np.diag(V_clasica)), index=nombres),
        "se_dk": pd.Series(se_dk, index=nombres), "t_dk": pd.Series(t_dk, index=nombres),
        "p_dk": pd.Series(p_dk, index=nombres), "resid": e, "ssr": ssr, "r2": 1 - ssr / sst,
        "n": n, "N": N, "gl": gl, "rezagos_dk": L, "T": T,
    }


def estrellas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── 3. Pruebas de diagnóstico ─────────────────────────────────────────────
def vif(df):
    filas = {}
    for x in XS:
        otras = [c for c in XS if c != x]
        X = np.column_stack([np.ones(len(df)), df[otras].to_numpy()])
        _, e, _ = ols(df[x].to_numpy(), X)
        r2 = 1 - (e @ e) / ((df[x] - df[x].mean()) ** 2).sum()
        filas[x] = {"R2_auxiliar": r2, "VIF": 1 / (1 - r2)}
    return pd.DataFrame(filas).T


def breusch_pagan(df, e):
    """LM = n·R² de regresar e² sobre las X; se distribuye chi²(k)."""
    X = np.column_stack([np.ones(len(df)), df[XS].to_numpy()])
    u2 = e ** 2
    _, v, _ = ols(u2, X)
    r2 = 1 - (v @ v) / ((u2 - u2.mean()) ** 2).sum()
    lm = len(df) * r2
    return lm, stats.chi2.sf(lm, len(XS))


def pesaran_cd(df, e):
    """CD = raíz(2T / N(N-1)) · Σ correlaciones de residuos entre pares de países ~ N(0,1)."""
    r = pd.DataFrame({"pais": df["pais"].values, "t": df["t"].values, "e": e}).pivot(index="t", columns="pais", values="e")
    N, T = r.shape[1], r.shape[0]
    corr = r.corr().to_numpy()
    suma = corr[np.triu_indices(N, 1)].sum()
    cd = np.sqrt(2 * T / (N * (N - 1))) * suma
    return cd, 2 * stats.norm.sf(abs(cd))


def ar1_residuos(df, e):
    """Coeficiente AR(1) promedio de los residuos dentro de cada país (Durbin-Watson ≈ 2(1-ρ))."""
    r = pd.DataFrame({"pais": df["pais"].values, "e": e})
    rhos = [np.corrcoef(g.e.values[1:], g.e.values[:-1])[0, 1] for _, g in r.groupby("pais")]
    rho = float(np.mean(rhos))
    return rho, 2 * (1 - rho)


# ── 4. Figuras ────────────────────────────────────────────────────────────
def figuras(df, residuos):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Figura 1: evolución de las 4 variables por país
    fig, ejes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for eje, var in zip(ejes.flat, [Y] + XS):
        for pais, g in df.groupby("pais"):
            eje.plot(g["fecha"], g[var], color=COLORES[pais], lw=1.8, label=pais)
        eje.set_title(ETIQUETAS[var], fontweight="bold")
        eje.grid(alpha=0.3)
    manejadores, nombres = ejes[0, 0].get_legend_handles_labels()
    fig.legend(manejadores, nombres, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Figura 1. Indicadores prudenciales mensuales, 2005–2025", fontweight="bold")
    fig.text(0.01, 0.005, "Fuente: FMI (FSIC) y Banco Mundial (WDI, GFDD) vía API. Elaboración propia.", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(SALIDAS / "figura1_series.png", dpi=300)
    plt.close(fig)

    # Figura 2: dispersión Y contra cada X, puntos llamativos y recta de ajuste
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.8))
    for eje, x in zip(ejes, XS):
        for pais, g in df.groupby("pais"):
            eje.scatter(g[x], g[Y], s=42, color=COLORES[pais], edgecolor="black", linewidth=0.6,
                        alpha=0.85, label=pais, zorder=3)
        m, c = np.polyfit(df[x], df[Y], 1)
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        r, p = stats.pearsonr(df[x], df[Y])
        eje.plot(xs, m * xs + c, color="black", lw=2.2, ls="--", zorder=4)
        eje.set_xlabel(ETIQUETAS[x], fontweight="bold")
        eje.set_ylabel(ETIQUETAS[Y], fontweight="bold")
        eje.set_title(f"r = {r:.3f}  (p = {p:.3g})", fontsize=10)
        eje.grid(alpha=0.3)
    ejes[0].legend(frameon=True, fontsize=8)
    fig.suptitle("Figura 2. Correlación entre la morosidad (Y) y sus determinantes (X)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(SALIDAS / "figura2_dispersion.png", dpi=300)
    plt.close(fig)

    # Figuras 2a, 2b, 2c: cada diagrama de dispersión por separado (Y vs X1, Y vs X2, Y vs X3)
    for letra, x in zip("abc", XS):
        fig, eje = plt.subplots(figsize=(7, 5.2))
        for pais, g in df.groupby("pais"):
            eje.scatter(g[x], g[Y], s=55, color=COLORES[pais], edgecolor="black", linewidth=0.7,
                        alpha=0.9, label=pais, zorder=3)
        m, c = np.polyfit(df[x], df[Y], 1)
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        r, p = stats.pearsonr(df[x], df[Y])
        eje.plot(xs, m * xs + c, color="black", lw=2.4, ls="--", zorder=4,
                 label=f"Ajuste lineal: Y = {c:.3f} {m:+.3f}·X")
        eje.set_xlabel(ETIQUETAS[x], fontweight="bold")
        eje.set_ylabel(ETIQUETAS[Y], fontweight="bold")
        eje.set_title(f"Figura 2{letra}. Dispersión entre Y y {ETIQUETAS[x].split(':')[0]}  (r = {r:.3f}; p = {p:.3g})",
                      fontweight="bold", fontsize=10)
        eje.grid(alpha=0.3)
        eje.legend(frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(SALIDAS / f"figura2{letra}_dispersion_Y_{x}.png", dpi=300)
        plt.close(fig)

    # Figura 3: matriz de correlaciones
    corr = df[[Y] + XS].corr()
    fig, eje = plt.subplots(figsize=(6.2, 5.2))
    im = eje.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    eje.set_xticks(range(4), [Y] + XS)
    eje.set_yticks(range(4), [Y] + XS)
    for i in range(4):
        for j in range(4):
            eje.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontweight="bold")
    fig.colorbar(im, fraction=0.046)
    eje.set_title("Figura 3. Matriz de correlaciones de Pearson", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SALIDAS / "figura3_correlaciones.png", dpi=300)
    plt.close(fig)

    # Figura 4: comparación de la solidez por país (cajas)
    fig, ejes = plt.subplots(1, 4, figsize=(15, 4.2))
    paises = list(COLORES)
    for eje, var in zip(ejes, [Y] + XS):
        cajas = eje.boxplot([df.loc[df.pais == p, var] for p in paises], tick_labels=paises, patch_artist=True)
        for caja, p in zip(cajas["boxes"], paises):
            caja.set_facecolor(COLORES[p])
            caja.set_alpha(0.75)
        eje.set_title(ETIQUETAS[var], fontsize=9, fontweight="bold")
        eje.grid(alpha=0.3, axis="y")
    fig.suptitle("Figura 4. Comparación de indicadores prudenciales por país, 2005–2025", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDAS / "figura4_comparacion_paises.png", dpi=300)
    plt.close(fig)

    # Figura 5: caja y bigotes de cada variable (panel completo)
    fig, ejes = plt.subplots(1, 4, figsize=(13, 4.2))
    for eje, var in zip(ejes, [Y] + XS):
        eje.boxplot(df[var], tick_labels=[var], patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor="#9ecae1"), medianprops=dict(color="black", lw=2))
        eje.set_title(ETIQUETAS[var], fontsize=9, fontweight="bold")
        eje.grid(alpha=0.3, axis="y")
    fig.suptitle("Figura 5. Diagrama de caja y bigotes de cada variable, 2005–2025", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDAS / "figura5_caja_bigotes.png", dpi=300)
    plt.close(fig)

    # Figura 6: residuos del modelo de efectos fijos
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    ejes[0].hist(residuos, bins=40, color="#6baed6", edgecolor="black")
    ejes[0].set_title("Histograma de residuos", fontweight="bold")
    stats.probplot(residuos, dist="norm", plot=ejes[1])
    ejes[1].set_title("Gráfico Q-Q normal de residuos", fontweight="bold")
    fig.suptitle("Figura 6. Diagnóstico de residuos del modelo de efectos fijos", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDAS / "figura6_residuos.png", dpi=300)
    plt.close(fig)


# ── 5. Programa principal ─────────────────────────────────────────────────
def main():
    SALIDAS.mkdir(exist_ok=True)
    df = pd.read_csv(ENTRADA)
    df["num_mes"] = df["mes"].map({m: i for i, m in enumerate(MESES_ES, start=1)})
    df["fecha"] = pd.to_datetime(dict(year=df["año"], month=df["num_mes"], day=1))
    df = df.sort_values(["pais", "fecha"]).reset_index(drop=True)
    df["t"] = (df["año"] - df["año"].min()) * 12 + df["num_mes"]
    registrar(f"Inicio · {len(df)} observaciones · {df.pais.nunique()} países · {df.t.nunique()} meses")

    # Tabla 1: estadísticos descriptivos
    t1 = df[[Y] + XS].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    t1.columns = ["n", "media", "desv_est", "minimo", "mediana", "maximo"]
    guardar_tabla(t1.round(4), "tabla1_descriptivos", "Estadísticos descriptivos del panel (2005-2025)")

    # Tabla 2: medias por país (comparación de solidez)
    t2 = df.groupby("pais")[[Y] + XS].mean().round(4)
    guardar_tabla(t2, "tabla2_medias_por_pais", "Promedio de los indicadores por país (2005-2025)")

    # Tabla 3: correlaciones de Pearson con valor p
    filas = {}
    for x in XS:
        r, p = stats.pearsonr(df[x], df[Y])
        filas[x] = {"r_con_MOR": round(r, 4), "valor_p": round(p, 6), "signif_5%": "Sí" if p < ALFA else "No"}
    t3 = pd.DataFrame(filas).T
    guardar_tabla(t3, "tabla3_correlaciones", "Correlación de Pearson entre la morosidad y cada variable explicativa")

    # Tabla 4: VIF (multicolinealidad; VIF > 10 sería problemático)
    t4 = vif(df).round(4)
    guardar_tabla(t4, "tabla4_vif", "Factor de inflación de varianza (VIF)")

    # Tabla 5: modelos
    agrupado = estimar(df, efectos_fijos=False)
    fe = estimar(df, efectos_fijos=True)
    t5 = pd.DataFrame({
        "MCO_agrupado": [f"{agrupado['coef'][x]:.4f}{estrellas(agrupado['p_dk'][x])} ({agrupado['se_dk'][x]:.4f})" for x in XS],
        "Efectos_fijos": [f"{fe['coef'][x]:.4f}{estrellas(fe['p_dk'][x])} ({fe['se_dk'][x]:.4f})" for x in XS],
    }, index=XS)
    t5.loc["R2"] = [f"{agrupado['r2']:.4f}", f"{fe['r2']:.4f} (within)"]
    t5.loc["Observaciones"] = [agrupado["n"], fe["n"]]
    t5.loc["Paises"] = [agrupado["N"], fe["N"]]
    t5.loc["Error estándar"] = ["Driscoll-Kraay", "Driscoll-Kraay"]
    guardar_tabla(t5, "tabla5_modelos",
                  "Determinantes de la morosidad bancaria: MCO agrupado y efectos fijos (errores DK entre paréntesis)")

    # Tabla 5b: salida detallada del modelo de regresión (efectos fijos)
    tcrit = stats.t.ppf(1 - ALFA / 2, fe["T"] - 1)
    t5b = pd.DataFrame({
        "coeficiente": fe["coef"], "error_DK": fe["se_dk"], "t": fe["t_dk"], "valor_p": fe["p_dk"],
        "IC95_inf": fe["coef"] - tcrit * fe["se_dk"], "IC95_sup": fe["coef"] + tcrit * fe["se_dk"],
    }).round(6)
    k = len(XS)
    r2_aj = 1 - (1 - fe["r2"]) * (fe["n"] - fe["N"]) / fe["gl"]
    F_glob = (fe["r2"] / k) / ((1 - fe["r2"]) / fe["gl"])
    p_F = stats.f.sf(F_glob, k, fe["gl"])
    t5b.loc["R2 within"] = [round(fe["r2"], 6)] + [""] * 5
    t5b.loc["R2 ajustado"] = [round(r2_aj, 6)] + [""] * 5
    t5b.loc["F global (p)"] = [round(F_glob, 4), "", "", round(p_F, 6), "", ""]
    t5b.loc["Observaciones"] = [fe["n"]] + [""] * 5
    guardar_tabla(t5b, "tabla5b_regresion_detallada",
                  "Resultados del modelo de regresión lineal de panel con efectos fijos")
    ecuacion = (f"MOR = a_i {fe['coef']['CAPR']:+.4f}·CAPR {fe['coef']['ROA']:+.4f}·ROA "
                f"{fe['coef']['CRED']:+.4f}·CRED")
    (SALIDAS / "ecuacion_estimada.txt").write_text(ecuacion + "\n", encoding="utf-8")

    # Tabla 6: pruebas de diagnóstico sobre el modelo de efectos fijos
    k = len(XS)
    F = ((agrupado["ssr"] - fe["ssr"]) / (fe["N"] - 1)) / (fe["ssr"] / fe["gl"])
    pF = stats.f.sf(F, fe["N"] - 1, fe["gl"])
    bp, pbp = breusch_pagan(df, fe["resid"])
    cd, pcd = pesaran_cd(df, fe["resid"])
    rho, dw = ar1_residuos(df, fe["resid"])
    jb, pjb = stats.jarque_bera(fe["resid"])
    t6 = pd.DataFrame({
        "estadistico": [F, bp, cd, rho, jb],
        "valor_p": [pF, pbp, pcd, np.nan, pjb],
        "conclusion": [
            "Se prefieren efectos fijos" if pF < ALFA else "MCO agrupado es suficiente",
            "Hay heterocedasticidad" if pbp < ALFA else "Homocedasticidad",
            "Hay dependencia entre países" if pcd < ALFA else "Sin dependencia entre países",
            f"Autocorrelación AR(1) de residuos (Durbin-Watson ≈ {dw:.3f})",
            "Residuos no normales (con n grande, la inferencia sigue siendo válida)" if pjb < ALFA
            else "Residuos normales",
        ],
    }, index=["F de efectos fijos", "Breusch-Pagan", "Pesaran CD", "rho AR(1)", "Jarque-Bera"]).round(6)
    guardar_tabla(t6, "tabla6_pruebas", "Pruebas de diagnóstico del modelo de efectos fijos")

    # Tabla 7: contraste de hipótesis (efectos fijos, Driscoll-Kraay)
    filas = {}
    for x in XS:
        b, p = fe["coef"][x], fe["p_dk"][x]
        signo_ok = np.sign(b) == SIGNO_ESPERADO[x]
        filas[HIPOTESIS[x]] = {
            "variable": x, "signo_esperado": "+" if SIGNO_ESPERADO[x] > 0 else "-",
            "coeficiente": round(b, 4), "error_DK": round(fe["se_dk"][x], 4), "t": round(fe["t_dk"][x], 3),
            "valor_p": round(p, 6),
            "decision": ("Se acepta la hipótesis" if (p < ALFA and signo_ok)
                         else "Signo contrario al esperado" if p < ALFA else "No significativa al 5%"),
        }
    t7 = pd.DataFrame(filas).T
    guardar_tabla(t7, "tabla7_hipotesis", "Contraste de hipótesis (efectos fijos, errores Driscoll-Kraay, alfa = 5%)")

    # Tabla 8: robustez
    rez = df.copy()
    for x in XS:
        rez[x] = rez.groupby("pais")[x].shift(12)
    rez = rez.dropna(subset=XS)
    sin_peru = df[df.pais != "Perú"]
    observ = df[df[[f"origen_{v}" for v in [Y, "CAPR", "ROA"]]].apply(
        lambda c: c.str.contains("observado")).all(axis=1)]
    modelos = {"Base (EF)": fe, "X rezagadas 12 meses": estimar(rez), "Sin Perú": estimar(sin_peru),
               "Solo meses observados": estimar(observ)}
    t8 = pd.DataFrame({nombre: [f"{m['coef'][x]:.4f}{estrellas(m['p_dk'][x])}" for x in XS] + [m["n"]]
                       for nombre, m in modelos.items()}, index=XS + ["Observaciones"])
    guardar_tabla(t8, "tabla8_robustez", "Pruebas de robustez del modelo de efectos fijos")

    figuras(df, fe["resid"])
    registrar("Tablas 1-8 y 5b (.csv y .tex), figuras 1-6 (.png, 300 dpi) y ecuación estimada guardadas en salidas/")

    print("\n=== Modelo de regresión (efectos fijos, errores Driscoll-Kraay) ===")
    print(t5.to_string())
    print("\n=== Resultados detallados del modelo ===")
    print(t5b.to_string())
    print("\nEcuación estimada:", ecuacion)
    print("\n=== Pruebas de diagnóstico ===")
    print(t6.to_string())
    print("\n=== Contraste de hipótesis ===")
    print(t7.to_string())
    print("\n=== Robustez ===")
    print(t8.to_string())


if __name__ == "__main__":
    main()
