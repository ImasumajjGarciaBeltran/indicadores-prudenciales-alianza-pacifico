# Autora: Imasumajj García Beltrán
# Código de matrícula: 2024200500F
# Tema N.º 16: Indicadores prudenciales comparados: Perú, Chile, Colombia y México
# Fecha de extracción: 2026-09-27
"""
04_analisis.py  ·  Fase 7: estimaciones, tablas y figuras del artículo
----------------------------------------------------------------------
Entrada : datos_procesados/datos_procesados_2024200500F.csv  (salida de 03_limpieza_datos.py)
Salidas : carpeta salidas/  (tablas .csv y .tex, salidas de regresión .txt, figuras .png)

Modelo principal (log-log, panel de 4 países × 252 meses, efectos fijos por país):
    Log_MOR_it = a_i + b1·Log_CAPR_it + b2·Log_ROA_it + b3·Log_CRED_it + e_it
    Y = Log_MOR (morosidad) · X1 = Log_CAPR (capital/APR) · X2 = Log_ROA · X3 = Log_CRED (crédito/PBI)
    Los coeficientes son ELASTICIDADES: si X sube 1 %, Y cambia b %.

Hipótesis:  H1: b1 < 0 (riesgo moral) · H2: b2 < 0 (mala gestión) · H3: b3 > 0 (auge crediticio)

Todo se calcula con numpy/scipy (fórmulas matriciales visibles), sin cajas negras:
  - MCO agrupado (como "regress" de Stata) y Efectos Fijos (transformación within)
  - Tabla ANOVA: SC, gl, CM, F, Prob > F, R², R² ajustado, Root MSE
  - Errores estándar clásicos y de Driscoll-Kraay (robustos a heterocedasticidad,
    autocorrelación y dependencia entre países)
  - Pruebas: F de efectos fijos, VIF, Breusch-Pagan, Pesaran CD, AR(1), Jarque-Bera
  - Robustez: modelo en niveles; X rezagadas 12 meses; winsorizado; sin Perú; solo meses observados
  - Por país: una regresión log-log para cada país (errores Newey-West) y prueba F de
    homogeneidad de pendientes (¿el efecto es el mismo en los 4 países?)
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
Y = "Log_MOR"                                   # modelo principal: log-log
XS = ["Log_CAPR", "Log_ROA", "Log_CRED"]
Y_NIV, XS_NIV = "MOR", ["CAPR", "ROA", "CRED"]  # modelo en niveles (robustez y figuras)
ETIQUETAS = {
    "MOR": "Y: Morosidad (%)",
    "CAPR": "X1: Capital regulatorio / APR (%)",
    "ROA": "X2: Rentabilidad sobre activos, ROA (%)",
    "CRED": "X3: Crédito privado / PBI (%)",
    "Log_MOR": "Y: Log de la morosidad",
    "Log_CAPR": "X1: Log del capital regulatorio / APR",
    "Log_ROA": "X2: Log del ROA",
    "Log_CRED": "X3: Log del crédito privado / PBI",
}
SIGNO_ESPERADO = {x: s for x, s in zip(XS + XS_NIV, [-1, -1, +1] * 2)}
HIPOTESIS = {x: h for x, h in zip(XS + XS_NIV, ["H1 riesgo moral", "H2 mala gestión", "H3 auge crediticio"] * 2)}
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
    k_mod = k
    gl_mod = k_mod
    return {
        "yv": yv, "X": X, "sst": sst, "ssm": sst - ssr, "gl_mod": gl_mod, "sigma2": sigma2,
        "p_clasico": pd.Series(2 * stats.t.sf(np.abs(b / np.sqrt(np.diag(V_clasica))), df=gl), index=nombres),
        "t_clasico": pd.Series(b / np.sqrt(np.diag(V_clasica)), index=nombres),
        "efectos_fijos": efectos_fijos, "y": y, "xs": xs,
        "coef": pd.Series(b, index=nombres), "se_clasico": pd.Series(np.sqrt(np.diag(V_clasica)), index=nombres),
        "se_dk": pd.Series(se_dk, index=nombres), "t_dk": pd.Series(t_dk, index=nombres),
        "p_dk": pd.Series(p_dk, index=nombres), "resid": e, "ssr": ssr, "r2": 1 - ssr / sst,
        "n": n, "N": N, "gl": gl, "rezagos_dk": L, "T": T,
    }


def formato(v):
    return f"{int(v)}" if float(v).is_integer() and abs(v) > 1 else f"{v:.4f}"


def salida_regresion(m, titulo, errores="clasico"):
    """Salida tipo Stata ('regress' / 'xtreg, fe'): tabla ANOVA + tabla de coeficientes."""
    n, gl_r, gl_m = m["n"], m["gl"], m["gl_mod"]
    ssm, ssr, sst = m["ssm"], m["ssr"], m["sst"]
    F = (ssm / gl_m) / (ssr / gl_r)
    pF = stats.f.sf(F, gl_m, gl_r)
    r2 = m["r2"]
    r2_aj = 1 - (1 - r2) * (gl_r + gl_m) / gl_r
    rmse = np.sqrt(ssr / gl_r)
    if errores == "dk":
        se, t, p, glt = m["se_dk"], m["t_dk"], m["p_dk"], m["T"] - 1
        nota_se = "Errores estándar de Driscoll-Kraay"
    else:
        se, t, p, glt = m["se_clasico"], m["t_clasico"], m["p_clasico"], gl_r
        nota_se = "Errores estándar clásicos (MCO)"
    tcrit = stats.t.ppf(0.975, glt)
    coef = pd.DataFrame({"Coeficiente": m["coef"], "Error_est": se, "t": t, "P>|t|": p,
                         "IC95_inf": m["coef"] - tcrit * se, "IC95_sup": m["coef"] + tcrit * se})
    anova = pd.DataFrame({"SC": [ssm, ssr, sst], "gl": [gl_m, gl_r, gl_m + gl_r],
                          "CM": [ssm / gl_m, ssr / gl_r, sst / (gl_m + gl_r)]},
                         index=["Modelo", "Residual", "Total"])
    resumen = pd.Series({"Observaciones": n, f"F({gl_m}, {gl_r})": F, "Prob > F": pF,
                         "R-cuadrado" + (" (within)" if m["efectos_fijos"] else ""): r2,
                         "R-cuadrado ajustado": r2_aj, "Root MSE": rmse})
    lineas = [titulo, "=" * 78,
              f"{'Fuente':<10}{'SC':>16}{'gl':>8}{'CM':>16}   {'':<22}",
              "-" * 78]
    etiquetas_res = list(resumen.items())
    for i, (fuente, fila) in enumerate(anova.iterrows()):
        izq = f"{fuente:<10}{fila.SC:>16.6f}{int(fila.gl):>8}{fila.CM:>16.6f}"
        der = etiquetas_res[i] if i < len(etiquetas_res) else ("", "")
        lineas.append(f"{izq}   {der[0]:<22} = {formato(der[1]):>10}" if der[0] else izq)
    for nombre, valor in etiquetas_res[3:]:
        lineas.append(f"{'':<50}   {nombre:<22} = {formato(valor):>10}")
    lineas += ["-" * 78,
               f"{m['y']:<12}{'Coef.':>11}{'Err. est.':>11}{'t':>9}{'P>|t|':>9}{'[IC 95 %]':>24}",
               "-" * 78]
    for var, f in coef.iterrows():
        lineas.append(f"{var:<12}{f.Coeficiente:>11.6f}{f.Error_est:>11.6f}{f.t:>9.2f}{f['P>|t|']:>9.3f}"
                      f"{f.IC95_inf:>12.6f}{f.IC95_sup:>12.6f}")
    lineas += ["-" * 78, nota_se + (" · efectos fijos por país (a_i) incluidos" if m["efectos_fijos"] else "")]
    texto = "\n".join(lineas)
    tabla = pd.concat([coef.round(6), pd.DataFrame({"Coeficiente": resumen.round(6)})])
    return texto, tabla


def por_pais(df, y=None, xs=None):
    """Regresión log-log de cada país por separado (MCO con errores Newey-West).
    Newey-West es el equivalente de Driscoll-Kraay cuando hay una sola serie de tiempo."""
    y, xs = y or Y, xs or XS
    resultados, ssr_total, n_total = {}, 0.0, 0
    for pais, g in df.groupby("pais"):
        g = g.sort_values("t")
        X = np.column_stack([np.ones(len(g)), g[xs].to_numpy()])
        b, e, XtX_inv = ols(g[y].to_numpy(), X)
        T, k = len(g), X.shape[1]
        L = int(np.floor(4 * (T / 100) ** (2 / 9)))
        u = X * e[:, None]
        S = u.T @ u
        for l in range(1, L + 1):
            G = u[l:].T @ u[:-l]
            S += (1 - l / (L + 1)) * (G + G.T)
        se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
        t = b / se
        p = 2 * stats.t.sf(np.abs(t), df=T - k)
        sst = ((g[y] - g[y].mean()) ** 2).sum()
        resultados[pais] = {"coef": pd.Series(b, index=["constante"] + xs), "se": pd.Series(se, index=["constante"] + xs),
                            "p": pd.Series(p, index=["constante"] + xs), "r2": 1 - (e @ e) / sst, "n": T}
        ssr_total += e @ e
        n_total += T
    return resultados, ssr_total, n_total


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
    for eje, var in zip(ejes.flat, [Y_NIV] + XS_NIV):
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
    for eje, x in zip(ejes, XS_NIV):
        for pais, g in df.groupby("pais"):
            eje.scatter(g[x], g[Y_NIV], s=42, color=COLORES[pais], edgecolor="black", linewidth=0.6,
                        alpha=0.85, label=pais, zorder=3)
        m, c = np.polyfit(df[x], df[Y_NIV], 1)
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        r, p = stats.pearsonr(df[x], df[Y_NIV])
        eje.plot(xs, m * xs + c, color="black", lw=2.2, ls="--", zorder=4)
        eje.set_xlabel(ETIQUETAS[x], fontweight="bold")
        eje.set_ylabel(ETIQUETAS[Y_NIV], fontweight="bold")
        eje.set_title(f"r = {r:.3f}  (p = {p:.3g})", fontsize=10)
        eje.grid(alpha=0.3)
    ejes[0].legend(frameon=True, fontsize=8)
    fig.suptitle("Figura 2. Correlación entre la morosidad (Y) y sus determinantes (X), en niveles", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(SALIDAS / "figura2_dispersion.png", dpi=300)
    plt.close(fig)

    # Figuras 2a, 2b, 2c: dispersión de las variables DEL MODELO (en logaritmos), una por figura
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
        eje.set_title(f"Figura 2{letra}. Dispersión entre Log_MOR (Y) y {x} ({ETIQUETAS[x].split(':')[0]})  (r = {r:.3f}; p = {p:.3g})",
                      fontweight="bold", fontsize=10)
        eje.grid(alpha=0.3)
        eje.legend(frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(SALIDAS / f"figura2{letra}_dispersion_Y_{x}.png", dpi=300)
        plt.close(fig)

    # Figura 3: matriz de correlaciones
    corr = df[[Y] + XS].corr()   # correlaciones de las variables del modelo (logaritmos)
    fig, eje = plt.subplots(figsize=(6.2, 5.2))
    im = eje.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    eje.set_xticks(range(4), [Y] + XS, rotation=20)
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
    for eje, var in zip(ejes, [Y_NIV] + XS_NIV):
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
    fig, ejes = plt.subplots(2, 4, figsize=(14, 7.5))
    for eje, var in zip(ejes.flat, [Y_NIV] + XS_NIV + [Y] + XS):
        eje.boxplot(df[var], tick_labels=[var], patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor="#9ecae1"), medianprops=dict(color="black", lw=2))
        eje.set_title(ETIQUETAS[var], fontsize=9, fontweight="bold")
        eje.grid(alpha=0.3, axis="y")
    fig.suptitle("Figura 5. Caja y bigotes de cada variable: niveles (arriba) y logaritmos (abajo)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDAS / "figura5_caja_bigotes.png", dpi=300)
    plt.close(fig)

    # Figura 6: residuos del modelo de efectos fijos
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    ejes[0].hist(residuos, bins=40, color="#6baed6", edgecolor="black")
    ejes[0].set_title("Histograma de residuos", fontweight="bold")
    stats.probplot(residuos, dist="norm", plot=ejes[1])
    ejes[1].set_title("Gráfico Q-Q normal de residuos", fontweight="bold")
    fig.suptitle("Figura 6. Diagnóstico de residuos del modelo log-log de efectos fijos", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDAS / "figura6_residuos.png", dpi=300)
    plt.close(fig)


# ── 5. Programa principal ─────────────────────────────────────────────────
def tabla_hipotesis(m):
    filas = {}
    for x in m["xs"]:
        b, p = m["coef"][x], m["p_dk"][x]
        signo_ok = np.sign(b) == SIGNO_ESPERADO[x]
        filas[HIPOTESIS[x]] = {
            "variable": x, "signo_esperado": "+" if SIGNO_ESPERADO[x] > 0 else "-",
            "coeficiente": round(b, 4), "error_DK": round(m["se_dk"][x], 4), "t": round(m["t_dk"][x], 3),
            "valor_p": round(p, 6),
            "decision": ("Se acepta la hipótesis" if (p < ALFA and signo_ok)
                         else "Signo contrario al esperado" if p < ALFA else "No significativa al 5%"),
        }
    return pd.DataFrame(filas).T


def main():
    SALIDAS.mkdir(exist_ok=True)
    df = pd.read_csv(ENTRADA)
    df["num_mes"] = df["mes"].map({m: i for i, m in enumerate(MESES_ES, start=1)})
    df["fecha"] = pd.to_datetime(dict(year=df["año"], month=df["num_mes"], day=1))
    df = df.sort_values(["pais", "fecha"]).reset_index(drop=True)
    df["t"] = (df["año"] - df["año"].min()) * 12 + df["num_mes"]
    registrar(f"Inicio · {len(df)} observaciones · {df.pais.nunique()} países · {df.t.nunique()} meses")

    # Tabla 1: estadísticos descriptivos (niveles y logaritmos)
    t1 = df[[Y_NIV] + XS_NIV + [Y] + XS].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    t1.columns = ["n", "media", "desv_est", "minimo", "mediana", "maximo"]
    guardar_tabla(t1.round(4), "tabla1_descriptivos", "Estadísticos descriptivos de las variables (2005-2025)")

    # Tabla 2: medias por país (comparación de solidez)
    t2 = df.groupby("pais")[[Y_NIV] + XS_NIV].mean().round(4)
    guardar_tabla(t2, "tabla2_medias_por_pais", "Promedio de los indicadores por país (2005-2025)")

    # Tabla 3: correlaciones de Pearson con valor p (variables del modelo)
    filas = {}
    for x in XS:
        r, p = stats.pearsonr(df[x], df[Y])
        filas[x] = {"r_con_Log_MOR": round(r, 4), "valor_p": round(p, 6), "signif_5%": "Sí" if p < ALFA else "No"}
    t3 = pd.DataFrame(filas).T
    guardar_tabla(t3, "tabla3_correlaciones", "Correlación de Pearson entre Log_MOR y cada variable explicativa")

    # Tabla 4: VIF
    t4 = vif(df).round(4)
    guardar_tabla(t4, "tabla4_vif", "Factor de inflación de varianza (VIF) de las variables del modelo")

    # Modelos
    agrupado = estimar(df, efectos_fijos=False)              # como "regress" de Stata
    fe = estimar(df, efectos_fijos=True)                     # modelo principal
    fe_niv = estimar(df, y=Y_NIV, xs=XS_NIV, efectos_fijos=True)

    # Salidas tipo Stata (texto) + tablas
    txt_mco, tab_mco = salida_regresion(agrupado, "Modelo 1. MCO agrupado log-log:  regress Log_MOR Log_CAPR Log_ROA Log_CRED")
    txt_fe, tab_fe = salida_regresion(fe, "Modelo 2 (principal). Efectos fijos log-log:  xtreg Log_MOR Log_CAPR Log_ROA Log_CRED, fe", errores="dk")
    (SALIDAS / "regresion_modelo1_MCO_loglog.txt").write_text(txt_mco, encoding="utf-8")
    (SALIDAS / "regresion_modelo2_EF_loglog.txt").write_text(txt_fe, encoding="utf-8")
    guardar_tabla(tab_mco, "tabla5a_regresion_MCO_loglog", "Resultados del modelo de regresión lineal log-log por MCO agrupado")
    guardar_tabla(tab_fe, "tabla5b_regresion_EF_loglog", "Resultados del modelo de regresión lineal log-log con efectos fijos (errores Driscoll-Kraay)")

    # Tabla 5: comparación de modelos
    def col(m):
        return [f"{m['coef'][x]:.4f}{estrellas(m['p_dk'][x])} ({m['se_dk'][x]:.4f})" for x in m["xs"]]
    t5 = pd.DataFrame({"MCO agrupado (log-log)": col(agrupado), "Efectos fijos (log-log)": col(fe),
                       "Efectos fijos (niveles)": col(fe_niv)}, index=["X1 CAPR", "X2 ROA", "X3 CRED"])
    t5.loc["R2"] = [f"{agrupado['r2']:.4f}", f"{fe['r2']:.4f} (within)", f"{fe_niv['r2']:.4f} (within)"]
    t5.loc["Observaciones"] = [agrupado["n"], fe["n"], fe_niv["n"]]
    guardar_tabla(t5, "tabla5_comparacion_modelos",
                  "Comparación de modelos (errores Driscoll-Kraay entre paréntesis; *** p<0,01, ** p<0,05, * p<0,10)")
    ecuacion = (f"Log_MOR = a_i {fe['coef'][XS[0]]:+.4f}·Log_CAPR {fe['coef'][XS[1]]:+.4f}·Log_ROA "
                f"{fe['coef'][XS[2]]:+.4f}·Log_CRED")
    (SALIDAS / "ecuacion_estimada.txt").write_text(ecuacion + "\n", encoding="utf-8")

    # Tabla 6: pruebas de diagnóstico (modelo principal)
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
    guardar_tabla(t6, "tabla6_pruebas", "Pruebas de diagnóstico del modelo log-log de efectos fijos")

    # Tabla 7: contraste de hipótesis (modelo principal) y 7b, 7c
    rez = df.copy()
    for x in XS:
        rez[x] = rez.groupby("pais")[x].shift(12)
    rez = rez.dropna(subset=XS)
    m_rez = estimar(rez)
    t7 = tabla_hipotesis(fe)
    guardar_tabla(t7, "tabla7_hipotesis", "Contraste de hipótesis: efectos fijos log-log, errores Driscoll-Kraay, alfa = 5%")
    t7b = tabla_hipotesis(m_rez)
    guardar_tabla(t7b, "tabla7b_hipotesis_rezago12", "Contraste de hipótesis con variables explicativas rezagadas 12 meses (log-log)")
    t7c = tabla_hipotesis(fe_niv)
    guardar_tabla(t7c, "tabla7c_hipotesis_niveles", "Contraste de hipótesis en el modelo en niveles")

    # Tabla 8: robustez (todas log-log salvo "Niveles")
    winsor = df.copy()
    for v in [Y_NIV] + XS_NIV:
        p1 = winsor.groupby("pais")[v].transform(lambda s: s.quantile(0.01))
        p99 = winsor.groupby("pais")[v].transform(lambda s: s.quantile(0.99))
        winsor[v] = winsor[v].clip(p1, p99)
        winsor[f"Log_{v}"] = np.log10(winsor[v])
    observ = df[df[[f"origen_{v}" for v in ["MOR", "CAPR", "ROA"]]].apply(
        lambda c: c.str.contains("observado")).all(axis=1)]
    modelos = {"Base: EF log-log": fe, "Niveles": fe_niv, "X rezagadas 12 meses": m_rez,
               "Winsorizado 1-99%": estimar(winsor), "Sin Perú": estimar(df[df.pais != "Perú"]),
               "Solo meses observados": estimar(observ)}
    t8 = pd.DataFrame({nombre: [f"{m['coef'][x]:.4f}{estrellas(m['p_dk'][x])}" for x in m["xs"]] + [m["n"]]
                       for nombre, m in modelos.items()}, index=["X1 CAPR", "X2 ROA", "X3 CRED", "Observaciones"])
    guardar_tabla(t8, "tabla8_robustez", "Pruebas de robustez (efectos fijos, errores Driscoll-Kraay)")

    # Tabla 9: modelo por país (cada país con su propia regresión log-log)
    paises_res, ssr_sep, n_sep = por_pais(df)
    filas = {}
    for pais, r in paises_res.items():
        fila = {x: f"{r['coef'][x]:.4f}{estrellas(r['p'][x])} ({r['se'][x]:.4f})" for x in XS}
        fila["R2"] = f"{r['r2']:.4f}"
        fila["Observaciones"] = r["n"]
        filas[pais] = fila
    t9 = pd.DataFrame(filas)
    t9.index = ["X1 Log_CAPR", "X2 Log_ROA", "X3 Log_CRED", "R2", "Observaciones"]
    guardar_tabla(t9, "tabla9_modelos_por_pais",
                  "Regresión log-log por país (MCO, errores Newey-West entre paréntesis; *** p<0,01, ** p<0,05, * p<0,10)")

    # Tabla 9b: por país con X rezagadas 12 meses (efecto del crédito con retraso)
    paises_rez, _, _ = por_pais(rez)
    filas = {pais: {x: f"{r['coef'][x]:.4f}{estrellas(r['p'][x])}" for x in XS} for pais, r in paises_rez.items()}
    t9b = pd.DataFrame(filas)
    t9b.index = ["X1 Log_CAPR (t-12)", "X2 Log_ROA (t-12)", "X3 Log_CRED (t-12)"]
    guardar_tabla(t9b, "tabla9b_por_pais_rezago12", "Regresión log-log por país con X rezagadas 12 meses")

    # Prueba de homogeneidad de pendientes (tipo Chow): ¿el efecto es igual en los 4 países?
    k, N = len(XS), df["pais"].nunique()
    F_hom = ((fe["ssr"] - ssr_sep) / (k * (N - 1))) / (ssr_sep / (n_sep - N * (k + 1)))
    p_hom = stats.f.sf(F_hom, k * (N - 1), n_sep - N * (k + 1))
    t9c = pd.DataFrame({"estadistico_F": [round(F_hom, 4)], "gl": [f"({k * (N - 1)}, {n_sep - N * (k + 1)})"],
                        "valor_p": [round(p_hom, 6)],
                        "conclusion": ["Los efectos difieren entre países" if p_hom < ALFA
                                       else "No se rechaza que los efectos sean iguales"]},
                       index=["Homogeneidad de pendientes"])
    guardar_tabla(t9c, "tabla9c_homogeneidad", "Prueba F de homogeneidad de pendientes entre países")

    # Figura 7: coeficientes por país con intervalo de confianza al 95 %
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.3))
    for eje, x in zip(ejes, XS):
        paises = list(paises_res)
        b = [paises_res[p]["coef"][x] for p in paises]
        ic = [1.96 * paises_res[p]["se"][x] for p in paises]
        eje.errorbar(range(len(paises)), b, yerr=ic, fmt="none", ecolor="black", capsize=6, lw=1.5, zorder=2)
        eje.scatter(range(len(paises)), b, s=160, c=[COLORES[p] for p in paises], edgecolor="black", zorder=3)
        eje.axhline(0, color="grey", ls="--", lw=1)
        eje.axhline(fe["coef"][x], color="black", ls=":", lw=1.5, label=f"Panel EF = {fe['coef'][x]:.3f}")
        eje.set_xticks(range(len(paises)), paises)
        eje.set_title(ETIQUETAS[x], fontweight="bold", fontsize=10)
        eje.legend(fontsize=8)
        eje.grid(alpha=0.3, axis="y")
    fig.suptitle("Figura 7. Elasticidades por país (IC 95 %, errores Newey-West)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SALIDAS / "figura7_coeficientes_por_pais.png", dpi=300)
    plt.close(fig)

    figuras(df, fe["resid"])
    registrar("Salidas de regresión (.txt), tablas 1-9 (.csv y .tex) y figuras 1-7 (.png, 300 dpi) guardadas en salidas/")

    print("\n" + txt_mco)
    print("\n" + txt_fe)
    print("\nEcuación estimada (modelo principal):", ecuacion)
    print("\n=== Comparación de modelos ===")
    print(t5.to_string())
    print("\n=== Pruebas de diagnóstico ===")
    print(t6.to_string())
    print("\n=== Contraste de hipótesis (modelo principal) ===")
    print(t7.to_string())
    print("\n=== Hipótesis con X rezagadas 12 meses ===")
    print(t7b.to_string())
    print("\n=== Robustez ===")
    print(t8.to_string())
    print("\n=== Modelo por país (MCO log-log, errores Newey-West) ===")
    print(t9.to_string())
    print("\n=== Por país con X rezagadas 12 meses ===")
    print(t9b.to_string())
    print("\n=== Homogeneidad de pendientes ===")
    print(t9c.to_string())


if __name__ == "__main__":
    main()
