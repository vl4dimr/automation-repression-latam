# -*- coding: utf-8 -*-
"""Traducción de rótulos de figuras.

Con FIG_LANG=es la función L devuelve la versión en español; sin esa variable devuelve el texto
original, de modo que las figuras en inglés se reproducen sin cambios. La carpeta de salida se
elige con FIG_DIR (por defecto "figures"), definida en config.py.

Uso:  FIG_LANG=es FIG_DIR=figures_es python 02_data/scripts/03_descriptives.py
"""
import os

LANG = os.environ.get("FIG_LANG", "en")

ES = {
    # países
    "Argentina": "Argentina", "Bolivia": "Bolivia", "Brazil": "Brasil", "Chile": "Chile",
    "Colombia": "Colombia", "Costa Rica": "Costa Rica", "Dominican Republic": "República Dominicana",
    "Ecuador": "Ecuador", "El Salvador": "El Salvador", "Guatemala": "Guatemala",
    "Honduras": "Honduras", "Mexico": "México", "Nicaragua": "Nicaragua", "Panama": "Panamá",
    "Paraguay": "Paraguay", "Peru": "Perú", "Uruguay": "Uruguay", "Venezuela": "Venezuela",
    # figura 1
    "labour share (left)": "participación laboral (izq.)",
    "log K/L (right)": "log K/L (der.)",
    "log K/L (right axis)": "log K/L (eje derecho)",
    "Figure 1. Labour share and capital per worker, 1960–2024 (shaded: baseline window). "
    "Sources: PWT 10.01, ILO, WDI.":
        "Figura 1. Participación laboral y capital por trabajador, 1960–2024 (sombreado: ventana de "
        "referencia). Fuentes: PWT 10.01, OIT, WDI.",
    # figura 2
    "Robot-import-value stock per worker (2015 US$)":
        "Acervo de valor importado de robots por trabajador (USD de 2015)",
    "Robot import-value proxy, 1996–2024\nSource: UN Comtrade; PIM with 10% depreciation":
        "Aproximación por valor importado de robots, 1996–2024\nFuente: UN Comtrade; inventario "
        "perpetuo con 10 % de depreciación",
    "Comtrade pull not complete": "Descarga de Comtrade incompleta",
    # figura 3
    "non-democracy (RoW)": "no democracia (RoW)",
    "physical repression (1 - v2x_clphy)": "represión física (1 − v2x_clphy)",
    "PTS (rescaled 0-1)": "PTS (reescalada 0–1)",
    "successful coup": "golpe exitoso",
    "Figure 3. Repression, regime and coups, 1960–2024. Sources: V-Dem v16, PTS-2025, Powell & Thyne.":
        "Figura 3. Represión, régimen y golpes de Estado, 1960–2024. Fuentes: V-Dem v16, PTS-2025, "
        "Powell y Thyne.",
    # figura 4
    "Physical repression (1 - v2x_clphy)": "Represión física (1 − v2x_clphy)",
    "Absolute redistribution (Gini market - Gini disposable)":
        "Redistribución absoluta (Gini de mercado − Gini disponible)",
    "(a) Country-years 1990–2024, colour = log K/L":
        "(a) Años-país 1990–2024, color = log K/L",
    "log K/L": "log K/L",
    "slope": "pendiente",
    "Repression, within-country and within-year deviation":
        "Represión, desviación intrapaís e intraaño",
    "Redistribution, same deviation": "Redistribución, misma desviación",
    "(b) Exact country and year fixed-effect residuals":
        "(b) Residuos exactos de efectos fijos de país y año",
    "Repression and redistribution: descriptive associations\nSources: V-Dem and SWIID 9.92":
        "Represión y redistribución: asociaciones descriptivas\nFuentes: V-Dem y SWIID 9.92",
    # figura 5
    "capital per worker rising (dk > 0)": "capital por trabajador creciente (dk > 0)",
    "capital per worker falling (dk <= 0)": "capital por trabajador decreciente (dk ≤ 0)",
    "repression tercile, t+1": "tercil de represión, t+1",
    "repression tercile, t": "tercil de represión, t",
    "Annual transitions between repression terciles, 1960–2024":
        "Transiciones anuales entre terciles de represión, 1960–2024",
    "low": "bajo", "mid": "medio", "high": "alto",
    # figura 6
    "coup attempt": "intento de golpe",
    "log capital per worker (sextile medians), democracies only":
        "log del capital por trabajador (medianas por sextil), solo democracias",
    "% of democracy-years with a coup": "% de años-democracia con golpe",
    "(a) Coup risk against capital per worker":
        "(a) Riesgo de golpe frente al capital por trabajador",
    "Tax revenue, % of GDP (quintile medians), democracies only":
        "Recaudación tributaria, % del PIB (medianas por quintil), solo democracias",
    "(b) Coup risk against fiscal capacity": "(b) Riesgo de golpe frente a la capacidad fiscal",
    "Figure 6. Coups in democracies, 1960–2024. Sources: Powell & Thyne via V-Dem, PWT, CEPALSTAT.":
        "Figura 6. Golpes de Estado en democracias, 1960–2024. Fuentes: Powell y Thyne mediante "
        "V-Dem, PWT, CEPALSTAT.",
    "Table 1. Descriptive statistics, 1990–2024": "Tabla 1. Estadísticos descriptivos, 1990–2024",
    # gráfico de coeficientes (01_fe_models)
    "outcome units per SD of regressor, 95% CI":
        "unidades del resultado por DE del regresor, IC 95 %",
    "H1a: labour share on log K/L": "H1a: participación laboral sobre log K/L",
    "H2: repression on log K/L": "H2: represión sobre log K/L",
    "H2: redistribution on lagged repression": "H2: redistribución sobre represión rezagada",
    "PWT/ILO": "PWT/OIT", "+robots": "+robots", "+routine": "+rutinarias",
    "ILO share": "part. OIT", "1960-2024": "1960–2024", "PWT varying": "PWT variable",
    "physical": "física", "physical+threat": "física+amenaza", "physical+R": "física+R",
    "CSO": "OSC", "PTS": "PTS", "composite": "compuesta",
    "physical (anti-sys.)": "física (antisist.)", "composite (anti-sys.)": "compuesta (antisist.)",
    "redistribution": "redistribución", "social exp.": "gasto social", "tax": "impuestos",
    "low rep. (K/L)": "baja repr. (K/L)", "high rep. (K/L)": "alta repr. (K/L)",
    # umbrales (02_threshold)
    "Threshold": "Umbral", "LR": "RV",
    "log K/L (t-1), 12 bins": "log K/L (t−1), 12 intervalos",
    "physical repression, two-way demeaned":
        "represión física, centrada por país y año",
    "Within-country repression against capital per worker":
        "Represión intrapaís frente al capital por trabajador",
    "Physical rep., threshold in K/L": "Repr. física, umbral en K/L",
    "Composite rep., threshold in K/L": "Repr. compuesta, umbral en K/L",
    "Physical rep., threshold in threat": "Repr. física, umbral en la amenaza",
    "Robots, threshold in K/L": "Robots, umbral en K/L",
    "Redistribution, threshold in K/L": "Redistribución, umbral en K/L",
    # bosque (03_causal_forest)
    "Mean fitted slope": "Pendiente ajustada media",
    "Exploratory slope: index points per log point of capital":
        "Pendiente exploratoria: puntos del índice por punto logarítmico de capital",
    "Exploratory fitted slope": "Pendiente ajustada exploratoria",
    "Capital profile (descriptive)": "Perfil de capital (descriptivo)",
    "Threat profile (descriptive)": "Perfil de amenaza (descriptivo)",
    "Military spending profile (descriptive)": "Perfil de gasto militar (descriptivo)",
    # eventos (04_h4_coups)
    "Descriptive profiles around autocratization onset":
        "Perfiles descriptivos alrededor del inicio de autocratización",
    "Years to onset": "Años respecto al inicio",
    "event countries": "países con evento",
    "log K/L": "log K/L",
    "log robots per worker": "log de robots por trabajador",
    "labour share": "participación laboral",
    "tax revenue % GDP": "recaudación tributaria % del PIB",
}


def L(text):
    """Devuelve el rótulo traducido cuando FIG_LANG=es; en otro caso, el original."""
    if LANG != "es":
        return text
    return ES.get(text, text)
