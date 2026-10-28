import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LOADBANK-MASTER | Busbar 2D FDM + DFM Layout",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. DATABASE MATERIALI E COEFFICIENTI (DIN 43671 / MELSOM & BOOTH / REV. 2.1)
# ==============================================================================
METALLI_K1 = {
    "Rame elettrolitico ETP CW004A (rho=0,0178 uOhm m)": {
        "K1": 1.26, "rho": 0.0178, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0, "alpha": 0.00393
    },
    "Rame elettrolitico OFHC CW008A (rho=0,0176 uOhm m)": {
        "K1": 1.27, "rho": 0.0176, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0, "alpha": 0.00393
    },
    "Lega CuCr1Zr CW106C invecchiata (rho=0,0223 uOhm m)": {
        "K1": 0.17778 * (0.0223**-0.487), "rho": 0.0223, "mu_r": 1.000000, "dens": 8890.0, "csp": 385.0, "alpha": 0.00320
    },
    "Alluminio purezza elettrica EN-AW 1350A (rho=0,028 uOhm m)": {
        "K1": 1.02, "rho": 0.0280, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0, "alpha": 0.00403
    },
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {
        "K1": 1.00, "rho": 0.0290, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0, "alpha": 0.00403
    },
    "Lega d'alluminio EN-AW 6101B T6 (rho=0,0325 uOhm m)": {
        "K1": 0.94, "rho": 0.0325, "mu_r": 1.000000, "dens": 2700.0, "csp": 880.0, "alpha": 0.00360
    },
    "Acciaio Inox AISI 304 EN 1.4301 (rho=0,72 uOhm m)": {
        "K1": 0.17778 * (0.72**-0.487), "rho": 0.7200, "mu_r": 1.020000, "dens": 8000.0, "csp": 500.0, "alpha": 0.00100
    },
    "Lega CuCr1Zr [Tabella Excel Storica] (rho=0,028 uOhm m)": {
        "K1": 1.01, "rho": 0.0280, "mu_r": 1.000000, "dens": 8910.0, "csp": 450.0, "alpha": 0.00300
    },
}

VENTILAZIONE_K4 = {
    "Ambiente interno - aria calma limitata / vano chiuso (K4 = 1,00)": 1.00,
    "Ambiente interno - aria calma non limitata / grande volume (K4 = 1,06)": 1.06,
    "Ambiente interno - ventilazione forzata leggera v ≈ 0,5 m/s (K4 = 1,10)": 1.10,
    "Ambiente esterno / condotto ventilazione normale v ≈ 1,0 m/s (K4 = 1,20)": 1.20,
    "Ambiente interno - conduttori chiusi in guaina non ventilati (K4 = 0,70)": 0.70,
}

SUPERFICIE_K5 = {
    "Barre grezze di estrusione o trafilatura (emissività ~0,4)": "grezza",
    "Barre verniciate opache o con termorestringente nero (emissività ~0,9)": "verniciata",
}

FORMA_K6 = {
    "Normale (piatti rettangolari) (K6 = 1,00)": 1.00,
    "Profili C o L accoppiati interni - dist. 1/4 h (K6 = 1,35)": 1.35,
    "Profili C accoppiati esterni - dist. 1/4 h (K6 = 1,43)": 1.43,
}

K7_BASE_TABLE = {1: 1.00, 2: 1.79, 3: 2.45, 4: 3.10, 5: 3.70, 6: 4.30}

MODO_ORIZZ = "Orizzontale (0° - Piega lungo X, ala di piatto)"
MODO_VERT = "Verticale (90° - Piega lungo Y, ala di costa)"


# ==============================================================================
# 2. MOTORE COMPLETO MELSOM & BOOTH (TELEMA REV. 2.1)
# ==============================================================================
def calcola_portata_accoppiata(h, s, n, d, giacitura_costa, metallo_str, freq, t_amb, dt, vent_str, sup_str, forma_str):
    mat = METALLI_K1[metallo_str]
    k1 = mat["K1"]

    k2 = 1.04 if t_amb == 20 else (
        1.01 if t_amb == 35 else (
            1.00 if t_amb == 40 else (
                0.98 if t_amb == 50 else -0.002 * t_amb + 1.08
            )
        )
    )
    k3_map = {30: 0.86, 40: 1.00, 50: 1.14, 65: 1.38}
    k3 = k3_map.get(dt, 0.0148 * dt + 0.4086)
    k4 = VENTILAZIONE_K4[vent_str]
    k6 = FORMA_K6[forma_str]
    is_profilo_cl = k6 > 1.00
    is_alluminio = "Alluminio" in metallo_str or "alluminio" in metallo_str

    if SUPERFICIE_K5[sup_str] == "grezza":
        k5 = 1.00
    else:
        k5 = (1.20 if is_alluminio else 1.12) if is_profilo_cl else (1.12 if is_alluminio else 1.08)

    if giacitura_costa:
        k8 = 1.00
    else:
        k8_map = {1: 0.86, 2: 0.80, 3: 0.74, 4: 0.68}
        k8 = k8_map.get(n, max(0.50, 0.86 - 0.06 * (n - 1)))

    p_1 = 2.0 * (h + s)
    sez_1 = float(h * s)
    sez_tot = float(h * s * n)

    if n == 1:
        k7 = 1.00
        s_eff = float(s)
    else:
        r_hs = h / s
        cr = 0.95 if r_hs <= 5.0 else (1.05 if r_hs >= 20.0 else 1.00)
        k7_std = K7_BASE_TABLE[n] * cr
        k7_att = ((h + n * s) / (h + s)) ** 0.39 * math.sqrt(n)
        if d <= 0:
            k7 = k7_att
            s_eff = float(s * n)
        elif d <= s:
            k7 = k7_att + (d / s) * (k7_std - k7_att)
            s_eff = float(s * n) - (d / s) * float(s * (n - 1))
        else:
            k7 = k7_std + min(1.0, (d - s) / max(1.0, h - s)) * (0.95 * n - k7_std)
            s_eff = float(s)

    mu = 4.0 * math.pi * 1e-7 * mat["mu_r"]
    if freq <= 0:
        skin_depth = float("inf")
        k9 = 1.00
    else:
        skin_depth = 1000.0 * math.sqrt(2.0 * 1e-6 * mat["rho"] / (2.0 * math.pi * freq * mu))
        lato_mag = max(float(h), s_eff)
        lato_min = min(float(h), s_eff)
        area_eq = lato_min * (lato_mag - lato_min) + math.pi * (lato_min ** 2) / 4.0
        num = (
            (2.0 * (lato_mag - lato_min) + math.pi * lato_min - 2.0 * math.pi * skin_depth)
            * (1.0 - math.exp(-lato_min / (2.0 * skin_depth)))
            + math.pi * lato_min * math.exp(-lato_min / (2.0 * skin_depth))
        )
        den_rad = (2.0 * (lato_mag - lato_min) - math.pi * skin_depth) * (
            1.0 - math.exp(-lato_min / skin_depth)
        ) + math.pi * lato_min
        if den_rad <= 0 or area_eq <= 0:
            k9 = 1.00
        else:
            pref = math.sqrt(2.0 * skin_depth / area_eq)
            k9 = max(0.10, min(1.00, pref * num / math.sqrt(den_rad)))

    k_tot = k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8 * k9
    i_amm = 5.0 * k_tot * (p_1 ** 0.39) * (sez_1 ** 0.5)
    j_amm = i_amm / sez_tot

    return {
        "I_amm": i_amm, "J_amm": j_amm, "sez_tot": sez_tot, "s_eff": s_eff,
        "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "K6": k6,
        "K7": k7, "K8": k8, "K9": k9, "K_tot": k_tot, "skin": skin_depth
    }


st.title("⚡ LOADBANK-MASTER | Progettazione Elettrotermica e Meccanica Busbar")
st.markdown(
    "Solutore alle differenze finite (2D FDM) con verifiche **DIN 43671 / Melsom & Booth (Rev. 2.1)**, "
    "proiezioni isometriche frontali $1:1$ e layout meccanico a tutta altezza (**Sezione Laterale $Y\\text{-}Z$** e **Pianta $X\\text{-}Z$**)."
)

# ==============================================================================
# 3. PANNELLO DI CONTROLLO LATERALE
# ==============================================================================
st.sidebar.header("1. Materiale e Termofluidodinamica (Rev. 2.1)")
mat_scelto = st.sidebar.selectbox("Lega Conduttrice (K1)", list(METALLI_K1.keys()), index=0)
dati_mat = METALLI_K1[mat_scelto]

freq_in = st.sidebar.number_input("Frequenza di rete [Hz] (0 = DC)", min_value=0, max_value=400, value=50, step=10)
tamb_in = st.sidebar.selectbox("Temperatura Ambiente Tamb [°C] (K2)", options=[20, 25, 35, 40, 50, 60, 70], index=3)
dt_in = st.sidebar.selectbox("Sovratemperatura Ammessa DT [K] (K3)", options=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70], index=5)
vent_in = st.sidebar.selectbox("Regime di Ventilazione (K4)", options=list(VENTILAZIONE_K4.keys()), index=0)
sup_in = st.sidebar.selectbox("Stato Superficie (K5)", options=list(SUPERFICIE_K5.keys()), index=0)
forma_in = st.sidebar.selectbox("Geometria Profilato Barre Uscita (K6)", options=list(FORMA_K6.keys()), index=0)

T_regime = tamb_in + dt_in
rho_caldo = dati_mat["rho"] * (1.0 + dati_mat["alpha"] * (T_regime - 20.0))

st.sidebar.caption(
    f"**Regime Termico:** $T_{{\\text{{regime}}}}$ = **{T_regime} °C** | "
    f"$\\rho({T_regime}\\text{{ °C}})$ = **{rho_caldo:.4f} µΩ·m**"
)

st.sidebar.header("2. Piastra Collettrice e Powerlock")
B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1500.0, value=250.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Singola Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
n_piastra = st.sidebar.selectbox("Numero Piastre Accoppiate // (n_P)", options=[1, 2, 3, 4], index=0)
d_piastra = 0.0 if n_piastra == 1 else st.sidebar.number_input("Distanza netta tra piastre d_P [mm]", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

I_totale = st.sidebar.number_input("Corrente Totale Iniettata [A]", min_value=100.0, max_value=40000.0, value=3600.0, step=100.0)
X_col = st.sidebar.number_input("Colonne Powerlock (X)", min_value=1, max_value=10, value=2, step=1)
Y_row = st.sidebar.number_input("Righe Powerlock (Y)", min_value=1, max_value=25, value=9, step=1)
passo_x = st.sidebar.number_input("Passo Orizzontale Colonne [mm]", min_value=20.0, max_value=300.0, value=80.0, step=5.0)
passo_y = st.sidebar.number_input("Passo Verticale Righe [mm]", min_value=20.0, max_value=300.0, value=85.0, step=5.0)
x_offset_pl = st.sidebar.number_input(
    "Quota X Centro Griglia Powerlock [mm]",
    min_value=10.0, max_value=float(B_piastra - 10.0), value=float(B_piastra / 2.0), step=5.0
)
y_offset_pl = st.sidebar.number_input("Quota Y Prima Riga dal Basso [mm]", min_value=10.0, max_value=2500.0, value=160.0, step=10.0)
diam_pl = st.sidebar.number_input("Diametro Impronta Powerlock [mm]", min_value=15.0, max_value=80.0, value=32.0, step=2.0)


def configura_barra_sidebar(id_barra, x_def, y_def, orient_def_idx, lato_def_idx):
    st.sidebar.header(f"3{id_barra}. Ramo {id_barra} in Uscita")
    orient = st.sidebar.selectbox(f"Orientamento Barra {id_barra}", [MODO_ORIZZ, MODO_VERT], index=orient_def_idx, key=f"or_{id_barra}")
    n_b = st.sidebar.selectbox(f"Numero Sbarre in Parallelo ({id_barra})", options=[1, 2, 3, 4, 5, 6], index=0, key=f"n_{id_barra}")
    d_b = 0.0 if n_b == 1 else st.sidebar.number_input(
        f"Distanza netta d tra sbarre ({id_barra}) [mm]",
        min_value=0.0, max_value=150.0, value=10.0, step=1.0, key=f"d_{id_barra}"
    )
    w_b = st.sidebar.number_input(f"Larghezza Barra {id_barra} (h) [mm]", min_value=20.0, max_value=800.0, value=120.0, step=10.0, key=f"w_{id_barra}")
    h_b = st.sidebar.number_input(f"Profondità Sormonto {id_barra} [mm]", min_value=15.0, max_value=400.0, value=60.0, step=5.0, key=f"h_{id_barra}")
    s_b = st.sidebar.number_input(f"Spessore Singola Barra {id_barra} (s) [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0, key=f"s_{id_barra}")
    L_b = st.sidebar.number_input(f"Lunghezza Ramo {id_barra} a sbalzo verso quadro [mm]", min_value=20.0, max_value=1500.0, value=120.0, step=10.0, key=f"L_{id_barra}")

    if orient == MODO_ORIZZ:
        dx_sorm, dy_sorm = w_b, h_b
        lato_piega = st.sidebar.selectbox(
            f"Bordo di stacco piega a sbalzo ({id_barra})",
            ["Bordo Inferiore del Sormonto", "Bordo Superiore del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
        giacitura_ala_costa = False
    else:
        dx_sorm, dy_sorm = h_b, w_b
        lato_piega = st.sidebar.selectbox(
            f"Bordo di stacco piega a sbalzo ({id_barra})",
            ["Bordo Sinistro del Sormonto", "Bordo Destro del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
        giacitura_ala_costa = True

    if dx_sorm > B_piastra:
        st.error(f"Errore geometrico Barra {id_barra}: La larghezza impronta ({dx_sorm:.0f} mm) supera la larghezza piastra ({B_piastra:.0f} mm).")
        st.stop()
    if dy_sorm > H_piastra:
        st.error(f"Errore geometrico Barra {id_barra}: L'altezza impronta ({dy_sorm:.0f} mm) supera l'altezza piastra ({H_piastra:.0f} mm).")
        st.stop()

    min_x = dx_sorm / 2.0
    max_x = B_piastra - dx_sorm / 2.0
    min_y = dy_sorm / 2.0
    max_y = H_piastra - dy_sorm / 2.0

    if min_x >= max_x:
        x_sorm = float(B_piastra / 2.0)
        st.sidebar.info(f"Quota X Centro Barra {id_barra}: **{x_sorm:.1f} mm** (centrata a tutta larghezza)")
    else:
        x_sorm = st.sidebar.slider(
            f"Quota X Centro Sormonto Barra {id_barra} [mm]",
            min_value=float(min_x), max_value=float(max_x),
            value=float(min(max(x_def, min_x), max_x)), step=5.0, key=f"xs_{id_barra}"
        )

    if min_y >= max_y:
        y_sorm = float(H_piastra / 2.0)
        st.sidebar.info(f"Quota Y Centro Barra {id_barra}: **{y_sorm:.1f} mm** (centrata a tutta altezza)")
    else:
        y_sorm = st.sidebar.slider(
            f"Quota Y Centro Sormonto Barra {id_barra} [mm]",
            min_value=float(min_y), max_value=float(max_y),
            value=float(min(max(y_def, min_y), max_y)), step=5.0, key=f"ys_{id_barra}"
        )

    if orient == MODO_ORIZZ:
        x_ala = x_sorm
        y_ala = (y_sorm - dy_sorm / 2.0 + s_b / 2.0) if "Inferiore" in lato_piega else (y_sorm + dy_sorm / 2.0 - s_b / 2.0)
    else:
        y_ala = y_sorm
        x_ala = (x_sorm - dx_sorm / 2.0 + s_b / 2.0) if "Sinistro" in lato_piega else (x_sorm + dx_sorm / 2.0 - s_b / 2.0)

    return {
        "id": id_barra, "orient": orient, "n": n_b, "d": d_b,
        "w": w_b, "h": h_b, "s": s_b, "L": L_b,
        "dx": dx_sorm, "dy": dy_sorm, "x_sorm": x_sorm, "y_sorm": y_sorm,
        "x_ala": x_ala, "y_ala": y_ala, "lato_piega": lato_piega,
        "costa": giacitura_ala_costa
    }


cfg_A = configura_barra_sidebar("A", x_def=B_piastra / 2.0, y_def=60.0, orient_def_idx=0, lato_def_idx=0)
cfg_B = configura_barra_sidebar("B", x_def=B_piastra / 2.0, y_def=500.0, orient_def_idx=0, lato_def_idx=1)

st.sidebar.header("4. Mutuo Accoppiamento tra Ramo A e Ramo B")
accoppia_AB = st.sidebar.checkbox(
    "Accoppiamento termomagnetico A+B (se parallele)",
    value=True,
    help="Valuta il declassamento (K7, K8, K9) come coppia accoppiata alla distanza netta tra le due uscite."
)

# --- VERIFICHE GEOMETRICHE E COLLISIONE 2D ---
x_min_pl = x_offset_pl - ((X_col - 1) * passo_x) / 2.0 - diam_pl / 2.0
x_max_pl = x_offset_pl + ((X_col - 1) * passo_x) / 2.0 + diam_pl / 2.0
y_min_pl = y_offset_pl - diam_pl / 2.0
y_max_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

errori_geom = []
if x_min_pl < 0 or x_max_pl > B_piastra:
    errori_geom.append(f"La griglia Powerlock (X da {x_min_pl:.1f} a {x_max_pl:.1f} mm) deborda dalla larghezza piastra (0 - {B_piastra:.1f} mm).")
if y_min_pl < 0 or y_max_pl > H_piastra:
    errori_geom.append(f"La griglia Powerlock (Y da {y_min_pl:.1f} a {y_max_pl:.1f} mm) deborda dall'altezza piastra (0 - {H_piastra:.1f} mm).")

xA1, xA2 = cfg_A["x_sorm"] - cfg_A["dx"] / 2.0, cfg_A["x_sorm"] + cfg_A["dx"] / 2.0
yA1, yA2 = cfg_A["y_sorm"] - cfg_A["dy"] / 2.0, cfg_A["y_sorm"] + cfg_A["dy"] / 2.0
xB1, xB2 = cfg_B["x_sorm"] - cfg_B["dx"] / 2.0, cfg_B["x_sorm"] + cfg_B["dx"] / 2.0
yB1, yB2 = cfg_B["y_sorm"] - cfg_B["dy"] / 2.0, cfg_B["y_sorm"] + cfg_B["dy"] / 2.0

if (xA1 < xB2) and (xA2 > xB1) and (yA1 < yB2) and (yA2 > yB1):
    errori_geom.append("Collisione meccanica 2D: Le impronte di sormonto della Barra A e della Barra B si sovrappongono sulla piastra!")

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()

# --- CALCOLO PORTATE AMMISSIBILI MELSOM & BOOTH (REV. 2.1) ---
forma_normale = list(FORMA_K6.keys())[0]
mb_piastra = calcola_portata_accoppiata(
    B_piastra, s_piastra, n_piastra, d_piastra, True,
    mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_normale
)

if accoppia_AB and cfg_A["n"] == 1 and cfg_B["n"] == 1 and cfg_A["orient"] == cfg_B["orient"]:
    if cfg_A["orient"] == MODO_ORIZZ:
        d_netta_AB = max(0.0, abs(cfg_B["y_ala"] - cfg_A["y_ala"]) - 0.5 * (cfg_A["s"] + cfg_B["s"]))
    else:
        d_netta_AB = max(0.0, abs(cfg_B["x_ala"] - cfg_A["x_ala"]) - 0.5 * (cfg_A["s"] + cfg_B["s"]))

    mb_coppia_A = calcola_portata_accoppiata(
        cfg_A["w"], cfg_A["s"], 2, d_netta_AB, cfg_A["costa"],
        mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in
    )
    mb_coppia_B = calcola_portata_accoppiata(
        cfg_B["w"], cfg_B["s"], 2, d_netta_AB, cfg_B["costa"],
        mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in
    )
    mb_A = dict(mb_coppia_A, I_amm=mb_coppia_A["I_amm"] / 2.0, sez_tot=cfg_A["w"] * cfg_A["s"], nota_acc=f"Coppia A+B (d={d_netta_AB:.0f} mm)")
    mb_B = dict(mb_coppia_B, I_amm=mb_coppia_B["I_amm"] / 2.0, sez_tot=cfg_B["w"] * cfg_B["s"], nota_acc=f"Coppia A+B (d={d_netta_AB:.0f} mm)")
else:
    mb_A = calcola_portata_accoppiata(
        cfg_A["w"], cfg_A["s"], cfg_A["n"], cfg_A["d"], cfg_A["costa"],
        mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in
    )
    mb_A["nota_acc"] = f"Fascio n={cfg_A['n']} (d={cfg_A['d']:.0f} mm)"
    mb_B = calcola_portata_accoppiata(
        cfg_B["w"], cfg_B["s"], cfg_B["n"], cfg_B["d"], cfg_B["costa"],
        mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in
    )
    mb_B["nota_acc"] = f"Fascio n={cfg_B['n']} (d={cfg_B['d']:.0f} mm)"


# ==============================================================================
# 4. SOLUTORE ELETTRODINAMICO 2D FDM
# ==============================================================================
@st.cache_data(show_spinner=False)
def risolvi_busbar_2d(
    B, H, s_eff_p, rho_uohm_m, I_tot, n_col, n_row, px, py, x0_center_pl, y0_pl, d_pl,
    xA1, xA2, yA1, yA2, wA, hA, sA, nA, k9A, LA,
    xB1, xB2, yB1, yB2, wB, hB, sB, nB, k9B, LB
):
    rho = rho_uohm_m * 1e-3
    sigma = 1.0 / rho

    mesh_sz = 4.0 if (B * H) <= 600000 else 6.0
    nx = max(14, int(round(B / mesh_sz)))
    ny = max(24, int(round(H / mesh_sz)))
    dx = B / nx
    dy = H / ny
    x_c = (np.arange(nx) + 0.5) * dx
    y_c = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x_c, y_c)

    Gx = sigma * (s_eff_p * dy) / dx
    Gy = sigma * (s_eff_p * dx) / dy

    N = nx * ny
    idx_grid = np.arange(N).reshape((ny, nx))
    diag_main = np.zeros(N, dtype=np.float64)

    k_left = idx_grid[:, :-1].ravel()
    k_right = idx_grid[:, 1:].ravel()
    np.add.at(diag_main, k_left, Gx)
    np.add.at(diag_main, k_right, Gx)
    k_bot = idx_grid[:-1, :].ravel()
    k_top = idx_grid[1:, :].ravel()
    np.add.at(diag_main, k_bot, Gy)
    np.add.at(diag_main, k_top, Gy)

    I_vec = np.zeros(N, dtype=np.float64)
    x_start_pl = x0_center_pl - ((n_col - 1) * px) / 2.0
    I_sing = I_tot / (n_col * n_row)
    r_pl = d_pl / 2.0
    coords_pl = []
    for r in range(n_row):
        yp = y0_pl + r * py
        for c in range(n_col):
            xp = x_start_pl + c * px
            coords_pl.append((xp, yp))
            dist2 = (X - xp) ** 2 + (Y - yp) ** 2
            ind = np.flatnonzero(dist2 <= r_pl ** 2)
            if len(ind) == 0:
                ind = np.array([np.argmin(dist2)])
            I_vec[ind] += I_sing / len(ind)

    mask_A = (X >= xA1) & (X <= xA2) & (Y >= yA1) & (Y <= yA2)
    mask_B = (X >= xB1) & (X <= xB2) & (Y >= yB1) & (Y <= yB2)
    ind_A = np.flatnonzero(mask_A)
    ind_B = np.flatnonzero(mask_B)

    if len(ind_A) == 0:
        ind_A = np.array([np.argmin((X - 0.5 * (xA1 + xA2)) ** 2 + (Y - 0.5 * (yA1 + yA2)) ** 2)])
    if len(ind_B) == 0:
        ind_B = np.array([np.argmin((X - 0.5 * (xB1 + xB2)) ** 2 + (Y - 0.5 * (yB1 + yB2)) ** 2)])

    R_att_A = rho * (0.5 * (s_eff_p + sA)) / (wA * hA * nA)
    R_sorm_A = rho * hA / (3.0 * wA * sA * nA * (k9A ** 2))
    R_A = R_att_A + R_sorm_A + (rho * LA) / (wA * sA * nA * (k9A ** 2))

    R_att_B = rho * (0.5 * (s_eff_p + sB)) / (wB * hB * nB)
    R_sorm_B = rho * hB / (3.0 * wB * sB * nB * (k9B ** 2))
    R_B = R_att_B + R_sorm_B + (rho * LB) / (wB * sB * nB * (k9B ** 2))

    g_A = (1.0 / R_A) / len(ind_A)
    g_B = (1.0 / R_B) / len(ind_B)
    diag_main[ind_A] += g_A
    diag_main[ind_B] += g_B

    rows = np.concatenate([np.arange(N), k_left, k_right, k_bot, k_top])
    cols = np.concatenate([np.arange(N), k_right, k_left, k_top, k_bot])
    vals = np.concatenate([
        diag_main,
        np.full(len(k_left), -Gx),
        np.full(len(k_left), -Gx),
        np.full(len(k_bot), -Gy),
        np.full(len(k_bot), -Gy)
    ])
    G_csr = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    V = spla.spsolve(G_csr, I_vec)

    I_A = float(np.sum(V[ind_A] * g_A))
    I_B = float(np.sum(V[ind_B] * g_B))
    V_2d = V.reshape((ny, nx))
    err_perc = abs((I_A + I_B) - I_tot) / I_tot * 100.0

    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)
    profilo_Iy = np.sum(Jy, axis=1) * dx * s_eff_p

    P_tot = float(I_vec @ V)
    P_ramo_A = (I_A ** 2) * R_A
    P_ramo_B = (I_B ** 2) * R_B
    P_piastra = max(0.0, P_tot - P_ramo_A - P_ramo_B)

    return {
        "x": x_c, "y": y_c, "X": X, "Y": Y,
        "V_mV": V_2d * 1e3, "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "I_A": I_A, "I_B": I_B, "err_perc": err_perc,
        "profilo_Iy": profilo_Iy, "coords_pl": coords_pl,
        "P_piastra": P_piastra, "P_A": P_ramo_A, "P_B": P_ramo_B,
        "R_A_uohm": R_A * 1e6, "R_B_uohm": R_B * 1e6
    }


s_cond_piastra = s_piastra * n_piastra * (mb_piastra["K9"] ** 2)

res = risolvi_busbar_2d(
    B_piastra, H_piastra, s_cond_piastra, rho_caldo, I_totale,
    X_col, Y_row, passo_x, passo_y, x_offset_pl, y_offset_pl, diam_pl,
    xA1, xA2, yA1, yA2, cfg_A["w"], cfg_A["h"], cfg_A["s"], cfg_A["n"], mb_A["K9"], cfg_A["L"],
    xB1, xB2, yB1, yB2, cfg_B["w"], cfg_B["h"], cfg_B["s"], cfg_B["n"], mb_B["K9"], cfg_B["L"]
)

# ==============================================================================
# 5. SCHEDE KPI E VALORI STRATEGICI
# ==============================================================================
p_A = res["I_A"] / I_totale * 100.0
p_B = res["I_B"] / I_totale * 100.0
sez_A = mb_A["sez_tot"]
sez_B = mb_B["sez_tot"]
sez_piastra = mb_piastra["sez_tot"]

J_L_A = res["I_A"] / sez_A
J_L_B = res["I_B"] / sez_B
I_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"])))
J_sez_piastra_max = I_sez_piastra_max / sez_piastra
J_99_piastra = float(np.percentile(res["J_mag"], 99))
P_tot = res["P_piastra"] + res["P_A"] + res["P_B"]

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label=f"Ramo A (Amm: {mb_A['I_amm']:.0f} A | K7={mb_A['K7']:.2f})",
    value=f"{res['I_A']:.0f} A ({p_A:.1f}%)",
    delta=f"J = {J_L_A:.2f} A/mm² (J_amm = {mb_A['J_amm']:.2f} A/mm²)",
    delta_color="inverse" if res["I_A"] > mb_A["I_amm"] else "normal"
)
col2.metric(
    label=f"Ramo B (Amm: {mb_B['I_amm']:.0f} A | K7={mb_B['K7']:.2f})",
    value=f"{res['I_B']:.0f} A ({p_B:.1f}%)",
    delta=f"J = {J_L_B:.2f} A/mm² (J_amm = {mb_B['J_amm']:.2f} A/mm²)",
    delta_color="inverse" if res["I_B"] > mb_B["I_amm"] else "normal"
)
col3.metric(
    label=f"Piastra (Amm: {mb_piastra['I_amm']:.0f} A | K7={mb_piastra['K7']:.2f})",
    value=f"{I_sez_piastra_max:.0f} A ({J_sez_piastra_max:.2f} A/mm²)",
    delta=f"J_amm Piastra = {mb_piastra['J_amm']:.2f} A/mm²",
    delta_color="inverse" if I_sez_piastra_max > mb_piastra["I_amm"] else "normal"
)
col4.metric(
    label=f"Potenza Dissipata (a {T_regime} °C)",
    value=f"{P_tot:.1f} watt",
    delta=f"Piastra: {res['P_piastra']:.1f} watt ({P_tot/1000.0:.2f} chilowatt tot)",
    delta_color="off"
)

st.markdown("---")

# ==============================================================================
# 6. DASHBOARD GRAFICA A 5 VISTE (ISOMETRIA 1:1 SU 1 E 2, TUTTA ALTEZZA SU 3, 4, 5)
# ==============================================================================
fig = plt.figure(figsize=(26, 10))
gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.95, 0.85, 1.05])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])


def disegna_impronta_e_piega(ax, cfg, perc, colore_bordo):
    x0 = cfg["x_sorm"] - cfg["dx"] / 2.0
    y0 = cfg["y_sorm"] - cfg["dy"] / 2.0
    ax.add_patch(patches.Rectangle((x0, y0), cfg["dx"], cfg["dy"], edgecolor=colore_bordo, facecolor='none', linewidth=2.2, linestyle='--'))
    if cfg["orient"] == MODO_ORIZZ:
        y_p = y0 if "Inferiore" in cfg["lato_piega"] else (y0 + cfg["dy"])
        ax.plot([x0, x0 + cfg["dx"]], [y_p, y_p], color='#00ff00', linewidth=4.5)
    else:
        x_p = x0 if "Sinistro" in cfg["lato_piega"] else (x0 + cfg["dx"])
        ax.plot([x_p, x_p], [y0, y0 + cfg["dy"]], color='#00ff00', linewidth=4.5)
    rot_tag = "0°" if cfg["orient"] == MODO_ORIZZ else "90°"
    ax.text(
        cfg["x_sorm"], cfg["y_sorm"], f"{cfg['id']} ({cfg['n']}x, {rot_tag})\n{perc:.1f}%",
        color='white', weight='bold', fontsize=9.5, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.80)
    )


# 1. MAPPA DENSITÀ |J| (ISOMETRICA 1:1 CON adjustable='box' PER CERCHI POWERLOCK PERFETTI)
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(mb_piastra["J_amm"] * 1.3, J_99_piastra))
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Densità |J| [A/mm²]", fontsize=11, weight='bold')
cbar1.ax.tick_params(labelsize=10)
ax1.streamplot(res["x"], res["y"], res["Jx"], res["Jy"], color='cyan', linewidth=0.7, density=0.85, arrowsize=0.7)
for (xp, yp) in res["coords_pl"]:
    ax1.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2))
disegna_impronta_e_piega(ax1, cfg_A, p_A, 'cyan')
disegna_impronta_e_piega(ax1, cfg_B, p_B, 'cyan')
ax1.set_title("1. Densità |J| e Piega (Verde)", fontsize=12, weight='bold')
ax1.set_xlabel("Larghezza X [mm]", fontsize=11, weight='bold')
ax1.set_ylabel("Quota Y [mm]", fontsize=11, weight='bold')
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')
ax1.tick_params(labelsize=10)

# 2. MAPPA POTENZIALE ELETTRICO V [mV] (ISOMETRICA 1:1)
im2 = ax2.pcolormesh(res["X"], res["Y"], res["V_mV"], shading='auto', cmap='viridis')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Potenziale [mV]", fontsize=11, weight='bold')
cbar2.ax.tick_params(labelsize=10)
for (xp, yp) in res["coords_pl"]:
    ax2.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7))
disegna_impronta_e_piega(ax2, cfg_A, p_A, 'red')
disegna_impronta_e_piega(ax2, cfg_B, p_B, 'red')
ax2.set_title("2. Potenziale Elettrico [mV]", fontsize=12, weight='bold')
ax2.set_xlabel("Larghezza X [mm]", fontsize=11, weight='bold')
ax2.set_ylabel("Quota Y [mm]", fontsize=11, weight='bold')
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')
ax2.tick_params(labelsize=10)

# 3. FLUSSO ASSIALE VERTICALE |Iy| (A TUTTA ALTEZZA)
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.4, label="|Iy(y)| in piastra")
ax3.axhline(cfg_A["y_sorm"], color='blue', linestyle='--', linewidth=1.4, label=f"Centro A ({cfg_A['y_sorm']:.0f} mm)")
ax3.axhline(cfg_B["y_sorm"], color='green', linestyle='--', linewidth=1.4, label=f"Centro B ({cfg_B['y_sorm']:.0f} mm)")
ax3.axvline(mb_piastra["I_amm"], color='red', linestyle=':', linewidth=2.0, label=f"I_amm ({mb_piastra['I_amm']:.0f} A)")
ax3.fill_betweenx(res["y"], 0, np.abs(res["profilo_Iy"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Flusso Assiale |Iy| [A]", fontsize=12, weight='bold')
ax3.set_xlabel("Corrente Assiale [A]", fontsize=11, weight='bold')
ax3.set_ylabel("Quota Y [mm]", fontsize=11, weight='bold')
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=9.5)
ax3.tick_params(labelsize=10)

# 4. PROIEZIONE LATERALE Y-Z (A TUTTA ALTEZZA)
L_DRAW_MAX = 150.0
prof_pl_disegno = diam_pl * 1.4
spess_tot_p = n_piastra * s_piastra + (n_piastra - 1) * d_piastra

for idx_p in range(n_piastra):
    z0_p = idx_p * (s_piastra + d_piastra)
    ax4.add_patch(patches.Rectangle((z0_p, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))

for cfg, colore, pc, corr_ramo in ((cfg_A, 'blue', p_A, res["I_A"]), (cfg_B, 'green', p_B, res["I_B"])):
    L_dis = min(cfg["L"], L_DRAW_MAX)
    y_bot = cfg["y_sorm"] - cfg["dy"] / 2.0
    for idx_b in range(cfg["n"]):
        off_b = idx_b * (cfg["s"] + cfg["d"])
        ax4.add_patch(patches.Rectangle((spess_tot_p + off_b, y_bot), cfg["s"], cfg["dy"], facecolor='#d9822b', edgecolor='black', linewidth=1.2))
        if cfg["orient"] == MODO_ORIZZ:
            ax4.add_patch(patches.Rectangle((spess_tot_p + cfg["s"] + off_b, cfg["y_ala"] - cfg["s"] / 2.0), L_dis, cfg["s"], facecolor='#e68a00', edgecolor='black', linewidth=1.1))
        else:
            ax4.add_patch(patches.Rectangle((spess_tot_p + cfg["s"] + off_b, y_bot), L_dis, cfg["dy"], facecolor='#e68a00', edgecolor='black', linewidth=1.1, alpha=0.6, hatch='//'))

    y_testo = cfg["y_ala"] if cfg["orient"] == MODO_ORIZZ else cfg["y_sorm"]
    ax4.text(
        spess_tot_p + cfg["s"] + L_dis + 10, y_testo,
        f"{cfg['id']} ({cfg['n']}x): {pc:.1f}%\n({corr_ramo:.0f} A)",
        color=colore, fontsize=9.5, weight='bold', va='center'
    )

y_pl_unici = sorted(set(yp for (_, yp) in res["coords_pl"]))
for yp in y_pl_unici:
    ax4.add_patch(patches.Rectangle((-prof_pl_disegno, yp - diam_pl / 2.0), prof_pl_disegno, diam_pl, facecolor='#333333', edgecolor='black', linewidth=1.0))

ax4.set_title("4. Sezione Laterale Y-Z", fontsize=12, weight='bold')
ax4.set_xlabel("Profondità Z [mm]", fontsize=11, weight='bold')
ax4.set_ylabel("Quota Y [mm]", fontsize=11, weight='bold')
ax4.set_xlim(-prof_pl_disegno - 30, spess_tot_p + L_DRAW_MAX + 160)
ax4.set_ylim(0, H_piastra)
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.tick_params(labelsize=10)

# 5. VISTA DALL'ALTO X-Z (PIANTA MECCANICA A TUTTA ALTEZZA)
for idx_p in range(n_piastra):
    z0_p = idx_p * (s_piastra + d_piastra)
    ax5.add_patch(patches.Rectangle((0, z0_p), B_piastra, s_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))

x_pl_unici = sorted(set(xp for (xp, _) in res["coords_pl"]))
for xp in x_pl_unici:
    ax5.add_patch(patches.Rectangle((xp - diam_pl / 2.0, -prof_pl_disegno), diam_pl, prof_pl_disegno, facecolor='#333333', edgecolor='black', linewidth=1.0))

for cfg, colore, pc, corr_ramo in ((cfg_A, 'blue', p_A, res["I_A"]), (cfg_B, 'green', p_B, res["I_B"])):
    L_dis = min(cfg["L"], L_DRAW_MAX)
    x_sorm_start = cfg["x_sorm"] - cfg["dx"] / 2.0

    for idx_b in range(cfg["n"]):
        off_z = idx_b * (cfg["s"] + cfg["d"])

        ax5.add_patch(patches.Rectangle(
            (x_sorm_start, spess_tot_p + off_z), cfg["dx"], cfg["s"],
            facecolor='#d9822b', edgecolor='black', linewidth=1.2, hatch='..'
        ))

        if cfg["orient"] == MODO_ORIZZ:
            ax5.add_patch(patches.Rectangle(
                (x_sorm_start, spess_tot_p + cfg["s"] + off_z), cfg["dx"], L_dis,
                facecolor='#e68a00', edgecolor='black', linewidth=1.1, alpha=0.75
            ))
        else:
            x_ala_start = cfg["x_ala"] - cfg["s"] / 2.0
            ax5.add_patch(patches.Rectangle(
                (x_ala_start, spess_tot_p + cfg["s"] + off_z), cfg["s"], L_dis,
                facecolor='#e68a00', edgecolor='black', linewidth=1.1, hatch='//'
            ))

    x_testo = cfg["x_sorm"] if cfg["orient"] == MODO_ORIZZ else cfg["x_ala"]
    ax5.text(
        x_testo, spess_tot_p + L_dis + 10,
        f"{cfg['id']} ({cfg['n']}x)\n{corr_ramo:.0f} A ({pc:.1f}%)",
        color=colore, fontsize=9.5, weight='bold', ha='center', va='bottom'
    )

ax5.axvline(0, color='gray', linestyle=':', linewidth=0.8)
ax5.axvline(B_piastra, color='gray', linestyle=':', linewidth=0.8)
ax5.set_title("5. Pianta Meccanica X-Z (Con Sormonto)", fontsize=12, weight='bold')
ax5.set_xlabel("Larghezza X [mm]", fontsize=11, weight='bold')
ax5.set_ylabel("Profondità Z [mm]", fontsize=11, weight='bold')
ax5.set_xlim(-20, B_piastra + 20)
ax5.set_ylim(-prof_pl_disegno - 30, spess_tot_p + L_DRAW_MAX + 150)
ax5.grid(True, linestyle=':', alpha=0.5)
ax5.tick_params(labelsize=10)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ==============================================================================
# 7. REPORT ANALITICO DEI COEFFICIENTI K1..K9 E CONFORMITÀ DFM
# ==============================================================================
st.subheader("📋 Report Analitico Coefficienti Melsom & Booth (K1..K9) e Verifiche")
st.table(pd.DataFrame({
    "Elemento Costruttivo": [
        f"Piastra Collettrice ({n_piastra}x {B_piastra:.0f}x{s_piastra:.0f} mm)",
        f"Ramo Barra A ({cfg_A['n']}x {cfg_A['w']:.0f}x{cfg_A['s']:.0f} mm - {mb_A['nota_acc']})",
        f"Ramo Barra B ({cfg_B['n']}x {cfg_B['w']:.0f}x{cfg_B['s']:.0f} mm - {mb_B['nota_acc']})",
        "Bilancio Energetico e Termico (a regime operativo)"
    ],
    "Corrente e Densità Operativa (2D FDM)": [
        f"I_max = {I_sez_piastra_max:.0f} A  |  J_max = {J_sez_piastra_max:.2f} A/mm² (Sez. {sez_piastra:.0f} mm²)",
        f"I_A = {res['I_A']:.0f} A ({p_A:.1f} %)  |  J_A = {J_L_A:.2f} A/mm² (Sez. {sez_A:.0f} mm²)",
        f"I_B = {res['I_B']:.0f} A ({p_B:.1f} %)  |  J_B = {J_L_B:.2f} A/mm² (Sez. {sez_B:.0f} mm²)",
        f"Piastra: {res['P_piastra']:.1f} watt | Rami: {(res['P_A'] + res['P_B']):.1f} watt | Totale: {P_tot:.1f} watt ({P_tot/1000.0:.2f} chilowatt)"
    ],
    "Catena Coefficienti Melsom & Booth (K1..K9)": [
        f"K1={mb_piastra['K1']:.2f} | K2={mb_piastra['K2']:.2f} | K3={mb_piastra['K3']:.2f} | K4={mb_piastra['K4']:.2f} | K6={mb_piastra['K6']:.2f} | K7={mb_piastra['K7']:.3f} | K8={mb_piastra['K8']:.2f} | K9={mb_piastra['K9']:.3f} (s_eff={mb_piastra['s_eff']:.1f} mm)",
        f"K1={mb_A['K1']:.2f} | K2={mb_A['K2']:.2f} | K3={mb_A['K3']:.2f} | K4={mb_A['K4']:.2f} | K6={mb_A['K6']:.2f} | K7={mb_A['K7']:.3f} | K8={mb_A['K8']:.2f} | K9={mb_A['K9']:.3f} (s_eff={mb_A['s_eff']:.1f} mm)",
        f"K1={mb_B['K1']:.2f} | K2={mb_B['K2']:.2f} | K3={mb_B['K3']:.2f} | K4={mb_B['K4']:.2f} | K6={mb_B['K6']:.2f} | K7={mb_B['K7']:.3f} | K8={mb_B['K8']:.2f} | K9={mb_B['K9']:.3f} (s_eff={mb_B['s_eff']:.1f} mm)",
        f"R_A = {res['R_A_uohm']:.2f} µΩ (L_A={cfg_A['L']:.0f} mm)  |  R_B = {res['R_B_uohm']:.2f} µΩ (L_B={cfg_B['L']:.0f} mm)"
    ],
    "Portata Limite ed Esito Elettrotermico": [
        f"I_amm = {mb_piastra['I_amm']:.0f} A (J_amm = {mb_piastra['J_amm']:.2f} A/mm²) -> " + ("CONFORME" if I_sez_piastra_max <= mb_piastra["I_amm"] else "SOVRACCARICO"),
        f"I_amm = {mb_A['I_amm']:.0f} A (J_amm = {mb_A['J_amm']:.2f} A/mm²) -> " + ("CONFORME" if res["I_A"] <= mb_A["I_amm"] else "SOVRACCARICO"),
        f"I_amm = {mb_B['I_amm']:.0f} A (J_amm = {mb_B['J_amm']:.2f} A/mm²) -> " + ("CONFORME" if res["I_B"] <= mb_B["I_amm"] else "SOVRACCARICO"),
        f"Errore numerico Kirchhoff: {res['err_perc']:.6f} %"
    ]
}))
