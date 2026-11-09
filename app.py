import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

# ==============================================================================
# CONFIGURAZIONE INTERFACCIA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="LOADBANK-MASTER | Busbar 2D FDM + Verifiche Rev. 2.1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. DATABASE MATERIALI E COEFFICIENTI (DIN 43671 / MELSOM & BOOTH / CEI EN 60865-1)
# ==============================================================================
METALLI_K1 = {
    "Rame elettrolitico ETP CW004A (rho=0,0178 uOhm m)": {
        "K1": 1.26, "rho": 0.0178, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0,
        "alpha": 0.00393, "alpha_L": 16.5e-6, "rp02": 180.0
    },
    "Rame elettrolitico OFHC CW008A (rho=0,0176 uOhm m)": {
        "K1": 1.27, "rho": 0.0176, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0,
        "alpha": 0.00393, "alpha_L": 16.5e-6, "rp02": 180.0
    },
    "Lega CuCr1Zr CW106C invecchiata (rho=0,0223 uOhm m)": {
        "K1": 0.17778 * (0.0223**-0.487), "rho": 0.0223, "mu_r": 1.000000, "dens": 8890.0, "csp": 385.0,
        "alpha": 0.00320, "alpha_L": 17.0e-6, "rp02": 300.0
    },
    "Alluminio purezza elettrica EN-AW 1350A (rho=0,028 uOhm m)": {
        "K1": 1.02, "rho": 0.0280, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0,
        "alpha": 0.00403, "alpha_L": 23.0e-6, "rp02": 70.0
    },
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {
        "K1": 1.00, "rho": 0.0290, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0,
        "alpha": 0.00403, "alpha_L": 23.0e-6, "rp02": 70.0
    },
    "Lega d'alluminio EN-AW 6101B T6 (rho=0,0325 uOhm m)": {
        "K1": 0.94, "rho": 0.0325, "mu_r": 1.000000, "dens": 2700.0, "csp": 880.0,
        "alpha": 0.00360, "alpha_L": 23.0e-6, "rp02": 160.0
    },
    "Acciaio Inox AISI 304 EN 1.4301 (rho=0,72 uOhm m)": {
        "K1": 0.17778 * (0.72**-0.487), "rho": 0.7200, "mu_r": 1.020000, "dens": 8000.0, "csp": 500.0,
        "alpha": 0.00100, "alpha_L": 16.0e-6, "rp02": 210.0
    },
    "Lega CuCr1Zr [Valore Tabella Excel Storica] (rho=0,028 uOhm m)": {
        "K1": 0.17778 * (0.028**-0.487), "rho": 0.0280, "mu_r": 1.000000, "dens": 8910.0, "csp": 450.0,
        "alpha": 0.00300, "alpha_L": 17.0e-6, "rp02": 280.0
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
# 2. MOTORE COMPLETO ELETTROTERMICO, CORTOCIRCUITO ED ELETTRODINAMICA (REV. 2.1)
# ==============================================================================
def esegui_calcolo_completo(
    h, s, n, d, giacitura_costa, metallo_str, freq, t_amb, dt,
    vent_str, sup_str, forma_str, lungh_tratta, t_lim_cw, ik_impianto_ka
):
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
            k9_raw = pref * num / math.sqrt(den_rad)
            k9 = max(0.10, min(1.00, k9_raw))

    k_tot = k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8 * k9
    i_amm = 5.0 * k_tot * (p_1 ** 0.39) * (sez_1 ** 0.5)
    j_amm = i_amm / sez_tot
    t_regime = t_amb + dt

    # Verifica Cortocircuito Adiabatico CEI EN 60865-1 / IEC 60949 vs Modello Excel
    dt_cw = max(0.0, t_lim_cw - t_regime)
    if dt_cw <= 0:
        kcw_iec = 0.0
        icw_1s_iec = 0.0
        icw_1s_excel = 0.0
    else:
        num_ln = 1.0 + mat["alpha"] * (t_lim_cw - 20.0)
        den_ln = 1.0 + mat["alpha"] * (t_regime - 20.0)
        kcw_iec = math.sqrt(
            (mat["csp"] * mat["dens"]) / (mat["rho"] * mat["alpha"] * 1e6) * math.log(num_ln / den_ln)
        )
        icw_1s_iec = (kcw_iec * sez_tot) / 1000.0

        kcw1_ex = math.sqrt((mat["csp"] / 385.0) * (mat["dens"] / 8890.0) * (0.0176 / mat["rho"]))
        kcw2_ex = math.sqrt(385.0 * 8890.0 / 0.0176 / 1e6 * dt_cw)
        icw_1s_excel = (kcw1_ex * kcw2_ex * sez_tot) / 1000.0

    i2t_ammissibile = icw_1s_iec ** 2

    # Dissipazione lineare e trifase a portata nominale
    r_dc_20 = mat["rho"] / sez_tot
    r_ac_op = r_dc_20 * (1.0 + mat["alpha"] * (t_regime - 20.0)) / (k9 ** 2)
    p_joule_fase_watt_m = r_ac_op * (i_amm ** 2)
    p_joule_tot_3ph_kw = (3.0 * p_joule_fase_watt_m * lungh_tratta) / 1000.0

    # Dilatazione termica lineare
    t_posa = 20.0
    delta_l_dt_mm = lungh_tratta * 1000.0 * mat["alpha_L"] * dt
    delta_l_tot_posa_mm = lungh_tratta * 1000.0 * mat["alpha_L"] * max(0.0, t_regime - t_posa)

    # Forze elettrodinamiche tra sub-conduttori paralleli e passo spaziatori
    i_picco_ka = 2.0 * math.sqrt(2.0) * ik_impianto_ka
    mu0_2pi = 2.0e-7

    if n > 1 and d > 0:
        i_picco_sub_a = (i_picco_ka * 1000.0) / n
        dist_baricentri_m = (d + s) / 1000.0
        f_sub_nm = mu0_2pi * (i_picco_sub_a ** 2) / dist_baricentri_m
        w_lamina = (h * (s ** 2)) / 6.0 if giacitura_costa else (s * (h ** 2)) / 6.0
        sigma_amm = 0.8 * mat["rp02"]
        l_max_spacers_mm = math.sqrt((12.0 * sigma_amm * w_lamina) / (f_sub_nm / 1000.0))
        l_max_spacers_mm = min(1000.0, max(150.0, l_max_spacers_mm))
    else:
        f_sub_nm = 0.0
        l_max_spacers_mm = 500.0

    return {
        "p_1": p_1, "sez_1": sez_1, "sez_tot": sez_tot, "s_eff": s_eff,
        "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "K6": k6,
        "K7": k7, "K8": k8, "K9": k9, "K_tot": k_tot,
        "skin": skin_depth, "I_amm": i_amm, "J_amm": j_amm, "T_regime": t_regime,
        "Kcw_IEC": kcw_iec, "Icw_1s_IEC": icw_1s_iec, "Icw_1s_Excel": icw_1s_excel, "I2t": i2t_ammissibile,
        "r_ac_op_ohm_m": r_ac_op,
        "P_joule_fase_watt_m": p_joule_fase_watt_m, "P_joule_tot_3ph_kw": p_joule_tot_3ph_kw,
        "Delta_L_DT_mm": delta_l_dt_mm, "Delta_L_posa_mm": delta_l_tot_posa_mm,
        "F_sub_Nm": f_sub_nm, "L_max_spacers_mm": l_max_spacers_mm, "i_picco_kA": i_picco_ka
    }


# ==============================================================================
# 3. PANNELLO DI CONTROLLO LATERALE
# ==============================================================================
st.sidebar.header("1. Materiale, Cortocircuito e Termofluidodinamica")
mat_scelto = st.sidebar.selectbox("Lega Conduttrice (K1)", list(METALLI_K1.keys()), index=0)
dati_mat = METALLI_K1[mat_scelto]

freq_in = st.sidebar.number_input("Frequenza di rete [Hz] (0 = DC)", min_value=0, max_value=400, value=50, step=10)
tamb_in = st.sidebar.selectbox("Temperatura Ambiente Tamb [°C] (K2)", options=[20, 25, 35, 40, 50, 60, 70], index=3)
dt_in = st.sidebar.selectbox("Sovratemperatura Ammessa DT [K] (K3)", options=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70], index=5)
vent_in = st.sidebar.selectbox("Regime di Ventilazione (K4)", options=list(VENTILAZIONE_K4.keys()), index=0)
sup_in = st.sidebar.selectbox("Stato Superficie (K5)", options=list(SUPERFICIE_K5.keys()), index=0)
forma_in = st.sidebar.selectbox("Geometria Profilato Barre Uscita (K6)", options=list(FORMA_K6.keys()), index=0)

lungh_tratta_in = st.sidebar.slider("Lunghezza tratta condotto sbarre [m] (per DeltaL)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
t_lim_cw_in = st.sidebar.selectbox(
    "Temp. limite finale cortocircuito Tlim [°C]",
    options=[140, 160, 180, 200],
    index=0,
    help="140 °C: giunzioni a contatto con isolanti standard | 160-200 °C: sbarre nude in aria (CEI EN 60865-1)"
)
ik_in = st.sidebar.number_input(
    "Corrente di cortocircuito impianto Ik'' [kA RMS]",
    min_value=5.0, max_value=150.0, value=50.0, step=5.0,
    help="Corrente presunta di cortocircuito del quadro per verifica elettrodinamica spaziatori e tenuta Icw"
)

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

st.sidebar.header("3. Configurazione Rami in Uscita")
num_rami = st.sidebar.radio(
    "Numero di rami in uscita verso quadro",
    options=[1, 2],
    format_func=lambda x: "1 Barra/Fascio in Uscita (Solo Ramo A)" if x == 1 else "2 Barre/Fasci in Uscita (Ramo A + Ramo B)",
    index=0
)
attiva_B = (num_rami == 2)


def configura_barra_sidebar(id_barra, x_def, y_def, orient_def_idx, lato_def_idx):
    st.sidebar.subheader(f"3{id_barra}. Ramo {id_barra} in Uscita")
    orient = st.sidebar.selectbox(f"Orientamento Barra {id_barra}", [MODO_ORIZZ, MODO_VERT], index=orient_def_idx, key=f"or_{id_barra}")
    n_b = st.sidebar.selectbox(f"Numero Sbarre in Parallelo // ({id_barra})", options=[1, 2, 3, 4, 5, 6], index=1, key=f"n_{id_barra}")
    d_b = 0.0 if n_b == 1 else st.sidebar.number_input(
        f"Distanza netta d tra sbarre parallele ({id_barra}) [mm]",
        min_value=0.0, max_value=150.0, value=10.0, step=1.0, key=f"d_{id_barra}",
        help="Distanza aria libera tra le facce interne del fascio per verifica effetto termico K7 ed elettrodinamico"
    )
    w_b = st.sidebar.number_input(f"Larghezza Barra {id_barra} (h) [mm]", min_value=20.0, max_value=800.0, value=120.0, step=10.0, key=f"w_{id_barra}")
    h_b = st.sidebar.number_input(f"Profondità Sormonto {id_barra} [mm]", min_value=15.0, max_value=400.0, value=60.0, step=5.0, key=f"h_{id_barra}")
    s_b = st.sidebar.number_input(f"Spessore Singola Barra {id_barra} (s) [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0, key=f"s_{id_barra}")
    L_b = st.sidebar.number_input(f"Lunghezza Ramo {id_barra} a sbalzo su piastra [mm]", min_value=20.0, max_value=1500.0, value=120.0, step=10.0, key=f"L_{id_barra}")

    if orient == MODO_ORIZZ:
        dx_sorm, dy_sorm = w_b, h_b
        lato_piega = st.sidebar.selectbox(
            f"Bordo di stacco piega a sbalzo ({id_barra})",
            ["Bordo Inferiore del Sormonto", "Bordo Superiore del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
        giacitura_ala_costa = False
        posa_str = "Di piatto (lato h orizzontale)"
    else:
        dx_sorm, dy_sorm = h_b, w_b
        lato_piega = st.sidebar.selectbox(
            f"Bordo di stacco piega a sbalzo ({id_barra})",
            ["Bordo Sinistro del Sormonto", "Bordo Destro del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
        giacitura_ala_costa = True
        posa_str = "Di costa (lato h verticale)"

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
        "costa": giacitura_ala_costa, "posa_str": posa_str
    }


cfg_A = configura_barra_sidebar("A", x_def=B_piastra / 2.0, y_def=60.0, orient_def_idx=0, lato_def_idx=0)
cfg_B = configura_barra_sidebar("B", x_def=B_piastra / 2.0, y_def=500.0, orient_def_idx=0, lato_def_idx=1) if attiva_B else None

st.sidebar.markdown("---")
st.sidebar.info(
    "**Ipotesi di Contatto e Accoppiamento:** Le barre sono approssimate come con **appoggio ideale** "
    "sulla piastra collettrice. Gli effetti termici ed elettrodinamici tra sbarre parallele ($K_7, K_8, K_9$, "
    "forze attrattive $F_{\\text{sub}}$) sono valutati internamente a ciascun fascio di sbarre."
)

# ==============================================================================
# 4. INTESTAZIONE, VERIFICHE GEOMETRICHE E CALCOLO REV. 2.1
# ==============================================================================
st.title("⚡ LOADBANK-MASTER | Progettazione Elettrotermica e Meccanica Busbar (2D FDM + Rev. 2.1)")
st.markdown(
    """
    **Quadro Normativo Applicato:**  
    `DIN 43671 / DIN 43670` (Coefficienti correttivi portata permanente) • 
    `Metodo Melsom & Booth / CDA Pub. No. 22` (Equazione base DC e mutuo accoppiamento fascio $K_7$) • 
    `CEI EN 61439-1/2` (Limiti di sovratemperatura $\\Delta T \\le 70\\text{ K}$ e morsetti apparecchiature) • 
    `CEI EN 60865-1 / IEC 60949` (Cortocircuito adiabatico con $\\rho(T)$ variabile e forze elettrodinamiche) • 
    `Solutore 2D FDM` (Mappa densità di corrente $|J|$ e potenziale su piastra con **appoggio ideale** dei rami)
    """
)

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

if attiva_B:
    xB1, xB2 = cfg_B["x_sorm"] - cfg_B["dx"] / 2.0, cfg_B["x_sorm"] + cfg_B["dx"] / 2.0
    yB1, yB2 = cfg_B["y_sorm"] - cfg_B["dy"] / 2.0, cfg_B["y_sorm"] + cfg_B["dy"] / 2.0
    if (xA1 < xB2) and (xA2 > xB1) and (yA1 < yB2) and (yA2 > yB1):
        errori_geom.append("Collisione meccanica 2D: Le impronte di sormonto della Barra A e della Barra B si sovrappongono sulla piastra!")
else:
    xB1 = xB2 = yB1 = yB2 = 0.0

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()

forma_normale = list(FORMA_K6.keys())[0]
mb_piastra = esegui_calcolo_completo(
    B_piastra, s_piastra, n_piastra, d_piastra, True,
    mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_normale,
    H_piastra / 1000.0, t_lim_cw_in, ik_in
)
mb_piastra["nota_acc"] = f"Appoggio ideale - Piastra n={n_piastra} (d={d_piastra:.0f} mm)"

mb_A = esegui_calcolo_completo(
    cfg_A["w"], cfg_A["s"], cfg_A["n"], cfg_A["d"], cfg_A["costa"],
    mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in,
    lungh_tratta_in, t_lim_cw_in, ik_in
)
mb_A["nota_acc"] = f"Appoggio ideale - Fascio n={cfg_A['n']} (d={cfg_A['d']:.0f} mm)"

if attiva_B:
    mb_B = esegui_calcolo_completo(
        cfg_B["w"], cfg_B["s"], cfg_B["n"], cfg_B["d"], cfg_B["costa"],
        mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in,
        lungh_tratta_in, t_lim_cw_in, ik_in
    )
    mb_B["nota_acc"] = f"Appoggio ideale - Fascio n={cfg_B['n']} (d={cfg_B['d']:.0f} mm)"
else:
    mb_B = None


# ==============================================================================
# 5. SOLUTORE ELETTRODINAMICO 2D FDM
# ==============================================================================
@st.cache_data(show_spinner=False)
def risolvi_busbar_2d(
    B, H, s_eff_p, rho_uohm_m, I_tot, n_col, n_row, px, py, x0_center_pl, y0_pl, d_pl,
    xA1, xA2, yA1, yA2, wA, hA, sA, nA, k9A, LA,
    usa_B, xB1, xB2, yB1, yB2, wB, hB, sB, nB, k9B, LB
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
    ind_A = np.flatnonzero(mask_A)
    if len(ind_A) == 0:
        ind_A = np.array([np.argmin((X - 0.5 * (xA1 + xA2)) ** 2 + (Y - 0.5 * (yA1 + yA2)) ** 2)])

    R_att_A = rho * (0.5 * (s_eff_p + sA)) / (wA * hA * nA)
    R_sorm_A = rho * hA / (3.0 * wA * sA * nA * (k9A ** 2))
    R_A = R_att_A + R_sorm_A + (rho * LA) / (wA * sA * nA * (k9A ** 2))

    g_A = (1.0 / R_A) / len(ind_A)
    diag_main[ind_A] += g_A

    if usa_B:
        mask_B = (X >= xB1) & (X <= xB2) & (Y >= yB1) & (Y <= yB2)
        ind_B = np.flatnonzero(mask_B)
        if len(ind_B) == 0:
            ind_B = np.array([np.argmin((X - 0.5 * (xB1 + xB2)) ** 2 + (Y - 0.5 * (yB1 + yB2)) ** 2)])

        R_att_B = rho * (0.5 * (s_eff_p + sB)) / (wB * hB * nB)
        R_sorm_B = rho * hB / (3.0 * wB * sB * nB * (k9B ** 2))
        R_B = R_att_B + R_sorm_B + (rho * LB) / (wB * sB * nB * (k9B ** 2))

        g_B = (1.0 / R_B) / len(ind_B)
        diag_main[ind_B] += g_B
    else:
        ind_B = np.array([], dtype=int)
        R_B = 0.0
        g_B = 0.0

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
    I_B = float(np.sum(V[ind_B] * g_B)) if usa_B else 0.0
    V_2d = V.reshape((ny, nx))
    err_perc = abs((I_A + I_B) - I_tot) / I_tot * 100.0

    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)
    profilo_Iy = np.sum(Jy, axis=1) * dx * s_eff_p

    P_tot = float(I_vec @ V)
    P_ramo_A = (I_A ** 2) * R_A
    P_ramo_B = ((I_B ** 2) * R_B) if usa_B else 0.0
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
    attiva_B,
    xB1 if attiva_B else 0.0, xB2 if attiva_B else 0.0,
    yB1 if attiva_B else 0.0, yB2 if attiva_B else 0.0,
    cfg_B["w"] if attiva_B else 1.0, cfg_B["h"] if attiva_B else 1.0,
    cfg_B["s"] if attiva_B else 1.0, cfg_B["n"] if attiva_B else 1,
    mb_B["K9"] if attiva_B else 1.0, cfg_B["L"] if attiva_B else 1.0
)

# ==============================================================================
# 6. AVVISI DI CONFORMITÀ NORMATIVA (REV. 2.1) E SCHEDE KPI
# ==============================================================================
p_A = res["I_A"] / I_totale * 100.0
p_B = (res["I_B"] / I_totale * 100.0) if attiva_B else 0.0
sez_A = mb_A["sez_tot"]
sez_B = mb_B["sez_tot"] if attiva_B else 0.0
sez_piastra = mb_piastra["sez_tot"]

J_L_A = res["I_A"] / sez_A
J_L_B = (res["I_B"] / sez_B) if attiva_B else 0.0
I_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"])))
J_sez_piastra_max = I_sez_piastra_max / sez_piastra
J_99_piastra = float(np.percentile(res["J_mag"], 99))
P_tot = res["P_piastra"] + res["P_A"] + res["P_B"]

if T_regime > 130:
    st.error(
        f"⚠️ **ALLARME CRITICO ISOLAMENTO**: Temperatura assoluta di regime a {T_regime:.1f} °C "
        f"(Tamb {tamb_in} °C + DT {dt_in} K). Superata la classe termica (130 °C) degli isolatori portanti "
        f"in poliestere SMC/DMC e vetronite G10/FR4!"
    )
elif dt_in > 65 or T_regime > 105:
    st.warning(
        f"⚠️ **AVVISO INTERFACCE APPARECCHIATURE (CEI EN 61439-1 Tab. 6)**: Temperatura assoluta sbarra = {T_regime:.1f} °C "
        f"(Sovratemperatura DeltaT = {dt_in} K). La norma ammette DeltaT <= 70 K sui terminali nudi, ma i costruttori "
        f"di interruttori (ACB/MCCB) prescrivono il declassamento per temperature ai morsetti superiori a 100 - 105 °C."
    )

icw_min_sistema = min(mb_piastra["Icw_1s_IEC"], mb_A["Icw_1s_IEC"], mb_B["Icw_1s_IEC"] if attiva_B else float("inf"))
if icw_min_sistema < ik_in:
    st.error(
        f"⚠️ **VERIFICA CORTOCIRCUITO NON SODDISFATTA**: La tenuta adiabatica minima a 1s del sistema ({icw_min_sistema:.1f} kA) "
        f"è inferiore alla corrente di cortocircuito dell'impianto Ik'' ({ik_in:.1f} kA)!"
    )

j_max_operativa = max(J_sez_piastra_max, J_L_A, J_L_B if attiva_B else 0.0)
if j_max_operativa > 3.0:
    st.warning(
        f"⚠️ **DENSITÀ DI CORRENTE OPERATIVA ELEVATA**: J_max = {j_max_operativa:.2f} A/mm². Ammissibile solo con ventilazione forzata diretta."
    )

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label=f"Ramo A (Amm: {mb_A['I_amm']:.0f} A | K7={mb_A['K7']:.2f})",
    value=f"{res['I_A']:.0f} A ({p_A:.1f}%)",
    delta=f"J = {J_L_A:.2f} A/mm² (J_amm = {mb_A['J_amm']:.2f} A/mm²)",
    delta_color="inverse" if res["I_A"] > mb_A["I_amm"] else "normal"
)
if attiva_B:
    col2.metric(
        label=f"Ramo B (Amm: {mb_B['I_amm']:.0f} A | K7={mb_B['K7']:.2f})",
        value=f"{res['I_B']:.0f} A ({p_B:.1f}%)",
        delta=f"J = {J_L_B:.2f} A/mm² (J_amm = {mb_B['J_amm']:.2f} A/mm²)",
        delta_color="inverse" if res["I_B"] > mb_B["I_amm"] else "normal"
    )
else:
    col2.metric(
        label="Ramo B (Disattivato - Uscita Singola)",
        value="Non presente",
        delta="Appoggio ideale su singola uscita A",
        delta_color="off"
    )

col3.metric(
    label=f"Piastra (Amm: {mb_piastra['I_amm']:.0f} A | Icw={mb_piastra['Icw_1s_IEC']:.1f} kA)",
    value=f"{I_sez_piastra_max:.0f} A ({J_sez_piastra_max:.2f} A/mm²)",
    delta=f"J_amm Piastra = {mb_piastra['J_amm']:.2f} A/mm²",
    delta_color="inverse" if I_sez_piastra_max > mb_piastra["I_amm"] else "normal"
)
col4.metric(
    label=f"Potenza Dissipata Nodo (a {T_regime} °C)",
    value=f"{P_tot:.1f} watt",
    delta=f"Piastra: {res['P_piastra']:.1f} watt ({P_tot/1000.0:.2f} chilowatt tot)",
    delta_color="off"
)

st.markdown("---")

# ==============================================================================
# 7. DASHBOARD GRAFICA 2D FDM E LAYOUT MECCANICO A 5 VISTE
# ==============================================================================
st.subheader("🗺️ Mappa Elettrodinamica 2D FDM e Proiezioni Meccaniche (Piastra + Sormonti)")
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


rami_attivi_plot = [(cfg_A, 'blue', p_A, res["I_A"])]
if attiva_B:
    rami_attivi_plot.append((cfg_B, 'green', p_B, res["I_B"]))

# 1. MAPPA DENSITÀ |J|
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(mb_piastra["J_amm"] * 1.3, J_99_piastra))
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Densità |J| [A/mm²]", fontsize=11, weight='bold')
cbar1.ax.tick_params(labelsize=10)
ax1.streamplot(res["x"], res["y"], res["Jx"], res["Jy"], color='cyan', linewidth=0.7, density=0.85, arrowsize=0.7)
for (xp, yp) in res["coords_pl"]:
    ax1.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2))
disegna_impronta_e_piega(ax1, cfg_A, p_A, 'cyan')
if attiva_B:
    disegna_impronta_e_piega(ax1, cfg_B, p_B, 'cyan')
ax1.set_title("1. Densità |J| e Piega (Verde)", fontsize=12, weight='bold')
ax1.set_xlabel("Larghezza X [mm]", fontsize=11, weight='bold')
ax1.set_ylabel("Quota Y [mm]", fontsize=11, weight='bold')
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')
ax1.tick_params(labelsize=10)

# 2. MAPPA POTENZIALE ELETTRICO V [mV]
im2 = ax2.pcolormesh(res["X"], res["Y"], res["V_mV"], shading='auto', cmap='viridis')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Potenziale [mV]", fontsize=11, weight='bold')
cbar2.ax.tick_params(labelsize=10)
for (xp, yp) in res["coords_pl"]:
    ax2.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7))
disegna_impronta_e_piega(ax2, cfg_A, p_A, 'red')
if attiva_B:
    disegna_impronta_e_piega(ax2, cfg_B, p_B, 'red')
ax2.set_title("2. Potenziale Elettrico [mV]", fontsize=12, weight='bold')
ax2.set_xlabel("Larghezza X [mm]", fontsize=11, weight='bold')
ax2.set_ylabel("Quota Y [mm]", fontsize=11, weight='bold')
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')
ax2.tick_params(labelsize=10)

# 3. FLUSSO ASSIALE VERTICALE |Iy|
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.4, label="|Iy(y)| in piastra")
ax3.axhline(cfg_A["y_sorm"], color='blue', linestyle='--', linewidth=1.4, label=f"Centro A ({cfg_A['y_sorm']:.0f} mm)")
if attiva_B:
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

# 4. PROIEZIONE LATERALE Y-Z
L_DRAW_MAX = 150.0
prof_pl_disegno = diam_pl * 1.4
spess_tot_p = n_piastra * s_piastra + (n_piastra - 1) * d_piastra

for idx_p in range(n_piastra):
    z0_p = idx_p * (s_piastra + d_piastra)
    ax4.add_patch(patches.Rectangle((z0_p, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))

for cfg, colore, pc, corr_ramo in rami_attivi_plot:
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

# 5. VISTA DALL'ALTO X-Z
for idx_p in range(n_piastra):
    z0_p = idx_p * (s_piastra + d_piastra)
    ax5.add_patch(patches.Rectangle((0, z0_p), B_piastra, s_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))

x_pl_unici = sorted(set(xp for (xp, _) in res["coords_pl"]))
for xp in x_pl_unici:
    ax5.add_patch(patches.Rectangle((xp - diam_pl / 2.0, -prof_pl_disegno), diam_pl, prof_pl_disegno, facecolor='#333333', edgecolor='black', linewidth=1.0))

for cfg, colore, pc, corr_ramo in rami_attivi_plot:
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
# 8. VERIFICA EFFETTI TERMICI TRA SBARRE PARALLELE DEL FASCIO (GRAFICI REV. 2.1)
# ==============================================================================
st.markdown("---")
st.subheader("🔥 Verifica Effetti Termici ed Effetto Pelle tra Sbarre Parallele del Ramo (Rev. 2.1)")

if attiva_B:
    ramo_ispezione = st.radio(
        "Seleziona il Ramo di Uscita da ispezionare nel dettaglio (Sezione Skin Depth + Curva Portata vs Distanza d):",
        options=["Ramo A", "Ramo B"],
        horizontal=True
    )
    cfg_isp = cfg_A if ramo_ispezione == "Ramo A" else cfg_B
    mb_isp = mb_A if ramo_ispezione == "Ramo A" else mb_B
    I_op_isp = res["I_A"] if ramo_ispezione == "Ramo A" else res["I_B"]
else:
    cfg_isp = cfg_A
    mb_isp = mb_A
    I_op_isp = res["I_A"]

h_isp = int(cfg_isp["w"])
s_isp = int(cfg_isp["s"])
n_isp = int(cfg_isp["n"])
d_isp = float(cfg_isp["d"])
costa_isp = cfg_isp["costa"]

fig_rev, (ax_cad, ax_curva) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1, 1.35]})
fig_rev.patch.set_facecolor('#F8F9FA')
ax_cad.set_facecolor('#FFFFFF')

colore_metallo = (
    '#D97724' if ("Rame" in mat_scelto or "Cu" in mat_scelto)
    else ('#6C757D' if "AISI" in mat_scelto else '#ADB5BD')
)
sk = mb_isp["skin"]

if costa_isp:
    tot_w = n_isp * s_isp + (n_isp - 1) * d_isp
    x_start = -tot_w / 2.0
    for i in range(n_isp):
        x0 = x_start + i * (s_isp + d_isp)
        rect = patches.Rectangle((x0, 0), s_isp, h_isp, lw=1.8, edgecolor='#3A1C00', facecolor=colore_metallo, alpha=0.9)
        ax_cad.add_patch(rect)
        if sk != float("inf") and s_isp > 2 * sk and h_isp > 2 * sk:
            core = patches.Rectangle(
                (x0 + sk, sk), s_isp - 2 * sk, h_isp - 2 * sk,
                lw=1.0, linestyle='--', edgecolor='#222222', facecolor='#F3B684', alpha=0.5
            )
            ax_cad.add_patch(core)
        if d_isp > 0 and i < n_isp - 1:
            xm = x0 + s_isp + d_isp / 2.0
            ax_cad.annotate(
                '', xy=(xm, h_isp * 0.70), xytext=(xm, h_isp * 0.25),
                arrowprops=dict(arrowstyle='->', color='#0077B6', lw=2.2)
            )

    quota_desc = "" if n_isp == 1 else (" - a contatto (d = 0 mm)" if d_isp == 0 else f" - d = {d_isp:.0f} mm")
    simbolo_posa = "[ | ]" if n_isp == 1 else (f"[ {'|'*n_isp} ]" if d_isp == 0 else f"[ {' | '*n_isp} ]")
    ax_cad.set_title(f"Sezione Ramo {cfg_isp['id']} Posa di Costa {simbolo_posa}\n{n_isp}x({h_isp}x{s_isp} mm){quota_desc}", fontweight='bold', fontsize=11)
    ax_cad.set_xlim(-max(55, tot_w * 0.85), max(55, tot_w * 0.85))
    ax_cad.set_ylim(-35, h_isp + 40)

    ax_cad.annotate('', xy=(x_start - 12, h_isp), xytext=(x_start - 12, 0), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax_cad.text(x_start - 15, h_isp / 2.0, f"h = {h_isp} mm", va='center', ha='right', rotation=90, fontweight='bold', fontsize=9)
else:
    tot_h = n_isp * s_isp + (n_isp - 1) * d_isp
    for i in range(n_isp):
        y0 = i * (s_isp + d_isp)
        rect = patches.Rectangle((-h_isp / 2.0, y0), h_isp, s_isp, lw=1.8, edgecolor='#3A1C00', facecolor=colore_metallo, alpha=0.9)
        ax_cad.add_patch(rect)

    quota_desc = "" if n_isp == 1 else (" - a contatto (d = 0 mm)" if d_isp == 0 else f" - distanziate d = {d_isp:.0f} mm")
    simbolo_posa = "[ — ]" if n_isp == 1 else ("[ = ]" if n_isp == 2 else "[ ≡ ]")
    ax_cad.set_title(f"Sezione Ramo {cfg_isp['id']} Posa di Piatto {simbolo_posa}\n{n_isp}x({h_isp}x{s_isp} mm){quota_desc}", fontweight='bold', fontsize=11)
    ax_cad.set_xlim(-h_isp * 0.75, h_isp * 0.75)
    ax_cad.set_ylim(-30, max(110, tot_h + 40))

ax_cad.set_xlabel("Larghezza [mm]")
ax_cad.set_ylabel("Altezza [mm]")
ax_cad.grid(True, linestyle=':', alpha=0.6)

d_axis = np.linspace(0, max(30.0, s_isp * 3.0), 60)
for n_idx in [1, 2, 3, 4]:
    curve_iam = [
        esegui_calcolo_completo(
            h_isp, s_isp, n_idx, dv, costa_isp, mat_scelto, freq_in,
            tamb_in, dt_in, vent_in, sup_in, forma_in, lungh_tratta_in, t_lim_cw_in, ik_in
        )["I_amm"]
        for dv in d_axis
    ]
    lw_line = 3.2 if n_idx == n_isp else 1.5
    ax_curva.plot(d_axis, curve_iam, lw=lw_line, label=f"n = {n_idx} sbarre // ({n_idx * h_isp * s_isp} mm²)")

ax_curva.scatter([d_isp], [mb_isp["I_amm"]], color='#D62828', s=90, zorder=6, label=f"Portata I_amm Ramo {cfg_isp['id']} ({mb_isp['I_amm']:.0f} A)")
ax_curva.axhline(I_op_isp, color='#003049', linestyle='-.', lw=1.8, label=f"Corrente Operativa FDM I_{cfg_isp['id']} ({I_op_isp:.0f} A)")
ax_curva.axvline(s_isp, color='#D62828', linestyle='--', alpha=0.7, label=f"Passo naturale d = s ({s_isp} mm)")
ax_curva.set_title(f"Portata Ammissibile I_amm [A] vs Distanza Netta d [mm] (Ramo {cfg_isp['id']})", fontweight='bold', fontsize=11)
ax_curva.set_xlabel("Distanza netta d tra i conduttori in parallelo [mm]")
ax_curva.set_ylabel("Portata Ammissibile I_amm [A]")
ax_curva.grid(True, linestyle='--', alpha=0.6)
ax_curva.legend(loc="lower right", frameon=True, fontsize=8.8)

plt.tight_layout()
st.pyplot(fig_rev)
plt.close(fig_rev)

# ==============================================================================
# 9. SPECIFICA TECNICA PER UFFICIO TECNICO E PRODUZIONE (DFM REV. 2.1)
# ==============================================================================
st.subheader(f"📋 Specifica di Industrializzazione e Linee Guida DFM (Ramo {cfg_isp['id']} - {n_isp}x {h_isp}x{s_isp} mm)")

col_d1, col_d2, col_d3 = st.columns(3)
skin_str = "∞ (Corrente Continua DC)" if mb_isp["skin"] == float("inf") else f"{mb_isp['skin']:.2f} mm a {freq_in} Hz"
p_op_ramo_watt_m = mb_isp["r_ac_op_ohm_m"] * (I_op_isp ** 2)

with col_d1:
    st.markdown("**Verifiche Termofluidodinamiche**")
    st.write(f"- Temperatura assoluta di regime: **{T_regime:.1f} °C** (Tamb = {tamb_in} °C, DeltaT = {dt_in} K)")
    st.write(f"- Dissipazione alla corrente FDM ({I_op_isp:.0f} A): **{p_op_ramo_watt_m:.1f} watt per metro**")
    st.write(f"- Dissipazione a portata limite ({mb_isp['I_amm']:.0f} A): **{mb_isp['P_joule_fase_watt_m']:.1f} watt per metro**")
    st.write(f"- Potenza termica trifase su {lungh_tratta_in:.1f} m (a I_amm): **{mb_isp['P_joule_tot_3ph_kw']:.2f} chilowatt**")
    st.write(f"- Profondità di penetrazione (Skin Depth): **{skin_str}**")

with col_d2:
    st.markdown("**Dilatazioni Termiche ed Elettrodinamica**")
    st.write(f"- Dilatazione da posa in officina (20 °C -> {T_regime:.0f} °C): **{mb_isp['Delta_L_posa_mm']:.2f} mm** (su {lungh_tratta_in:.1f} m)")
    st.write(f"- Dilatazione per solo salto termico (DeltaT = {dt_in} K): **{mb_isp['Delta_L_DT_mm']:.2f} mm**")
    st.write(f"- Luce minima asola su supporti CAD: **{max(6.0, mb_isp['Delta_L_posa_mm'] + 4.0):.1f} mm**")
    if n_isp > 1 and d_isp > 0:
        st.write(f"- Forza attrattiva sub-conduttori (Ip = {mb_isp['i_picco_kA']:.0f} kA): **{mb_isp['F_sub_Nm']:,.0f} N/m**")
        st.write(f"- Passo max calcolato spaziatori intermedi: **<= {mb_isp['L_max_spacers_mm']:.0f} mm**")
    else:
        st.write("- Passo consigliato supporti portanti: **<= 500 mm**")

with col_d3:
    st.markdown("**Tenuta al Cortocircuito e Componenti**")
    st.write(f"- Icw 1s rigorosa CEI EN 60865-1 (rho(T) variabile): **{mb_isp['Icw_1s_IEC']:.1f} kA**")
    st.write(f"- Icw 1s modello Excel (rho20 costante): **{mb_isp['Icw_1s_Excel']:.1f} kA**")
    st.write(f"- Integrale di Joule ammissibile (I^2 t a 1s): **{mb_isp['I2t']:,.0f} kA²s**")
    st.write(f"- Isolatori portanti prescritti: **Serie a pettine DMC/SMC 1000V (passo {s_isp} mm)**")

# ==============================================================================
# 10. TABELLA ANALITICA COEFFICIENTI K1..K9 ED ESPORTAZIONE CSV
# ==============================================================================
st.markdown("---")
st.subheader("📊 Report Analitico Completo (Piastra e Rami in Uscita con Appoggio Ideale)")
st.caption(
    "**Nota Tecnica:** Le barre sono approssimate come con **appoggio ideale** sulla piastra collettrice. "
    "Il coefficiente $K_7$ valuta il mutuo accoppiamento termico tra le sbarre in parallelo appartenenti al medesimo fascio."
)

righe_export = [
    {
        "Elemento": "Piastra Collettrice",
        "Configurazione Posa": f"{n_piastra}x({B_piastra:.0f}x{s_piastra:.0f} mm) Di costa",
        "Metallo": mat_scelto,
        "Sezione Totale [mm²]": int(sez_piastra),
        "Distanza d [mm]": d_piastra if n_piastra > 1 else 0,
        "I_operativa FDM [A]": round(I_sez_piastra_max, 1),
        "J_operativa [A/mm²]": round(J_sez_piastra_max, 2),
        "I_amm [A]": round(mb_piastra["I_amm"], 1),
        "J_amm [A/mm²]": round(mb_piastra["J_amm"], 2),
        "Esito Termico": "CONFORME" if I_sez_piastra_max <= mb_piastra["I_amm"] else "SOVRACCARICO",
        "K1": round(mb_piastra["K1"], 3), "K2": round(mb_piastra["K2"], 3), "K3": round(mb_piastra["K3"], 3),
        "K4": round(mb_piastra["K4"], 3), "K5": round(mb_piastra["K5"], 3), "K6": round(mb_piastra["K6"], 3),
        "K7 (Fascio)": round(mb_piastra["K7"], 3), "K8": round(mb_piastra["K8"], 3), "K9 (Skin)": round(mb_piastra["K9"], 4),
        "K_tot": round(mb_piastra["K_tot"], 4),
        "P_Nodo_FDM [watt]": round(res["P_piastra"], 1),
        "Icw 1s IEC [kA]": round(mb_piastra["Icw_1s_IEC"], 1),
        "Icw 1s Excel [kA]": round(mb_piastra["Icw_1s_Excel"], 1),
        "Delta_L da 20°C [mm]": round(mb_piastra["Delta_L_posa_mm"], 2),
        "Passo Max Spaziatori [mm]": round(mb_piastra["L_max_spacers_mm"], 0)
    },
    {
        "Elemento": "Ramo Uscita A (Appoggio Ideale)",
        "Configurazione Posa": f"{cfg_A['n']}x({cfg_A['w']:.0f}x{cfg_A['s']:.0f} mm) {cfg_A['posa_str']}",
        "Metallo": mat_scelto,
        "Sezione Totale [mm²]": int(sez_A),
        "Distanza d [mm]": cfg_A["d"] if cfg_A["n"] > 1 else 0,
        "I_operativa FDM [A]": round(res["I_A"], 1),
        "J_operativa [A/mm²]": round(J_L_A, 2),
        "I_amm [A]": round(mb_A["I_amm"], 1),
        "J_amm [A/mm²]": round(mb_A["J_amm"], 2),
        "Esito Termico": "CONFORME" if res["I_A"] <= mb_A["I_amm"] else "SOVRACCARICO",
        "K1": round(mb_A["K1"], 3), "K2": round(mb_A["K2"], 3), "K3": round(mb_A["K3"], 3),
        "K4": round(mb_A["K4"], 3), "K5": round(mb_A["K5"], 3), "K6": round(mb_A["K6"], 3),
        "K7 (Fascio)": round(mb_A["K7"], 3), "K8": round(mb_A["K8"], 3), "K9 (Skin)": round(mb_A["K9"], 4),
        "K_tot": round(mb_A["K_tot"], 4),
        "P_Nodo_FDM [watt]": round(res["P_A"], 1),
        "Icw 1s IEC [kA]": round(mb_A["Icw_1s_IEC"], 1),
        "Icw 1s Excel [kA]": round(mb_A["Icw_1s_Excel"], 1),
        "Delta_L da 20°C [mm]": round(mb_A["Delta_L_posa_mm"], 2),
        "Passo Max Spaziatori [mm]": round(mb_A["L_max_spacers_mm"], 0)
    }
]

if attiva_B:
    righe_export.append({
        "Elemento": "Ramo Uscita B (Appoggio Ideale)",
        "Configurazione Posa": f"{cfg_B['n']}x({cfg_B['w']:.0f}x{cfg_B['s']:.0f} mm) {cfg_B['posa_str']}",
        "Metallo": mat_scelto,
        "Sezione Totale [mm²]": int(sez_B),
        "Distanza d [mm]": cfg_B["d"] if cfg_B["n"] > 1 else 0,
        "I_operativa FDM [A]": round(res["I_B"], 1),
        "J_operativa [A/mm²]": round(J_L_B, 2),
        "I_amm [A]": round(mb_B["I_amm"], 1),
        "J_amm [A/mm²]": round(mb_B["J_amm"], 2),
        "Esito Termico": "CONFORME" if res["I_B"] <= mb_B["I_amm"] else "SOVRACCARICO",
        "K1": round(mb_B["K1"], 3), "K2": round(mb_B["K2"], 3), "K3": round(mb_B["K3"], 3),
        "K4": round(mb_B["K4"], 3), "K5": round(mb_B["K5"], 3), "K6": round(mb_B["K6"], 3),
        "K7 (Fascio)": round(mb_B["K7"], 3), "K8": round(mb_B["K8"], 3), "K9 (Skin)": round(mb_B["K9"], 4),
        "K_tot": round(mb_B["K_tot"], 4),
        "P_Nodo_FDM [watt]": round(res["P_B"], 1),
        "Icw 1s IEC [kA]": round(mb_B["Icw_1s_IEC"], 1),
        "Icw 1s Excel [kA]": round(mb_B["Icw_1s_Excel"], 1),
        "Delta_L da 20°C [mm]": round(mb_B["Delta_L_posa_mm"], 2),
        "Passo Max Spaziatori [mm]": round(mb_B["L_max_spacers_mm"], 0)
    })

df_export = pd.DataFrame(righe_export)
st.dataframe(df_export, use_container_width=True)

csv_data = df_export.to_csv(index=False, sep=";").encode('utf-8')
st.download_button(
    label="💾 Scarica Report Tecnico Unificato 2D FDM + Rev. 2.1 (CSV)",
    data=csv_data,
    file_name=f"Report_Busbar_FDM_Rev21_{num_rami}Rami_{B_piastra:.0f}x{H_piastra:.0f}mm.csv",
    mime="text/csv"
)
