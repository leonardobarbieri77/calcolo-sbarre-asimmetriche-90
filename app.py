import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LOADBANK-MASTER | Busbar 2D FDM + Verifica Melsom & Booth Rev. 2.1",
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
    "Barre grezze di estrusione o trafilatura (emissività ~0,4) (K5 = 1,00)": "grezza",
    "Barre verniciate opache o con termorestringente nero (emissività ~0,9)": "verniciata",
}

MODO_ORIZZ = "Orizzontale (0° - Piega lungo X)"
MODO_VERT = "Verticale (90° - Piega lungo Y)"


# ==============================================================================
# 2. MOTORE DI PORTATA AMMISSIBILE MELSOM & BOOTH (DA TELEMA REV. 2.1)
# ==============================================================================
def calcola_portata_melsom_booth(h, s, metallo_str, freq, t_amb, dt, vent_str, sup_str, giacitura_costa=True):
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
    is_alluminio = "Alluminio" in metallo_str or "alluminio" in metallo_str
    k5 = 1.00 if SUPERFICIE_K5[sup_str] == "grezza" else (1.12 if is_alluminio else 1.08)
    k6 = 1.00
    k7 = 1.00
    k8 = 1.00 if giacitura_costa else 0.86

    mu = 4.0 * math.pi * 1e-7 * mat["mu_r"]
    if freq <= 0:
        skin_depth = float("inf")
        k9 = 1.00
    else:
        skin_depth = 1000.0 * math.sqrt(2.0 * 1e-6 * mat["rho"] / (2.0 * math.pi * freq * mu))
        lato_mag = max(float(h), float(s))
        lato_min = min(float(h), float(s))
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

    p_1 = 2.0 * (h + s)
    sez_1 = float(h * s)
    k_tot = k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8 * k9
    i_amm = 5.0 * k_tot * (p_1 ** 0.39) * (sez_1 ** 0.5)
    j_amm = i_amm / sez_1

    return {
        "I_amm": i_amm, "J_amm": j_amm, "K_tot": k_tot,
        "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "K8": k8, "K9": k9, "skin": skin_depth
    }


st.title("⚡ LOADBANK-MASTER | Ripartizione 2D Powerlock + Verifica Melsom & Booth (Rev. 2.1)")
st.markdown(
    "Solutore alle differenze finite (2D FDM) per piastre collettrice con **criteri di accettabilità termoelettrica "
    "unificati al calcolatore Telema Busbar Rev. 2.1** (`DIN 43671` / `Melsom & Booth` $I_{\\text{amm}} = 5 \\cdot K_{\\text{tot}} \\cdot P^{0,39} \\cdot S^{0,5}$)."
)

# ==============================================================================
# 3. BARRA LATERALE DI CONTROLLO PARAMETRI
# ==============================================================================
st.sidebar.header("1. Materiale e Condizioni Ambientali (Rev. 2.1)")
mat_scelto = st.sidebar.selectbox("Lega Conduttrice (K1)", list(METALLI_K1.keys()), index=0)
dati_mat = METALLI_K1[mat_scelto]

freq_in = st.sidebar.number_input("Frequenza di rete [Hz] (0 = DC)", min_value=0, max_value=400, value=50, step=10)
tamb_in = st.sidebar.selectbox("Temperatura Ambiente Tamb [°C] (K2)", options=[20, 25, 35, 40, 50, 60, 70], index=3)
dt_in = st.sidebar.selectbox("Sovratemperatura Ammessa DT [K] (K3)", options=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70], index=5)
vent_in = st.sidebar.selectbox("Regime di Ventilazione (K4)", options=list(VENTILAZIONE_K4.keys()), index=0)
sup_in = st.sidebar.selectbox("Stato Superficie (K5)", options=list(SUPERFICIE_K5.keys()), index=0)

T_regime = tamb_in + dt_in
rho_caldo = dati_mat["rho"] * (1.0 + dati_mat["alpha"] * (T_regime - 20.0))

st.sidebar.caption(
    f"**Parametri Operativi:** $K_1$ = **{dati_mat['K1']:.2f}** | $T_{{\\text{{regime}}}}$ = **{T_regime} °C** | "
    f"$\\rho({T_regime}\\text{{ °C}})$ = **{rho_caldo:.4f} µΩ·m**"
)

st.sidebar.header("2. Piastra Collettrice e Powerlock")
B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1500.0, value=200.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

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


def configura_barra_sidebar(id_barra, y_def, orient_def_idx, lato_def_idx):
    st.sidebar.header(f"3{id_barra}. Barra {id_barra} in Uscita")
    orient = st.sidebar.selectbox(f"Orientamento Barra {id_barra}", [MODO_ORIZZ, MODO_VERT], index=orient_def_idx, key=f"or_{id_barra}")
    w_b = st.sidebar.number_input(f"Larghezza Barra {id_barra} (h_barra) [mm]", min_value=20.0, max_value=800.0, value=120.0, step=10.0, key=f"w_{id_barra}")
    h_b = st.sidebar.number_input(f"Profondità Sormonto Barra {id_barra} [mm]", min_value=15.0, max_value=400.0, value=60.0, step=5.0, key=f"h_{id_barra}")
    s_b = st.sidebar.number_input(f"Spessore Barra {id_barra} (s_barra) [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0, key=f"s_{id_barra}")
    L_b = st.sidebar.number_input(f"Lunghezza Ramo {id_barra} fino a Barra Equipotenziale [mm]", min_value=20.0, max_value=1500.0, value=120.0, step=10.0, key=f"L_{id_barra}")

    if orient == MODO_ORIZZ:
        dx_sorm, dy_sorm = w_b, h_b
        lato_piega = st.sidebar.selectbox(
            f"Lato Piega d'Uscita Barra {id_barra}",
            ["Bordo Inferiore del Sormonto", "Bordo Superiore del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
        giacitura_ala_costa = False
    else:
        dx_sorm, dy_sorm = h_b, w_b
        lato_piega = st.sidebar.selectbox(
            f"Lato Piega d'Uscita Barra {id_barra}",
            ["Bordo Sinistro del Sormonto", "Bordo Destro del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
        giacitura_ala_costa = True

    if dx_sorm > B_piastra or dy_sorm > H_piastra:
        st.error(f"Errore geometrico Barra {id_barra}: L'impronta ({dx_sorm:.0f}x{dy_sorm:.0f} mm) supera le dimensioni della piastra ({B_piastra:.0f}x{H_piastra:.0f} mm).")
        st.stop()

    min_x, max_x = dx_sorm / 2.0, B_piastra - dx_sorm / 2.0
    min_y, max_y = dy_sorm / 2.0, H_piastra - dy_sorm / 2.0

    x_sorm = st.sidebar.slider(
        f"Quota X Centro Sormonto Barra {id_barra} [mm]",
        min_value=float(min_x), max_value=float(max_x),
        value=float(min(max(B_piastra / 2.0, min_x), max_x)), step=5.0, key=f"xs_{id_barra}"
    )
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
        "id": id_barra, "orient": orient, "w": w_b, "h": h_b, "s": s_b, "L": L_b,
        "dx": dx_sorm, "dy": dy_sorm, "x_sorm": x_sorm, "y_sorm": y_sorm,
        "x_ala": x_ala, "y_ala": y_ala, "lato_piega": lato_piega,
        "costa": giacitura_ala_costa
    }


cfg_A = configura_barra_sidebar("A", y_def=60.0, orient_def_idx=0, lato_def_idx=0)
cfg_B = configura_barra_sidebar("B", y_def=500.0, orient_def_idx=0, lato_def_idx=1)

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
    errori_geom.append("Collisione meccanica 2D: Le impronte di sormonto della Barra A e della Barra B si sovrappongono!")

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()

# --- CALCOLO PORTATE AMMISSIBILI MELSOM & BOOTH (PIASTRA, BARRA A, BARRA B) ---
mb_piastra = calcola_portata_melsom_booth(B_piastra, s_piastra, mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, giacitura_costa=True)
mb_A = calcola_portata_melsom_booth(cfg_A["w"], cfg_A["s"], mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, giacitura_costa=cfg_A["costa"])
mb_B = calcola_portata_melsom_booth(cfg_B["w"], cfg_B["s"], mat_scelto, freq_in, tamb_in, dt_in, vent_in, sup_in, giacitura_costa=cfg_B["costa"])


# ==============================================================================
# 4. SOLUTORE ELETTRODINAMICO 2D FDM VETTORIALIZZATO
# ==============================================================================
@st.cache_data(show_spinner=False)
def risolvi_busbar_2d(
    B, H, s, rho_uohm_m, I_tot, n_col, n_row, px, py, x0_center_pl, y0_pl, d_pl,
    xA1, xA2, yA1, yA2, wA, hA, sA, LA,
    xB1, xB2, yB1, yB2, wB, hB, sB, LB
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

    Gx = sigma * (s * dy) / dx
    Gy = sigma * (s * dx) / dy

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

    R_att_A = rho * (0.5 * (s + sA)) / (wA * hA)
    R_sorm_A = rho * hA / (3.0 * wA * sA)
    R_A = R_att_A + R_sorm_A + rho * LA / (wA * sA)

    R_att_B = rho * (0.5 * (s + sB)) / (wB * hB)
    R_sorm_B = rho * hB / (3.0 * wB * sB)
    R_B = R_att_B + R_sorm_B + rho * LB / (wB * sB)

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
    profilo_Iy = np.sum(Jy, axis=1) * dx * s

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


res = risolvi_busbar_2d(
    B_piastra, H_piastra, s_piastra, rho_caldo, I_totale,
    X_col, Y_row, passo_x, passo_y, x_offset_pl, y_offset_pl, diam_pl,
    xA1, xA2, yA1, yA2, cfg_A["w"], cfg_A["h"], cfg_A["s"], cfg_A["L"],
    xB1, xB2, yB1, yB2, cfg_B["w"], cfg_B["h"], cfg_B["s"], cfg_B["L"]
)

# ==============================================================================
# 5. METRICHE KPI CONFRONTATE CON PORTATA MELSOM & BOOTH REV. 2.1
# ==============================================================================
p_A = res["I_A"] / I_totale * 100.0
p_B = res["I_B"] / I_totale * 100.0
sez_A = cfg_A["w"] * cfg_A["s"]
sez_B = cfg_B["w"] * cfg_B["s"]
sez_piastra = B_piastra * s_piastra

J_L_A = res["I_A"] / sez_A
J_L_B = res["I_B"] / sez_B
I_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"])))
J_sez_piastra_max = I_sez_piastra_max / sez_piastra
J_99_piastra = float(np.percentile(res["J_mag"], 99))
P_tot = res["P_piastra"] + res["P_A"] + res["P_B"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"Corrente Barra A (Amm: {mb_A['I_amm']:.0f} A)",
    f"{res['I_A']:.0f} A ({p_A:.1f}%)",
    f"J = {J_L_A:.2f} A/mm² (J_amm = {mb_A['J_amm']:.2f} A/mm²)",
    delta_color="inverse" if res["I_A"] > mb_A["I_amm"] else "normal"
)
c2.metric(
    f"Corrente Barra B (Amm: {mb_B['I_amm']:.0f} A)",
    f"{res['I_B']:.0f} A ({p_B:.1f}%)",
    f"J = {J_L_B:.2f} A/mm² (J_amm = {mb_B['J_amm']:.2f} A/mm²)",
    delta_color="inverse" if res["I_B"] > mb_B["I_amm"] else "normal"
)
c3.metric(
    f"Max Sezione Piastra (Amm: {mb_piastra['I_amm']:.0f} A)",
    f"{I_sez_piastra_max:.0f} A ({J_sez_piastra_max:.2f} A/mm²)",
    f"J_amm Piastra = {mb_piastra['J_amm']:.2f} A/mm² (K1={dati_mat['K1']:.2f})",
    delta_color="inverse" if I_sez_piastra_max > mb_piastra["I_amm"] else "normal"
)
c4.metric(
    f"Potenza Dissipata (a {T_regime} °C)",
    f"{P_tot:.1f} watt",
    f"Solo piastra: {res['P_piastra']:.1f} watt ({P_tot/1000.0:.2f} chilowatt)",
    delta_color="off"
)

st.markdown("---")

# ==============================================================================
# 6. VISUALIZZAZIONE 4 GRAFICI IN SCALA 1:1
# ==============================================================================
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(21, 10), gridspec_kw={'width_ratios': [1.15, 1.15, 1.0, 1.0]})


def disegna_impronta_e_piega(ax, cfg, perc, colore_bordo):
    x0 = cfg["x_sorm"] - cfg["dx"] / 2.0
    y0 = cfg["y_sorm"] - cfg["dy"] / 2.0
    ax.add_patch(patches.Rectangle((x0, y0), cfg["dx"], cfg["dy"], edgecolor=colore_bordo, facecolor='none', linewidth=2.0, linestyle='--'))
    if cfg["orient"] == MODO_ORIZZ:
        y_p = y0 if "Inferiore" in cfg["lato_piega"] else (y0 + cfg["dy"])
        ax.plot([x0, x0 + cfg["dx"]], [y_p, y_p], color='#00ff00', linewidth=4.0)
    else:
        x_p = x0 if "Sinistro" in cfg["lato_piega"] else (x0 + cfg["dx"])
        ax.plot([x_p, x_p], [y0, y0 + cfg["dy"]], color='#00ff00', linewidth=4.0)
    rot_tag = "0°" if cfg["orient"] == MODO_ORIZZ else "90°"
    ax.text(
        cfg["x_sorm"], cfg["y_sorm"], f"{cfg['id']} ({rot_tag})\n{perc:.1f}%",
        color='white', weight='bold', fontsize=8, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75)
    )


# 1. Mappa Densità |J|
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(mb_piastra["J_amm"] * 1.3, J_99_piastra))
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Densità |J| [A/mm²]")
ax1.streamplot(res["x"], res["y"], res["Jx"], res["Jy"], color='cyan', linewidth=0.7, density=0.85, arrowsize=0.7)
for (xp, yp) in res["coords_pl"]:
    ax1.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2))
disegna_impronta_e_piega(ax1, cfg_A, p_A, 'cyan')
disegna_impronta_e_piega(ax1, cfg_B, p_B, 'cyan')
ax1.set_title("1. Densità |J| e Lato Piega (Verde)", fontsize=10, weight='bold')
ax1.set_xlabel("Larghezza X [mm]")
ax1.set_ylabel("Quota Y [mm]")
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')

# 2. Mappa Potenziale V [mV]
im2 = ax2.pcolormesh(res["X"], res["Y"], res["V_mV"], shading='auto', cmap='viridis')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Potenziale [mV]")
for (xp, yp) in res["coords_pl"]:
    ax2.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7))
disegna_impronta_e_piega(ax2, cfg_A, p_A, 'red')
disegna_impronta_e_piega(ax2, cfg_B, p_B, 'red')
ax2.set_title("2. Potenziale Elettrico [mV]", fontsize=10, weight='bold')
ax2.set_xlabel("Larghezza X [mm]")
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')

# 3. Flusso Assiale Verticale |Iy| con Portata Melsom & Booth Reale
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)| in piastra")
ax3.axhline(cfg_A["y_sorm"], color='blue', linestyle='--', linewidth=1.3, label=f"Sormonto A (Y={cfg_A['y_sorm']:.0f} mm)")
ax3.axhline(cfg_B["y_sorm"], color='green', linestyle='--', linewidth=1.3, label=f"Sormonto B (Y={cfg_B['y_sorm']:.0f} mm)")
ax3.axvline(mb_piastra["I_amm"], color='red', linestyle=':', linewidth=1.8, label=f"I_amm Rev.2.1 ({mb_piastra['I_amm']:.0f} A)")
ax3.fill_betweenx(res["y"], 0, np.abs(res["profilo_Iy"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Flusso Assiale |Iy| [A]", fontsize=10, weight='bold')
ax3.set_xlabel("Corrente Assiale [A]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

# 4. Proiezione Laterale Meccanica Y-Z
L_DRAW_MAX = 150.0
prof_pl_disegno = diam_pl * 1.4
ax4.add_patch(patches.Rectangle((0, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))
for cfg, colore, pc, corr_ramo in ((cfg_A, 'blue', p_A, res["I_A"]), (cfg_B, 'green', p_B, res["I_B"])):
    L_dis = min(cfg["L"], L_DRAW_MAX)
    y_bot = cfg["y_sorm"] - cfg["dy"] / 2.0
    ax4.add_patch(patches.Rectangle((s_piastra, y_bot), cfg["s"], cfg["dy"], facecolor='#e68a00', edgecolor='black', linewidth=1.1))
    if cfg["orient"] == MODO_ORIZZ:
        ax4.add_patch(patches.Rectangle((s_piastra + cfg["s"], cfg["y_ala"] - cfg["s"] / 2.0), L_dis, cfg["s"], facecolor='#e68a00', edgecolor='black', linewidth=1.1))
        y_testo = cfg["y_ala"]
    else:
        ax4.add_patch(patches.Rectangle((s_piastra + cfg["s"], y_bot), L_dis, cfg["dy"], facecolor='#e68a00', edgecolor='black', linewidth=1.1, alpha=0.55, hatch='//'))
        y_testo = cfg["y_sorm"]
    ax4.text(
        s_piastra + cfg["s"] + L_dis + 8, y_testo,
        f"{cfg['id']}: {pc:.1f}%\n({corr_ramo:.0f} A)\nL={cfg['L']:.0f} mm",
        color=colore, fontsize=8, weight='bold', va='center'
    )
y_pl_unici = sorted(set(yp for (_, yp) in res["coords_pl"]))
for yp in y_pl_unici:
    ax4.add_patch(patches.Rectangle((-prof_pl_disegno, yp - diam_pl / 2.0), prof_pl_disegno, diam_pl, facecolor='#333333', edgecolor='black', linewidth=0.9))
ax4.set_title("4. Proiezione Laterale Y-Z", fontsize=10, weight='bold')
ax4.set_xlabel("Profondità Z [mm]")
ax4.set_xlim(-prof_pl_disegno - 30, s_piastra + max(cfg_A["s"], cfg_B["s"]) + L_DRAW_MAX + 150)
ax4.set_ylim(0, H_piastra)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ==============================================================================
# 7. REPORT TECNICO DI VERIFICA MELSOM & BOOTH REV. 2.1
# ==============================================================================
st.subheader("📋 Report di Verifica Elettrotermica Unificato (Metodo Melsom & Booth Rev. 2.1)")
st.table(pd.DataFrame({
    "Elemento / Parametro": [
        "Condizioni Operative e Coefficienti Ambientali",
        f"Piastra Collettrice ({B_piastra:.0f}x{s_piastra:.0f} mm - Posa di Costa)",
        f"Ramo Barra A ({cfg_A['w']:.0f}x{cfg_A['s']:.0f} mm - {cfg_A['orient']})",
        f"Ramo Barra B ({cfg_B['w']:.0f}x{cfg_B['s']:.0f} mm - {cfg_B['orient']})",
        "Resistenze Ohmiche a Caldo dei Rami (A / B)",
        f"Potenza Dissipata per Effetto Joule a {T_regime} °C"
    ],
    "Valore Operativo Calcolato (2D FDM)": [
        f"Tamb = {tamb_in} °C | DT = {dt_in} K | T_regime = {T_regime} °C | f = {freq_in} Hz",
        f"I_max sezione = {I_sez_piastra_max:.0f} A  |  J_max sezione = {J_sez_piastra_max:.2f} A/mm²",
        f"I_A = {res['I_A']:.0f} A ({p_A:.1f} %)  |  J_A = {J_L_A:.2f} A/mm²",
        f"I_B = {res['I_B']:.0f} A ({p_B:.1f} %)  |  J_B = {J_L_B:.2f} A/mm²",
        f"R_A = {res['R_A_uohm']:.2f} µΩ (L_A={cfg_A['L']:.0f} mm)  |  R_B = {res['R_B_uohm']:.2f} µΩ (L_B={cfg_B['L']:.0f} mm)",
        f"Piastra: {res['P_piastra']:.1f} watt | Barre a L: {(res['P_A'] + res['P_B']):.1f} watt | Totale: {P_tot:.1f} watt ({P_tot/1000.0:.2f} chilowatt)"
    ],
    "Limite Ammissibile Melsom & Booth (Rev. 2.1)": [
        f"K1={mb_piastra['K1']:.2f}, K2={mb_piastra['K2']:.2f}, K3={mb_piastra['K3']:.2f}, K4={mb_piastra['K4']:.2f}, K5={mb_piastra['K5']:.2f}, K9={mb_piastra['K9']:.3f}",
        f"I_amm = {mb_piastra['I_amm']:.0f} A  |  J_amm = {mb_piastra['J_amm']:.2f} A/mm² (K_tot = {mb_piastra['K_tot']:.3f})",
        f"I_amm = {mb_A['I_amm']:.0f} A  |  J_amm = {mb_A['J_amm']:.2f} A/mm² (K8 = {mb_A['K8']:.2f}, K_tot = {mb_A['K_tot']:.3f})",
        f"I_amm = {mb_B['I_amm']:.0f} A  |  J_amm = {mb_B['J_amm']:.2f} A/mm² (K8 = {mb_B['K8']:.2f}, K_tot = {mb_B['K_tot']:.3f})",
        f"Calcolate con resistività a caldo rho({T_regime} °C) = {rho_caldo:.4f} µΩ·m",
        "Da evacuare nel vano sbarre per mantenere DT <= " + f"{dt_in} K"
    ],
    "Esito Verifica": [
        "Parametri normativi DIN 43671 / CEI EN 61439-1",
        "VERIFICATO" if I_sez_piastra_max <= mb_piastra["I_amm"] else "SUPERAMENTO PORTATA PIASTRA",
        "VERIFICATO" if res["I_A"] <= mb_A["I_amm"] else "SUPERAMENTO PORTATA BARRA A",
        "VERIFICATO" if res["I_B"] <= mb_B["I_amm"] else "SUPERAMENTO PORTATA BARRA B",
        "Continuità materiale sul sormonto",
        f"Errore Kirchhoff: {res['err_perc']:.6f} %"
    ]
}))
