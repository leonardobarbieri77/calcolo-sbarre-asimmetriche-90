import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LOADBANK-MASTER | Busbar 2D Asimmetrico & 90°",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATABASE MATERIALI TELEMA DA TABELLA TECNICA ---
MATERIALI_DB = {
    "Rame elettrolitico (rho=0,0176 uOhm m)": {"K1": 1.27, "rho_20": 0.0176, "densita": 8890.0, "c_sp": 385.0},
    "Rame elettrolitico (rho=0,0178 uOhm m)": {"K1": 1.26, "rho_20": 0.0178, "densita": 8890.0, "c_sp": 385.0},
    "Alluminio purezza elettrica (rho=0,028 uOhm m)": {"K1": 1.02, "rho_20": 0.0280, "densita": 2700.0, "c_sp": 880.0},
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {"K1": 1.00, "rho_20": 0.0290, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0325 uOhm m)": {"K1": 0.94, "rho_20": 0.0325, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0385 uOhm m)": {"K1": 0.87, "rho_20": 0.0385, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0400 uOhm m)": {"K1": 0.85, "rho_20": 0.0400, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0580 uOhm m)": {"K1": 0.71, "rho_20": 0.0580, "densita": 2700.0, "c_sp": 880.0},
    "AISI 304": {"K1": 0.21, "rho_20": 0.7200, "densita": 8000.0, "c_sp": 500.0},
    "CuCr1Zr (rho=0,028 uOhm m)": {"K1": 1.01, "rho_20": 0.0280, "densita": 8910.0, "c_sp": 450.0}
}

VITI_DB = {
    "M8":  {"d_foro": 9.0,  "D_rond": 16.0, "coppia": 25.0, "d_nom": 8.0},
    "M10": {"d_foro": 11.0, "D_rond": 20.0, "coppia": 44.0, "d_nom": 10.0},
    "M12": {"d_foro": 13.5, "D_rond": 24.0, "coppia": 75.0, "d_nom": 12.0},
}

TRATTAMENTI_DB = {
    "Ideale (R_film = 0)": 0.0,
    "Argentato": 150.0,
    "Stagnato": 300.0,
    "Nichelato": 450.0,
    "Nudo (Spazzolato)": 600.0
}

K_ATTRITO = 0.20
ANGOLO_CONO_DEG = 30.0

MODO_ORIZZ = "Orizzontale (0° - Piega lungo X)"
MODO_VERT = "Verticale (90° - Piega lungo Y)"

st.title("LOADBANK-MASTER | Ripartizione Corrente su Barre a L Asimmetriche e Ruotabili (0° / 90°)")
st.markdown(
    "Solutore alle differenze finite (2D FDM) per piastre collettrice con griglia **Powerlock** e derivazioni a L "
    "posizionabili liberamente lungo $X$ e $Y$, con orientamento indipendente orizzontale o verticale."
)

# --- SIDEBAR 1: MATERIALE E PIASTRA ---
st.sidebar.header("1. Materiale e Piastra Collettrice")
mat_scelto = st.sidebar.selectbox("Materiale Sbarre", list(MATERIALI_DB.keys()), index=0)
dati_mat = MATERIALI_DB[mat_scelto]
K1_mat = dati_mat["K1"]
rho_mat = dati_mat["rho_20"]
dens_mat = dati_mat["densita"]
csp_mat = dati_mat["c_sp"]

st.sidebar.caption(
    f"**Parametri Tabella:** $K_1$ = **{K1_mat:.2f} A/mm²** | "
    f"$\\rho_{{20}}$ = **{rho_mat:.4f} µΩ·m** | $\\delta$ = **{dens_mat:.0f} kg/m³**"
)

B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1500.0, value=250.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

# --- SIDEBAR 2: GRIGLIA POWERLOCK ---
st.sidebar.header("2. Griglia Powerlock (Ingresso)")
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
    st.sidebar.header(f"3{id_barra}. Geometria e Posizione Barra {id_barra}")
    orient = st.sidebar.selectbox(f"Orientamento Barra {id_barra}", [MODO_ORIZZ, MODO_VERT], index=orient_def_idx, key=f"or_{id_barra}")
    w_b = st.sidebar.number_input(f"Larghezza Piatto Barra {id_barra} (w_{id_barra}) [mm]", min_value=20.0, max_value=800.0, value=120.0, step=10.0, key=f"w_{id_barra}")
    h_b = st.sidebar.number_input(f"Profondità Sormonto Barra {id_barra} (h_{id_barra}) [mm]", min_value=15.0, max_value=400.0, value=60.0, step=5.0, key=f"h_{id_barra}")
    s_b = st.sidebar.number_input(f"Spessore Barra {id_barra} (s_{id_barra}) [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0, key=f"s_{id_barra}")
    L_b = st.sidebar.number_input(f"Lunghezza Ramo {id_barra} fino a Barra Equipotenziale [mm]", min_value=20.0, max_value=1500.0, value=120.0, step=10.0, key=f"L_{id_barra}")

    if orient == MODO_ORIZZ:
        dx_sorm, dy_sorm = w_b, h_b
        lato_piega = st.sidebar.selectbox(
            f"Lato Piega d'Uscita Barra {id_barra}",
            ["Bordo Inferiore del Sormonto", "Bordo Superiore del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )
    else:
        dx_sorm, dy_sorm = h_b, w_b
        lato_piega = st.sidebar.selectbox(
            f"Lato Piega d'Uscita Barra {id_barra}",
            ["Bordo Sinistro del Sormonto", "Bordo Destro del Sormonto"],
            index=lato_def_idx, key=f"lato_{id_barra}"
        )

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
        info_ala = f"Mezzeria ala uscita a Y = {y_ala:.1f} mm"
    else:
        y_ala = y_sorm
        x_ala = (x_sorm - dx_sorm / 2.0 + s_b / 2.0) if "Sinistro" in lato_piega else (x_sorm + dx_sorm / 2.0 - s_b / 2.0)
        info_ala = f"Mezzeria ala uscita a X = {x_ala:.1f} mm"

    st.sidebar.caption(f"Impronta: {dx_sorm:.0f}x{dy_sorm:.0f} mm | {info_ala}")

    return {
        "id": id_barra, "orient": orient, "w": w_b, "h": h_b, "s": s_b, "L": L_b,
        "dx": dx_sorm, "dy": dy_sorm, "x_sorm": x_sorm, "y_sorm": y_sorm,
        "x_ala": x_ala, "y_ala": y_ala, "lato_piega": lato_piega, "info_ala": info_ala
    }


cfg_A = configura_barra_sidebar("A", y_def=60.0, orient_def_idx=0, lato_def_idx=0)
cfg_B = configura_barra_sidebar("B", y_def=500.0, orient_def_idx=0, lato_def_idx=1)

# --- SIDEBAR 4: INTERFACCIA DI CONTATTO ---
st.sidebar.header("4. Modello Interfaccia Sormonto")
continuita_materiale = st.sidebar.checkbox(
    "Continuità del materiale (Preimpostato)",
    value=True,
    help="Se attivo, modella il contatto metallico continuo su tutta la superficie del sormonto. "
         "Disattiva per calcolare i coni di pressione delle viti."
)

area_sorm_A = cfg_A["w"] * cfg_A["h"]
area_sorm_B = cfg_B["w"] * cfg_B["h"]
errore_viti = None

if continuita_materiale:
    A_eff_A = area_sorm_A
    A_eff_B = area_sorm_B
    R_film_A = 0.0
    R_film_B = 0.0
    descr_giunto = "Continuità del materiale sull'intera impronta del sormonto"
else:
    vite_scelta = st.sidebar.selectbox("Calibro Vite", list(VITI_DB.keys()), index=1)
    n_viti = st.sidebar.number_input("Numero di viti per ciascun giunto", min_value=1, max_value=24, value=4, step=1)
    trattamento = st.sidebar.selectbox("Trattamento superficiale", list(TRATTAMENTI_DB.keys()), index=0)

    dati_vite = VITI_DB[vite_scelta]
    coppia_Nm = dati_vite["coppia"]
    precarico_N = coppia_Nm / (K_ATTRITO * dati_vite["d_nom"] * 1e-3)

    t_cono_A = min(s_piastra, cfg_A["s"])
    t_cono_B = min(s_piastra, cfg_B["s"])
    D_cono_A = dati_vite["D_rond"] + 2.0 * t_cono_A * np.tan(np.radians(ANGOLO_CONO_DEG))
    D_cono_B = dati_vite["D_rond"] + 2.0 * t_cono_B * np.tan(np.radians(ANGOLO_CONO_DEG))
    area_cono_A = np.pi / 4.0 * (D_cono_A ** 2 - dati_vite["d_foro"] ** 2)
    area_cono_B = np.pi / 4.0 * (D_cono_B ** 2 - dati_vite["d_foro"] ** 2)

    A_eff_A = float(min(n_viti * area_cono_A, area_sorm_A))
    A_eff_B = float(min(n_viti * area_cono_B, area_sorm_B))
    rho_film = TRATTAMENTI_DB[trattamento]
    R_film_A = (rho_film / A_eff_A) * 1e-6
    R_film_B = (rho_film / A_eff_B) * 1e-6

    pressione_MPa = precarico_N / area_cono_A
    descr_giunto = f"{n_viti} x {vite_scelta} ({trattamento}), Precarico ~{precarico_N/1000.0:.0f} kN/vite, Pressione cono ~{pressione_MPa:.0f} MPa"

    if n_viti * (dati_vite["D_rond"] ** 2) > min(area_sorm_A, area_sorm_B):
        errore_viti = f"Le {n_viti} rondelle da {dati_vite['D_rond']:.0f} mm superano l'area del sormonto impostato."

# --- VERIFICHE GEOMETRICHE E INTERSEZIONE 2D ---
x_min_pl = x_offset_pl - ((X_col - 1) * passo_x) / 2.0 - diam_pl / 2.0
x_max_pl = x_offset_pl + ((X_col - 1) * passo_x) / 2.0 + diam_pl / 2.0
y_min_pl = y_offset_pl - diam_pl / 2.0
y_max_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

errori_geom = []
if x_min_pl < 0 or x_max_pl > B_piastra:
    errori_geom.append(f"La griglia Powerlock (X da {x_min_pl:.1f} a {x_max_pl:.1f} mm) esce dalla larghezza piastra (0 - {B_piastra:.1f} mm).")
if y_min_pl < 0 or y_max_pl > H_piastra:
    errori_geom.append(f"La griglia Powerlock (Y da {y_min_pl:.1f} a {y_max_pl:.1f} mm) esce dall'altezza piastra (0 - {H_piastra:.1f} mm).")

# Controllo collisione rettangoli 2D tra Barra A e Barra B
xA1, xA2 = cfg_A["x_sorm"] - cfg_A["dx"] / 2.0, cfg_A["x_sorm"] + cfg_A["dx"] / 2.0
yA1, yA2 = cfg_A["y_sorm"] - cfg_A["dy"] / 2.0, cfg_A["y_sorm"] + cfg_A["dy"] / 2.0
xB1, xB2 = cfg_B["x_sorm"] - cfg_B["dx"] / 2.0, cfg_B["x_sorm"] + cfg_B["dx"] / 2.0
yB1, yB2 = cfg_B["y_sorm"] - cfg_B["dy"] / 2.0, cfg_B["y_sorm"] + cfg_B["dy"] / 2.0

sovrapposizione_2d = (xA1 < xB2) and (xA2 > xB1) and (yA1 < yB2) and (yA2 > yB1)
if sovrapposizione_2d:
    errori_geom.append("Collisione meccanica 2D: Le impronte di sormonto della Barra A e della Barra B si sovrappongono sulla piastra!")

if errore_viti is not None:
    errori_geom.append(errore_viti)

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()


# --- SOLUTORE ELETTRODINAMICO 2D FDM ---
@st.cache_data(show_spinner=False)
def risolvi_busbar_generalizzato(
    B, H, s, rho_uohm_m, I_tot, n_col, n_row, px, py, x0_center_pl, y0_pl, d_pl,
    xA1, xA2, yA1, yA2, wA, hA, sA, LA, AeffA, RfilmA,
    xB1, xB2, yB1, yB2, wB, hB, sB, LB, AeffB, RfilmB
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

    # Iniezione corrente dai Powerlock
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

    # Nodi di sormonto Barra A e Barra B
    mask_A = (X >= xA1) & (X <= xA2) & (Y >= yA1) & (Y <= yA2)
    mask_B = (X >= xB1) & (X <= xB2) & (Y >= yB1) & (Y <= yB2)
    ind_A = np.flatnonzero(mask_A)
    ind_B = np.flatnonzero(mask_B)

    if len(ind_A) == 0:
        ind_A = np.array([np.argmin((X - 0.5 * (xA1 + xA2)) ** 2 + (Y - 0.5 * (yA1 + yA2)) ** 2)])
    if len(ind_B) == 0:
        ind_B = np.array([np.argmin((X - 0.5 * (xB1 + xB2)) ** 2 + (Y - 0.5 * (yB1 + yB2)) ** 2)])

    # Resistenze di ramo indipendenti
    R_att_A = rho * (0.5 * (s + sA)) / AeffA + RfilmA
    R_sorm_A = rho * hA / (3.0 * wA * sA)
    R_A = R_att_A + R_sorm_A + rho * LA / (wA * sA)

    R_att_B = rho * (0.5 * (s + sB)) / AeffB + RfilmB
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
    profilo_Ix = np.sum(Jx, axis=0) * dy * s

    P_tot = float(I_vec @ V)
    P_ramo_A = (I_A ** 2) * R_A
    P_ramo_B = (I_B ** 2) * R_B
    P_piastra = max(0.0, P_tot - P_ramo_A - P_ramo_B)

    return {
        "x": x_c, "y": y_c, "X": X, "Y": Y,
        "V_mV": V_2d * 1e3, "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "I_A": I_A, "I_B": I_B, "err_perc": err_perc,
        "profilo_Iy": profilo_Iy, "profilo_Ix": profilo_Ix,
        "coords_pl": coords_pl,
        "P_piastra": P_piastra, "P_A": P_ramo_A, "P_B": P_ramo_B,
        "R_A_uohm": R_A * 1e6, "R_B_uohm": R_B * 1e6
    }


res = risolvi_busbar_generalizzato(
    B_piastra, H_piastra, s_piastra, rho_mat, I_totale,
    X_col, Y_row, passo_x, passo_y, x_offset_pl, y_offset_pl, diam_pl,
    xA1, xA2, yA1, yA2, cfg_A["w"], cfg_A["h"], cfg_A["s"], cfg_A["L"], A_eff_A, R_film_A,
    xB1, xB2, yB1, yB2, cfg_B["w"], cfg_B["h"], cfg_B["s"], cfg_B["L"], A_eff_B, R_film_B
)

# --- GRANDEZZE INGEGNERISTICHE DI VERIFICA ---
p_A = res["I_A"] / I_totale * 100.0
p_B = res["I_B"] / I_totale * 100.0
sez_A = cfg_A["w"] * cfg_A["s"]
sez_B = cfg_B["w"] * cfg_B["s"]
sez_piastra_Y = B_piastra * s_piastra

J_L_A = res["I_A"] / sez_A
J_L_B = res["I_B"] / sez_B
J_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"]))) / sez_piastra_Y
J_99_piastra = float(np.percentile(res["J_mag"], 99))

vol_piastra_m3 = B_piastra * H_piastra * s_piastra * 1e-9
vol_A_m3 = cfg_A["w"] * (cfg_A["h"] + cfg_A["L"]) * cfg_A["s"] * 1e-9
vol_B_m3 = cfg_B["w"] * (cfg_B["h"] + cfg_B["L"]) * cfg_B["s"] * 1e-9
massa_piastra_kg = vol_piastra_m3 * dens_mat
massa_totale_kg = (vol_piastra_m3 + vol_A_m3 + vol_B_m3) * dens_mat
P_tot = res["P_piastra"] + res["P_A"] + res["P_B"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Corrente Ramo A",
    f"{res['I_A']:.0f} A ({p_A:.1f}%)",
    f"J Barra A: {J_L_A:.2f} A/mm² (K1={K1_mat:.2f})",
    delta_color="inverse" if J_L_A > K1_mat else "normal"
)
c2.metric(
    "Corrente Ramo B",
    f"{res['I_B']:.0f} A ({p_B:.1f}%)",
    f"J Barra B: {J_L_B:.2f} A/mm² (K1={K1_mat:.2f})",
    delta_color="inverse" if J_L_B > K1_mat else "normal"
)
c3.metric(
    "Densità Sezione Piastra",
    f"{J_sez_piastra_max:.2f} A/mm²",
    f"Hotspot 99° perc: {J_99_piastra:.2f} A/mm²",
    delta_color="inverse" if J_sez_piastra_max > K1_mat else "normal"
)
c4.metric(
    "Potenza Dissipata Totale",
    f"{P_tot:.1f} watt",
    f"Solo piastra: {res['P_piastra']:.1f} watt",
    delta_color="off"
)

st.markdown("---")

# --- VISUALIZZAZIONE 4 GRAFICI IN SCALA 1:1 ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(21, 10), gridspec_kw={'width_ratios': [1.15, 1.15, 1.0, 1.0]})


def disegna_impronta_e_piega(ax, cfg, perc, colore_bordo):
    x0 = cfg["x_sorm"] - cfg["dx"] / 2.0
    y0 = cfg["y_sorm"] - cfg["dy"] / 2.0
    ax.add_patch(patches.Rectangle((x0, y0), cfg["dx"], cfg["dy"], edgecolor=colore_bordo, facecolor='none', linewidth=2.0, linestyle='--'))
    # Evidenzia con linea spessa il lato in cui la barra piega a 90° verso Z
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
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(K1_mat * 1.4, J_99_piastra))
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

# 3. Flusso Assiale Verticale |Iy|
portata_limite_piastra = sez_piastra_Y * K1_mat
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)| in piastra")
ax3.axhline(cfg_A["y_sorm"], color='blue', linestyle='--', linewidth=1.3, label=f"Centro A (Y={cfg_A['y_sorm']:.0f} mm)")
ax3.axhline(cfg_B["y_sorm"], color='green', linestyle='--', linewidth=1.3, label=f"Centro B (Y={cfg_B['y_sorm']:.0f} mm)")
ax3.axvline(portata_limite_piastra, color='red', linestyle=':', linewidth=1.5, label=f"Portata K1 ({portata_limite_piastra:.0f} A)")
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
    # Sormonto a contatto con la piastra
    ax4.add_patch(patches.Rectangle((s_piastra, y_bot), cfg["s"], cfg["dy"], facecolor='#e68a00', edgecolor='black', linewidth=1.1))
    if cfg["orient"] == MODO_ORIZZ:
        # Vista di taglio dell'ala orizzontale (spessore s_b)
        ax4.add_patch(patches.Rectangle((s_piastra + cfg["s"], cfg["y_ala"] - cfg["s"] / 2.0), L_dis, cfg["s"], facecolor='#e68a00', edgecolor='black', linewidth=1.1))
        y_testo = cfg["y_ala"]
    else:
        # Vista laterale della fascia verticale ruotata di 90° (alta dy = w_b)
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

# --- TABELLA DI REPORT TECNICO ---
st.subheader("Report di Verifica Elettromeccanica e Costruttiva")
st.table(pd.DataFrame({
    "Parametro Tecnico": [
        "Materiale e Coefficiente K1 (da Tabella)",
        "Massa Semilavorati (Piastra / Sistema Completo)",
        "Corrente Totale Iniettata / Singolo Powerlock",
        "Configurazione e Corrente Ramo A",
        "Configurazione e Corrente Ramo B",
        "Densità di Corrente Sezione Barre a L (A / B)",
        "Densità Massima nella Sezione Piastra / Hotspot 99°",
        "Modello Interfaccia Sormonto",
        "Resistenza Ohmica dei Rami (Ramo A / Ramo B)",
        "Potenza Termica Dissipata (Piastra / Rami a L / Totale)",
        "Controllo Numerico di Conservazione di Kirchhoff"
    ],
    "Valore Calcolato": [
        f"{mat_scelto}  |  K1 = {K1_mat:.2f} A/mm²",
        f"Piastra: {massa_piastra_kg:.1f} kg  |  Totale con barre: {massa_totale_kg:.1f} kg",
        f"{I_totale:.0f} A  |  {I_totale / (X_col * Y_row):.1f} A per connettore ({X_col * Y_row} poli)",
        f"{res['I_A']:.0f} A ({p_A:.1f} %) | {cfg_A['orient']} | Centro Sormonto (X={cfg_A['x_sorm']:.0f}, Y={cfg_A['y_sorm']:.0f} mm) | {cfg_A['info_ala']}",
        f"{res['I_B']:.0f} A ({p_B:.1f} %) | {cfg_B['orient']} | Centro Sormonto (X={cfg_B['x_sorm']:.0f}, Y={cfg_B['y_sorm']:.0f} mm) | {cfg_B['info_ala']}",
        f"Barra A: {J_L_A:.2f} A/mm² (Sez. {sez_A:.0f} mm²)  |  Barra B: {J_L_B:.2f} A/mm² (Sez. {sez_B:.0f} mm²)",
        f"Sezione Piastra: {J_sez_piastra_max:.2f} A/mm²  |  Hotspot 99° perc.: {J_99_piastra:.2f} A/mm²",
        descr_giunto,
        f"Ramo A: {res['R_A_uohm']:.2f} µΩ (L_A={cfg_A['L']:.0f} mm)  |  Ramo B: {res['R_B_uohm']:.2f} µΩ (L_B={cfg_B['L']:.0f} mm)",
        f"Piastra: {res['P_piastra']:.1f} watt  |  Barre a L: {(res['P_A'] + res['P_B']):.1f} watt  |  Totale: {P_tot:.1f} watt ({P_tot / 1000.0:.2f} chilowatt)",
        f"{res['err_perc']:.6f} %"
    ],
    "Esito e Indicazioni DFM": [
        f"Densità: {dens_mat:.0f} kg/m³ | Calore specifico: {csp_mat:.0f} J/(kg·K)",
        "Peso netto parti attive per distinta base carpenteria",
        "Ripartizione uniforme lato ingressi",
        "Bilanciato (45-55 %)" if 45.0 <= p_A <= 55.0 else ("Ramo più carico" if p_A > 55.0 else "Ramo più scarico"),
        "Bilanciato (45-55 %)" if 45.0 <= p_B <= 55.0 else ("Ramo più carico" if p_B > 55.0 else "Ramo più scarico"),
        "Entro K1" if max(J_L_A, J_L_B) <= K1_mat else f"SUPERAMENTO K1 ({K1_mat:.2f} A/mm²): Maggiorare sezione L",
        "Entro K1" if J_sez_piastra_max <= K1_mat else f"SUPERAMENTO K1 ({K1_mat:.2f} A/mm²): Maggiorare spessore/larghezza piastra",
        "Preimpostato in continuità materiale; deselezionare in sidebar per viteria",
        "Per pareggiare correnti asimmetriche variare L_A / L_B o le sezioni",
        "Calore da smaltire tramite ventilazione quadro (CEI EN 61439-1)",
        "Solutore convergente"
    ]
}))
