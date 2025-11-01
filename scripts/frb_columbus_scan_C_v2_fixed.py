# ==========================================================
# frb_columbus_scan_C_v2_fixed.py
# Autor: Cosmic Thinker & ChatGPT (Toko)
# Proyecto: The Siamese Big Bang (v2) — Columbus Scan (Mode C)
# ==========================================================
import os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Rutas base ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES  = os.path.join(BASE, "results", "v2_fixed")
FIG  = os.path.join(RES, "figures_output")
os.makedirs(FIG, exist_ok=True)

CSV_FRB  = os.path.join(DATA, "chimefrbcat1.csv")
CSV_OUT  = os.path.join(RES, "rotational_sweep_C_v2.csv")
JSON_OUT = os.path.join(RES, "rotational_fit_summary_C_v2.json")
FIG_OUT  = os.path.join(FIG, "Figure_2_RotationalSweep_C_v2.png")

# --- Parámetros físicos ---
DM_MIN  = 800.0
B_MIN   = 20.0
STEP_DEG = 2.0
AXIS_RA, AXIS_DEC = 170.0, 40.0

# --- Funciones auxiliares ---
def radec_to_unit(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.stack([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)], axis=-1)

def unit(v):
    v = np.asarray(v)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def build_basis(axis_vec):
    a = unit(axis_vec)
    z = np.array([0,0,1.])
    e1 = np.cross(z, a)
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(np.array([1,0,0]), a)
    e1 = unit(e1)
    e2 = unit(np.cross(a, e1))
    return e1, e2, a

def sine_model(x, A, phi0, C):
    return A*np.sin(np.deg2rad(x - phi0)) + C

def fit_sine(x, y):
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    A0 = 0.5*(np.nanmax(y)-np.nanmin(y))
    C0 = np.nanmean(y)
    phi0 = x[np.nanargmax(y)]
    try:
        popt,_ = curve_fit(sine_model, x, y, p0=[A0, phi0, C0], maxfev=10000)
        A,phi0,C = popt
        yhat = sine_model(x, *popt)
        R2 = 1 - np.var(y - yhat)/np.var(y)
        return float(A), float(phi0 % 360), float(C), float(R2)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

# --- Carga del catálogo ---
if not os.path.exists(CSV_FRB):
    raise FileNotFoundError(f"No se encuentra {CSV_FRB}")

df = pd.read_csv(CSV_FRB)
cols_needed = ["ra", "dec", "dm_fitb", "gb"]
if not all(c in df.columns for c in cols_needed):
    raise ValueError(f"Faltan columnas necesarias: {cols_needed}")

# --- Filtro ---
df = df[(df["dm_fitb"]>=DM_MIN) & (np.abs(df["gb"])>=B_MIN)].reset_index(drop=True)
N_FRB = len(df)
if N_FRB < 30:
    raise RuntimeError(f"Demasiado pocos FRBs tras el filtro (N={N_FRB})")

V = radec_to_unit(df["ra"].to_numpy(), df["dec"].to_numpy())
axis = radec_to_unit(np.array([AXIS_RA]), np.array([AXIS_DEC]))[0,:]
e1,e2,a = build_basis(axis)

# --- Barrido rotacional completo ---
psi_grid = np.arange(0, 360, STEP_DEG)
deltaDM = []
for psi in psi_grid:
    n = math.cos(math.radians(psi))*e1 + math.sin(math.radians(psi))*e2
    mask = (V @ n) >= 0
    if mask.sum()==0 or (~mask).sum()==0:
        deltaDM.append(np.nan); continue
    dm1 = df.loc[mask, "dm_fitb"].mean()
    dm2 = df.loc[~mask, "dm_fitb"].mean()
    deltaDM.append(dm1 - dm2)

deltaDM = np.array(deltaDM)
mask = np.isfinite(deltaDM)
A, phi0, C, R2 = fit_sine(psi_grid[mask], deltaDM[mask])

# --- Guardar CSV y JSON ---
pd.DataFrame({"psi_deg": psi_grid, "delta_DM": deltaDM}).to_csv(CSV_OUT, index=False)
summary = {
    "axis":{"ra_deg":AXIS_RA, "dec_deg":AXIS_DEC},
    "filters":{"DM_min":DM_MIN, "|b|_min":B_MIN, "N_FRB":int(N_FRB)},
    "fit":{"A":A,"phi0_deg":phi0,"C":C,"R2":R2}
}
with open(JSON_OUT,"w") as f: json.dump(summary,f,indent=2)

# --- Figura ---
plt.figure(figsize=(8,4))
plt.plot(psi_grid, deltaDM, "o-", alpha=0.6, label="Δ⟨DM⟩ data")
plt.plot(psi_grid, sine_model(psi_grid, A, phi0, C), "r--", lw=1.5, label="fit")
plt.xlabel("Rotation ψ [deg]")
plt.ylabel("Δ⟨DM⟩ [pc cm$^{-3}$]")
plt.title(f"Columbus Scan Mode C (v2) — dm_fitb, |A|={abs(A):.1f}, R²={R2:.2f}")
plt.legend()
plt.grid(ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200)
plt.close()

print(f"✅ Columbus Scan Mode C (v2) completado con N={N_FRB} FRBs.")
print(f"→ Figura: {FIG_OUT}")
print(f"→ JSON:  {JSON_OUT}")
print(f"→ CSV:   {CSV_OUT}")
