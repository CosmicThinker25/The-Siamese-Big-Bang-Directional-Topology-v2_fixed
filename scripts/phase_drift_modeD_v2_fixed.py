# ==============================================================
# phase_drift_modeD_v2_fixed.py
# Autor: Cosmic Thinker & ChatGPT (Toko)
# Proyecto: The Siamese Big Bang (v2) — Mode D: Phase Drift Analysis
# ==============================================================

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
OUT_JSON = os.path.join(RES, "phase_drift_modeD_v2.json")
OUT_FIG  = os.path.join(FIG, "Figure_6_PhaseDrift_ModeD_v2.png")

# --- Parámetros ---
DM_MIN = 800.0
B_MIN  = 20.0
STEP_DEG = 2.0
AXIS_RA, AXIS_DEC = 170.0, 40.0

# --- Funciones geométricas ---
def radec_to_unit(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.stack([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)], axis=-1)

def unit(v):
    v = np.asarray(v); n = np.linalg.norm(v)
    return v/n if n != 0 else v

def build_basis(axis_vec):
    a = unit(axis_vec); z = np.array([0,0,1.])
    e1 = np.cross(z,a)
    if np.linalg.norm(e1)<1e-8: e1 = np.cross(np.array([1,0,0]),a)
    e1 = unit(e1); e2 = unit(np.cross(a,e1))
    return e1,e2,a

# --- Datos ---
df = pd.read_csv(CSV_FRB)
df = df[(df["dm_fitb"]>=DM_MIN)&(np.abs(df["gb"])>=B_MIN)].reset_index(drop=True)
V = radec_to_unit(df["ra"], df["dec"])
dm = df["dm_fitb"].to_numpy()
axis = radec_to_unit(np.array([AXIS_RA]), np.array([AXIS_DEC]))[0,:]
e1,e2,a = build_basis(axis)

# --- Curva ΔDM(ψ) ---
psi = np.arange(0,360,STEP_DEG)
dDM=[]
for p in psi:
    n = math.cos(math.radians(p))*e1 + math.sin(math.radians(p))*e2
    mask = (V @ n) >= 0
    if mask.sum()==0 or (~mask).sum()==0:
        dDM.append(np.nan); continue
    d1, d2 = dm[mask].mean(), dm[~mask].mean()
    dDM.append(d1-d2)
dDM = np.array(dDM)

# --- Ajuste seno + drift (modo D) ---
def sine_drift(x, A, phi0, C, D):
    return A*np.sin(np.radians(x)-np.radians(phi0)) + D*(x/180.0-1) + C

m = np.isfinite(dDM)
x, y = psi[m], dDM[m]
A0 = (np.nanmax(y)-np.nanmin(y))/2
phi0 = x[np.nanargmax(y)] if len(x)>0 else 0
C0 = np.nanmean(y)
D0 = 0.0
popt,_ = curve_fit(sine_drift, x, y, p0=[A0, phi0, C0, D0], maxfev=10000)
A, phi0, C, D = popt
y_fit = sine_drift(x,*popt)
R2 = 1 - np.sum((y - y_fit)**2)/np.sum((y - np.nanmean(y))**2)

# --- Guardar y graficar ---
with open(OUT_JSON,"w") as f:
    json.dump({"A":float(A),"phi0":float(phi0),"C":float(C),
               "D":float(D),"R2":float(R2)},f,indent=2)

plt.figure(figsize=(8,4))
plt.plot(x,y,"o",ms=3,alpha=0.5,label="Δ⟨DM⟩(ψ)")
plt.plot(x,y_fit,"r-",lw=2,label=f"Fit: A={A:.1f}, φ₀={phi0:.1f}°, D={D:.2f}")
plt.title(f"Mode D — Phase Drift Fit (v2_fixed)\nR²={R2:.2f}")
plt.xlabel("ψ [deg]"); plt.ylabel("Δ⟨DM⟩ [pc cm$^{-3}$]")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_FIG,dpi=200); plt.close()

print(f"✅ Mode D completado — D={D:.3f}, R²={R2:.2f}")
print(f"→ Figura: {OUT_FIG}")
print(f"→ JSON:   {OUT_JSON}")
