# ==========================================================
# frb_symmetry_test_v2_fixed.py
# Autor: Cosmic Thinker & ChatGPT (Toko)
# Proyecto: The Siamese Big Bang (v2) — Symmetry Mirror Test
# ==========================================================
import os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- Rutas base ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES  = os.path.join(BASE, "results", "v2_fixed")
FIG  = os.path.join(RES, "figures_output")
os.makedirs(FIG, exist_ok=True)

CSV_FRB  = os.path.join(DATA, "chimefrbcat1.csv")
CSV_OUT  = os.path.join(RES, "symmetry_test_results_v2.csv")
JSON_OUT = os.path.join(RES, "symmetry_test_summary_v2.json")
FIG_OUT  = os.path.join(FIG, "Figure_4_Symmetry_Test_v2.png")

# --- Parámetros ---
DM_MIN = 800.0
B_MIN  = 20.0
STEP_DEG = 2.0
AXIS_RA, AXIS_DEC = 170.0, 40.0
N_PERM = 2000
SEED = 20251031
np.random.seed(SEED)

# --- Funciones ---
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
    e1 = np.cross(z,a)
    if np.linalg.norm(e1)<1e-8:
        e1 = np.cross(np.array([1,0,0]),a)
    e1 = unit(e1)
    e2 = unit(np.cross(a,e1))
    return e1,e2,a

def delta_dm_curve(V, dm, e1, e2, step_deg=2.0):
    psi = np.arange(0,360,step_deg)
    dDM=[]
    for p in psi:
        n = math.cos(math.radians(p))*e1 + math.sin(math.radians(p))*e2
        mask = (V @ n) >= 0
        if mask.sum()==0 or (~mask).sum()==0:
            dDM.append(np.nan); continue
        d1 = dm[mask].mean(); d2 = dm[~mask].mean()
        dDM.append(d1-d2)
    return psi, np.array(dDM)

# --- Carga del catálogo ---
if not os.path.exists(CSV_FRB):
    raise FileNotFoundError(f"No se encuentra {CSV_FRB}")

df = pd.read_csv(CSV_FRB)
for c in ["ra","dec","dm_fitb","gb"]:
    if c not in df.columns:
        raise ValueError(f"Falta columna: {c}")

df = df[(df["dm_fitb"]>=DM_MIN)&(np.abs(df["gb"])>=B_MIN)].reset_index(drop=True)
N_FRB = len(df)
if N_FRB<30:
    raise RuntimeError(f"Demasiado pocos FRBs tras filtros (N={N_FRB})")

V = radec_to_unit(df["ra"].to_numpy(), df["dec"].to_numpy())
axis = radec_to_unit(np.array([AXIS_RA]), np.array([AXIS_DEC]))[0,:]
e1,e2,a = build_basis(axis)

# --- Cálculo principal ---
psi, dDM = delta_dm_curve(V, df["dm_fitb"].to_numpy(), e1, e2, STEP_DEG)
m = np.isfinite(dDM)
dDM = dDM[m]; psi = psi[m]
n = len(psi)//2
d1, d2 = dDM[:n], dDM[n:n*2]
r_mirror, p_mirror = pearsonr(d1, -d2)

# --- Permutaciones ---
perm_r = np.empty(N_PERM)
for i in range(N_PERM):
    dm_perm = np.random.permutation(df["dm_fitb"].to_numpy())
    _, dDMp = delta_dm_curve(V, dm_perm, e1, e2, STEP_DEG)
    dDMp = dDMp[np.isfinite(dDMp)]
    n2 = len(dDMp)//2
    if n2==0:
        perm_r[i]=np.nan; continue
    try:
        perm_r[i] = pearsonr(dDMp[:n2], -dDMp[n2:n2*2])[0]
    except:
        perm_r[i]=np.nan
perm_r = perm_r[np.isfinite(perm_r)]
p_perm_r = (np.sum(np.abs(perm_r) >= np.abs(r_mirror)) + 1) / (len(perm_r)+1)

# --- Guardar resultados ---
pd.DataFrame({"psi_deg":psi, "delta_DM":dDM}).to_csv(CSV_OUT,index=False)
summary = {
    "axis":{"ra_deg":AXIS_RA,"dec_deg":AXIS_DEC},
    "filters":{"DM_min":DM_MIN,"|b|_min":B_MIN,"N_FRB":int(N_FRB)},
    "mirror_test":{
        "r_mirror":float(r_mirror),
        "p_mirror":float(p_mirror),
        "p_perm(|r|)":float(p_perm_r),
        "N_perm":int(len(perm_r))
    }
}
with open(JSON_OUT,"w") as f: json.dump(summary,f,indent=2)

# --- Figura robusta ---
plt.figure(figsize=(6,4))
if len(perm_r) > 1 and np.nanstd(perm_r) > 1e-6:
    nbins = min(40, max(5, int(len(perm_r)/10)))
    plt.hist(perm_r, bins=nbins, alpha=0.8)
else:
    plt.axvline(r_mirror, color="r", linestyle="--")
    plt.text(0.02, 0.9, "Distribución nula degenerada", transform=plt.gca().transAxes)

plt.axvline(r_mirror, color="r", linestyle="--", label=f"r_obs={r_mirror:.3f}")
plt.axvline(-r_mirror, color="r", linestyle="--", alpha=0.5)
plt.title(f"Mirror Correlation Test (v2)\np_perm(|r|)={p_perm_r:.3f}")
plt.xlabel("r_mirror (null distribution)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200)
plt.close()

print(f"✅ Symmetry mirror test completado (v2_fixed). N={N_FRB} FRBs")
print(f"→ r_mirror = {r_mirror:.3f}, p_mirror = {p_mirror:.3g}, p_perm(|r|) = {p_perm_r:.3g}")
print(f"→ Figura: {FIG_OUT}")
print(f"→ JSON:  {JSON_OUT}")
