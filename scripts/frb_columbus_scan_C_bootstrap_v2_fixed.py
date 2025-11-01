# ==========================================================
# frb_columbus_scan_C_bootstrap_v2_fixed.py
# Autor: Cosmic Thinker & ChatGPT (Toko)
# Proyecto: The Siamese Big Bang (v2) — Permutation + Bootstrap
# Descripción:
#   - Reproduce el barrido rotacional (Modo C) con dm_fitb.
#   - Permutaciones (shuffle de dm_fitb) → p_perm(|A|).
#   - Bootstrap (resampling con reemplazo) → CI95% de A, phi0, R2.
# Salidas:
#   results/v2_fixed/rotational_fit_summary_C_bootstrap_v2.json
#   results/v2_fixed/rotational_bootstrap_samples_C_v2.csv
#   results/v2_fixed/rotational_permutation_null_C_v2.csv
#   results/v2_fixed/figures_output/Figure_3_Bootstrap_Permutation_v2.png
# ==========================================================
import os, math, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# --------- Rutas ---------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES  = os.path.join(BASE, "results", "v2_fixed")
FIG  = os.path.join(RES, "figures_output")
os.makedirs(FIG, exist_ok=True)

CSV_FRB   = os.path.join(DATA, "chimefrbcat1.csv")
CSV_BOOT  = os.path.join(RES,  "rotational_bootstrap_samples_C_v2.csv")
CSV_NULL  = os.path.join(RES,  "rotational_permutation_null_C_v2.csv")
JSON_OUT  = os.path.join(RES,  "rotational_fit_summary_C_bootstrap_v2.json")
FIG_OUT   = os.path.join(FIG,  "Figure_3_Bootstrap_Permutation_v2.png")

# --------- Parámetros ---------
DM_MIN   = 800.0
B_MIN    = 20.0
STEP_DEG = 2.0
AXIS_RA, AXIS_DEC = 170.0, 40.0

N_PERM   = 2000   # permutaciones (nulo)
N_BOOT   = 1000   # réplicas bootstrap (IC95%)
SEED     = 20251031

np.random.seed(SEED)

# --------- Utilidades ---------
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
    a  = unit(axis_vec)
    z  = np.array([0,0,1.])
    e1 = np.cross(z, a)
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(np.array([1,0,0]), a)
    e1 = unit(e1)
    e2 = unit(np.cross(a, e1))
    return e1, e2, a

def sine_model(x, A, phi0, C):
    return A*np.sin(np.deg2rad(x - phi0)) + C

def fit_sine(x, y):
    if len(x) < 5 or np.allclose(np.nanvar(y), 0):
        return np.nan, np.nan, np.nan, np.nan
    A0   = 0.5*(np.nanmax(y) - np.nanmin(y))
    C0   = np.nanmean(y)
    phi0 = x[np.nanargmax(y)]
    try:
        popt, _ = curve_fit(sine_model, x, y, p0=[A0, phi0, C0], maxfev=10000)
        A, phi0, C = popt
        yhat = sine_model(x, *popt)
        R2 = 1 - (np.var(y - yhat) / np.var(y))
        return float(A), float(phi0 % 360), float(C), float(R2)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def rotational_deltaDM(V_unit, dm_vals, e1, e2, step_deg=2.0):
    psi = np.arange(0, 360, step_deg)
    dDM = []
    for p in psi:
        n = math.cos(math.radians(p))*e1 + math.sin(math.radians(p))*e2
        mask = (V_unit @ n) >= 0
        if mask.sum()==0 or (~mask).sum()==0:
            dDM.append(np.nan); continue
        d1 = dm_vals[mask].mean()
        d2 = dm_vals[~mask].mean()
        dDM.append(d1 - d2)
    dDM = np.array(dDM, dtype=float)
    m = np.isfinite(dDM)
    A, phi0, C, R2 = fit_sine(psi[m], dDM[m])
    return psi, dDM, A, phi0, C, R2

# --------- Carga y filtro ---------
if not os.path.exists(CSV_FRB):
    raise FileNotFoundError(f"No se encuentra {CSV_FRB}")

df = pd.read_csv(CSV_FRB)
for c in ["ra","dec","dm_fitb","gb"]:
    if c not in df.columns:
        raise ValueError(f"Falta columna requerida: {c}")

df = df[(df["dm_fitb"]>=DM_MIN) & (np.abs(df["gb"])>=B_MIN)].reset_index(drop=True)
N_FRB = len(df)
if N_FRB < 30:
    raise RuntimeError(f"Demasiado pocos FRBs tras filtros (N={N_FRB})")

# Geometría fija
V = radec_to_unit(df["ra"].to_numpy(), df["dec"].to_numpy())
axis = radec_to_unit(np.array([AXIS_RA]), np.array([AXIS_DEC]))[0,:]
e1, e2, a = build_basis(axis)

# --------- Observado ---------
psi_obs, dDM_obs, A_obs, phi0_obs, C_obs, R2_obs = rotational_deltaDM(V, df["dm_fitb"].to_numpy(), e1, e2, STEP_DEG)

# --------- Permutaciones (nulo) ---------
perm_absA = np.empty(N_PERM, dtype=float)
perm_R2   = np.empty(N_PERM, dtype=float)

dm_arr = df["dm_fitb"].to_numpy().copy()
for i in range(N_PERM):
    np.random.shuffle(dm_arr)             # reetiqueta DM entre FRBs
    _, _, A_p, _, _, R2_p = rotational_deltaDM(V, dm_arr, e1, e2, STEP_DEG)
    perm_absA[i] = np.abs(A_p) if np.isfinite(A_p) else np.nan
    perm_R2[i]   = R2_p if np.isfinite(R2_p) else np.nan

# Limpieza NaN
perm_absA = perm_absA[np.isfinite(perm_absA)]
perm_R2   = perm_R2[np.isfinite(perm_R2)]

p_perm_absA = float((perm_absA >= np.abs(A_obs)).mean()) if perm_absA.size>0 else float("nan")
p_perm_R2   = float((perm_R2   >= R2_obs).mean())        if perm_R2.size>0   else float("nan")

# --------- Bootstrap (IC95%) ---------
boot_A, boot_phi0, boot_R2 = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)

idx_all = np.arange(N_FRB)
for i in range(N_BOOT):
    idx = np.random.choice(idx_all, size=N_FRB, replace=True)
    Vb  = V[idx, :]
    DMb = df["dm_fitb"].to_numpy()[idx]
    _, _, A_b, phi0_b, _, R2_b = rotational_deltaDM(Vb, DMb, e1, e2, STEP_DEG)
    boot_A[i], boot_phi0[i], boot_R2[i] = A_b, phi0_b, R2_b

# IC95%
def ci95(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan"), float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)),
            float(np.percentile(x, 50.0)),
            float(np.percentile(x, 97.5))]

A_ci   = ci95(boot_A)
phi_ci = ci95(boot_phi0)
R2_ci  = ci95(boot_R2)

# --------- Guardados ---------
pd.DataFrame({
    "A_boot": boot_A, "phi0_boot": boot_phi0, "R2_boot": boot_R2
}).to_csv(CSV_BOOT, index=False)

pd.DataFrame({
    "absA_perm": perm_absA, "R2_perm": perm_R2
}).to_csv(CSV_NULL, index=False)

summary = {
    "timestamp": datetime.utcnow().isoformat()+"Z",
    "axis": {"ra_deg": AXIS_RA, "dec_deg": AXIS_DEC},
    "filters": {"DM_min": DM_MIN, "|b|_min": B_MIN, "N_FRB": int(N_FRB)},
    "observed_fit": {"A": float(A_obs), "phi0_deg": float(phi0_obs), "C": float(C_obs), "R2": float(R2_obs)},
    "permutation_test": {
        "N_perm": N_PERM,
        "p_perm_absA": p_perm_absA,
        "p_perm_R2":   p_perm_R2,
        "absA_null_stats": {
            "mean": float(np.nanmean(perm_absA)) if perm_absA.size>0 else float("nan"),
            "p95":  float(np.nanpercentile(perm_absA,95)) if perm_absA.size>0 else float("nan")
        },
        "R2_null_stats": {
            "mean": float(np.nanmean(perm_R2)) if perm_R2.size>0 else float("nan"),
            "p95":  float(np.nanpercentile(perm_R2,95)) if perm_R2.size>0 else float("nan")
        }
    },
    "bootstrap": {
        "N_boot": N_BOOT,
        "A_CI95":   {"lo": A_ci[0],   "med": A_ci[1],   "hi": A_ci[2]},
        "phi_CI95": {"lo": phi_ci[0], "med": phi_ci[1], "hi": phi_ci[2]},
        "R2_CI95":  {"lo": R2_ci[0],  "med": R2_ci[1],  "hi": R2_ci[2]}
    }
}
with open(JSON_OUT, "w") as f:
    json.dump(summary, f, indent=2)

# --------- Figura ---------
plt.figure(figsize=(10,4.8))

# Hist nulo |A|
plt.subplot(1,2,1)
plt.hist(perm_absA, bins=40, alpha=0.85)
plt.axvline(np.abs(A_obs), linestyle="--")
plt.xlabel("|A| under permutations [pc cm$^{-3}$]")
plt.ylabel("Count")
plt.title(f"Permutation null (N={len(perm_absA)}), p={p_perm_absA:.3g}")

# Hist bootstrap A
plt.subplot(1,2,2)
plt.hist(boot_A[np.isfinite(boot_A)], bins=40, alpha=0.85)
plt.axvline(A_ci[0], linestyle=":")
plt.axvline(A_ci[1], linestyle="--")
plt.axvline(A_ci[2], linestyle=":")
plt.axvline(A_obs, linestyle="-")
plt.xlabel("A bootstrap [pc cm$^{-3}$]")
plt.ylabel("Count")
plt.title(f"Bootstrap (N={N_BOOT}) — A_obs={A_obs:.1f}")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200)
plt.close()

print(f"✅ Bootstrap + Permutation completados (v2). N={N_FRB} FRBs.")
print(f"→ JSON:  {JSON_OUT}")
print(f"→ CSV:   {CSV_BOOT} (bootstrap), {CSV_NULL} (null)")
print(f"→ FIG:   {FIG_OUT}")
