# ==============================================================
# frb_random_axes_test_v2_fixed.py
# Autor: Cosmic Thinker & ChatGPT (Toko)
# Proyecto: The Siamese Big Bang (v2) — Random Axes Control Test
# CLI: --n-axes 1000 --step-deg 2 --seed 20251031 --eta-chunk 50
# ==============================================================

import os, math, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# ---------------- Utilidades ----------------
def radec_to_unit(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.stack([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)], axis=-1)

def unit(v):
    v = np.asarray(v)
    n = np.linalg.norm(v)
    return v/n if n != 0 else v

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

def sine_model(x, A, phi0, C):
    return A*np.sin(np.radians(x)-np.radians(phi0)) + C

def fit_sine(psi, dDM):
    m = np.isfinite(dDM)
    if np.sum(m)<10:
        return np.nan, np.nan, np.nan, np.nan
    x, y = psi[m], dDM[m]
    A0 = (np.nanmax(y)-np.nanmin(y))/2
    phi0 = x[np.nanargmax(y)] if len(x)>0 else 0
    C0 = np.nanmean(y)
    try:
        popt,_ = curve_fit(sine_model, x, y, p0=[A0, phi0, C0], maxfev=10000)
        A, phi0, C = popt
        y_fit = sine_model(x, *popt)
        R2 = 1 - np.sum((y - y_fit)**2)/np.sum((y - np.nanmean(y))**2)
        return A, phi0, C, R2
    except:
        return np.nan, np.nan, np.nan, np.nan

# ---------------- Rutas ----------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES  = os.path.join(BASE, "results", "v2_fixed")
FIG  = os.path.join(RES, "figures_output")
os.makedirs(FIG, exist_ok=True)

CSV_FRB  = os.path.join(DATA, "chimefrbcat1.csv")
JSON_AXIS = os.path.join(RES, "rotational_fit_summary_C_v2.json")
OUT_JSON  = os.path.join(RES, "random_axes_test_summary_v2.json")
OUT_CSV   = os.path.join(RES, "random_axes_distribution_v2.csv")
OUT_FIG   = os.path.join(FIG, "Figure_5_Random_Axes_Test_v2.png")
OUT_LOG   = os.path.join(RES, "random_axes_log.txt")

# ---------------- Argumentos CLI ----------------
parser = argparse.ArgumentParser(description="Random Axes Control Test (v2_fixed)")
parser.add_argument("--n-axes", type=int, default=500, help="Número de ejes aleatorios")
parser.add_argument("--step-deg", type=float, default=2.0, help="Paso angular del barrido (deg)")
parser.add_argument("--seed", type=int, default=20251031, help="Semilla RNG")
parser.add_argument("--eta-chunk", type=int, default=50, help="Frecuencia (iter) para imprimir ETA")
args = parser.parse_args()

N_AXES   = int(args.n_axes)
STEP_DEG = float(args.step_deg)
SEED     = int(args.seed)
CHUNK    = max(1, int(args.eta_chunk))

np.random.seed(SEED)

# ---------------- Carga de datos ----------------
df = pd.read_csv(CSV_FRB)
for c in ["ra","dec","dm_fitb","gb"]:
    if c not in df.columns:
        raise ValueError(f"Falta columna: {c}")

DM_MIN, B_MIN = 800.0, 20.0
df = df[(df["dm_fitb"]>=DM_MIN)&(np.abs(df["gb"])>=B_MIN)].reset_index(drop=True)
N_FRB = len(df)
V = radec_to_unit(df["ra"].to_numpy(), df["dec"].to_numpy())
dm = df["dm_fitb"].to_numpy()

# ---------------- Cargar valores observados ----------------
if os.path.exists(JSON_AXIS):
    real = json.load(open(JSON_AXIS))
    A_obs = abs(real["fit"]["A"]) if "fit" in real else np.nan
    R2_obs = real["fit"]["R2"] if "fit" in real else np.nan
else:
    raise FileNotFoundError("No se encuentra rotational_fit_summary_C_v2.json")

print(f"🔭 Analizando {N_AXES} ejes aleatorios (step={STEP_DEG}°).")
with open(OUT_LOG, "a", encoding="utf-8") as lf:
    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Start N_AXES={N_AXES}, STEP={STEP_DEG}, SEED={SEED}\n")

# ---------------- Bucle con progreso + ETA + log ----------------
A_list, R2_list = [], []
start_time = time.time()

for i in tqdm(range(N_AXES), desc="Ejes procesados", ncols=80):
    ra_rand = np.random.uniform(0,360)
    dec_rand = np.degrees(np.arcsin(np.random.uniform(-1,1)))
    axis = radec_to_unit(np.array([ra_rand]), np.array([dec_rand]))[0,:]
    e1,e2,a = build_basis(axis)
    psi, dDM = delta_dm_curve(V, dm, e1, e2, STEP_DEG)
    A, phi0, C, R2 = fit_sine(psi, dDM)
    if not np.isnan(A):
        A_list.append(abs(A))
        R2_list.append(R2)

    if (i+1) % CHUNK == 0:
        elapsed = time.time() - start_time
        eta = elapsed * (N_AXES - (i+1)) / (i+1)
        msg = f"⏱  Progreso {i+1}/{N_AXES} — ETA: {eta/60:.1f} min"
        print(msg)
        with open(OUT_LOG, "a", encoding="utf-8") as lf:
            lf.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

# ---------------- Estadística ----------------
A_list, R2_list = np.array(A_list), np.array(R2_list)
z_A  = (A_obs - np.nanmean(A_list)) / np.nanstd(A_list)
z_R2 = (R2_obs - np.nanmean(R2_list)) / np.nanstd(R2_list)
p_A  = np.mean(A_list >= A_obs)
p_R2 = np.mean(R2_list >= R2_obs)

# ---------------- Guardar ----------------
pd.DataFrame({"A_random":A_list, "R2_random":R2_list}).to_csv(OUT_CSV, index=False)
summary = {
    "N_axes":int(N_AXES),
    "N_FRB":int(N_FRB),
    "A_obs":float(A_obs),
    "R2_obs":float(R2_obs),
    "A_mean":float(np.nanmean(A_list)),
    "R2_mean":float(np.nanmean(R2_list)),
    "z_A":float(z_A),
    "z_R2":float(z_R2),
    "p_A":float(p_A),
    "p_R2":float(p_R2),
    "step_deg": STEP_DEG,
    "seed": SEED
}
with open(OUT_JSON,"w") as f: json.dump(summary,f,indent=2)

# ---------------- Figura ----------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(A_list, bins=30, alpha=0.7)
plt.axvline(A_obs, color="r", linestyle="--", label=f"A_obs={A_obs:.1f}")
plt.title(f"|A| distribution (N={N_AXES})\n z={z_A:.2f}, p={p_A:.3f}")
plt.xlabel("|A| [pc cm$^{-3}$]"); plt.ylabel("Count"); plt.legend()

plt.subplot(1,2,2)
plt.hist(R2_list, bins=30, alpha=0.7)
plt.axvline(R2_obs, color="r", linestyle="--", label=f"R$^2$_obs={R2_obs:.2f}")
plt.title(f"R$^2$ distribution (N={N_AXES})\n z={z_R2:.2f}, p={p_R2:.3f}")
plt.xlabel("R$^2$"); plt.legend()

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
plt.close()

elapsed = (time.time()-start_time)/60
print(f"\n✅ Random Axes Test completado en {elapsed:.1f} min.")
print(f"→ z_A={z_A:.2f}, p_A={p_A:.3f}")
print(f"→ z_R2={z_R2:.2f}, p_R2={p_R2:.3f}")
print(f"→ Figura: {OUT_FIG}")
print(f"→ JSON:   {OUT_JSON}")

with open(OUT_LOG, "a", encoding="utf-8") as lf:
    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] End z_A={z_A:.2f}, p_A={p_A:.3f}, z_R2={z_R2:.2f}, p_R2={p_R2:.3f}\n")
