# ==============================================================
# frb_through_axis_map_v2.py
# Autor: Cosmic Thinker & ChatGPT (Toko)
# Proyecto: The Siamese Big Bang (v2_fixed)
# Descripción: Mapa Mollweide de FRBs (DM_fitb) con el eje siamés
# ==============================================================

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# --- Rutas ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES  = os.path.join(BASE, "results", "v2_fixed")
FIG  = os.path.join(RES, "figures_output")
os.makedirs(FIG, exist_ok=True)

CSV_FRB = os.path.join(DATA, "chimefrbcat1.csv")
OUT_FIG = os.path.join(FIG, "Figure_7_ThroughAxis_Map_v2.png")

# --- Parámetros y filtros ---
DM_MIN, B_MIN = 800.0, 20.0
AXIS_RA, AXIS_DEC = 170.0, 40.0   # eje siamés nominal (deg)

# --- Cargar datos ---
df = pd.read_csv(CSV_FRB)
need = {"ra","dec","dm_fitb","gb"}
if not need.issubset(df.columns):
    raise ValueError(f"Faltan columnas en {CSV_FRB}. Requiere: {sorted(need)}")

df = df[(df["dm_fitb"]>=DM_MIN) & (np.abs(df["gb"])>=B_MIN)].reset_index(drop=True)

# --- Conversión a radianes y proyección Mollweide ---
ra  = np.deg2rad(df["ra"].to_numpy())
dec = np.deg2rad(df["dec"].to_numpy())
dm  = df["dm_fitb"].to_numpy()

# poner RA en [-pi, pi] y virar eje X a la izquierda (convención astro)
ra_plot = np.remainder(ra + 2*np.pi, 2*np.pi)
ra_plot[ra_plot > np.pi] -= 2*np.pi
x = -ra_plot
y = dec

# --- Preparar figura ---
plt.figure(figsize=(10,5.6))
ax = plt.subplot(111, projection="mollweide")
ax.grid(True, alpha=0.35)

# rango centrado en la mediana para resaltar exceso/defecto
norm = TwoSlopeNorm(vcenter=np.nanmedian(dm))
sc = ax.scatter(x, y, c=dm, s=16, alpha=0.85, cmap="coolwarm", norm=norm, linewidths=0.2, edgecolors="none")
cb = plt.colorbar(sc, orientation="horizontal", pad=0.06)
cb.set_label(r"DM$_{\rm fitb}$  [pc cm$^{-3}$]")

# --- Marcar el eje siamés (polo y antípoda) ---
ra_ax, dec_ax = np.deg2rad(AXIS_RA), np.deg2rad(AXIS_DEC)
ra_ax = np.remainder(ra_ax + 2*np.pi, 2*np.pi)
if ra_ax > np.pi: ra_ax -= 2*np.pi
ax.plot([-ra_ax], [dec_ax], marker="*", ms=12, color="k", mec="w", mew=0.8, zorder=5)
# antípoda
ra_ax2 = ra_ax + np.pi
if ra_ax2 > np.pi: ra_ax2 -= 2*np.pi
ax.plot([-ra_ax2], [-dec_ax], marker="*", ms=10, color="k", mec="w", mew=0.8, zorder=5)

# --- Título y guardar ---
ax.set_title("FRB Dispersion Map through the Siamese Axis (v2_fixed)", pad=14)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=220)
plt.close()

print("✅ Mapa completado.")
print(f"→ Figura: {OUT_FIG}")
print(f"→ N_FRB (filtrados): {len(df)}")
