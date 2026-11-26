# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:21:00 2025

@author: VK
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads

# --- Parametrit ---
T0, T_env = 90.0, 20.0
rho, c = 7800.0, 500.0
k, h = 15.0, 40.0
Lx, Ly = 0.5, 0.25
nx, ny = 500, 500
# --- Käyttäjän määrittämä ydinmäärä ---
num_threads = 6
set_num_threads(num_threads)

# --- Simulointiaika ja aika-askel ---
sim_time = 1800.0   # sekuntia
dt = 0.04       # sekuntia
nt = int(sim_time / dt)

# --- Johdettuja suureita ---
dx, dy = Lx/nx, Ly/ny
alpha = k / (rho * c)

# --- Alustus ---
T = np.full((ny, nx), T0, dtype=np.float64)

# --- Tarkastelupiste (esim. x=0.03, y=0.03) ---
x_point, y_point = 0.03, 0.03
i_point = int(y_point / dy)
j_point = int(x_point / dx)

# Taulukko tarkastelupisteen lämpötilalle
T_point_history = []

@njit(parallel=True)
def step(T, alpha, dt, dx, dy, h, rho, c, T_env):
    ny, nx = T.shape
    T_new = T.copy()
    # Sisäpisteet rinnakkaisesti
    for i in prange(1, ny-1):
        for j in range(1, nx-1):
            d2Tdx2 = (T[i, j+1] - 2.0*T[i, j] + T[i, j-1]) / dx**2
            d2Tdy2 = (T[i+1, j] - 2.0*T[i, j] + T[i-1, j]) / dy**2
            T_new[i, j] = T[i, j] + alpha * dt * (d2Tdx2 + d2Tdy2)
    # Reunat konvektiolla
    T_new[0, :]   += dt * (2*alpha/dy**2 * (T[1, :] - T[0, :])   - 2*h/(rho*c*dy) * (T[0, :] - T_env))
    T_new[-1, :]  += dt * (2*alpha/dy**2 * (T[-2, :] - T[-1, :]) - 2*h/(rho*c*dy) * (T[-1, :] - T_env))
    T_new[:, 0]   += dt * (2*alpha/dx**2 * (T[:, 1] - T[:, 0])   - 2*h/(rho*c*dx) * (T[:, 0] - T_env))
    T_new[:, -1]  += dt * (2*alpha/dx**2 * (T[:, -2] - T[:, -1]) - 2*h/(rho*c*dx) * (T[:, -1] - T_env))
    return T_new

# --- Aikaintegrointi ---
for step_i in range(nt):
    T = step(T, alpha, dt, dx, dy, h, rho, c, T_env)
    T_point_history.append(T[i_point, j_point])

# --- Piirrä lopputilanne ---
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(T, cmap='hot', origin='lower', extent=[0, Lx, 0, Ly], vmin=T_env, vmax=T0)
plt.colorbar(im, ax=ax, label="Lämpötila (°C)")
ax.set_title(f"Poikkileikkauksen lämpötilakenttä (t = {sim_time:.1f} s)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.tight_layout()
plt.show()

# --- Piirrä tarkastelupisteen lämpötila ajan funktiona ---
time = np.arange(0, sim_time, dt)
plt.figure(figsize=(8, 5))
plt.plot(time, T_point_history, 'r-')
plt.title(f"Tarkastelupisteen lämpötila (x={x_point}, y={y_point})")
plt.xlabel("Aika (s)")
plt.ylabel("Lämpötila (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Tulosta arvot päätteelle ---
print(f"Tarkastelupisteen (x={x_point}, y={y_point}) lämpötilat:")
for t, val in zip(time, T_point_history):
    print(f"t = {t:.1f} s : T = {val:.2f} °C")