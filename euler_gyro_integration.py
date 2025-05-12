import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load gyroscope data ===
gyro_df = pd.read_csv("ArsGyro.csv")  # Make sure this file exists in the same directory
dt = 0.01
N = len(gyro_df)

# === Initialize orientation (roll, pitch, yaw) ===
phi = 0.0   # roll (ϕ)
theta = 0.0 # pitch (θ)
psi = 0.0   # yaw (ψ)

euler_angles = []

# === Nonlinear Euler integration ===
for i in range(N):
    p = gyro_df.loc[i, 'wx']  # roll rate
    q = gyro_df.loc[i, 'wy']  # pitch rate
    r = gyro_df.loc[i, 'wz']  # yaw rate

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)
    cos_theta = np.cos(theta)

    # Derivatives of Euler angles from body rates
    phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    theta_dot = q * cos_phi - r * sin_phi
    psi_dot = q * sin_phi / cos_theta + r * cos_phi / cos_theta

    # Integrate over time
    phi += dt * phi_dot
    theta += dt * theta_dot
    psi += dt * psi_dot

    euler_angles.append([phi, theta, psi])

# Convert to degrees
euler_angles = np.rad2deg(np.array(euler_angles))
time = np.arange(N) * dt

# === Plot ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(time, euler_angles[:, 0], label="Roll (ϕ)")
plt.xlabel("Time (s)"); plt.ylabel("Degrees"); plt.title("Roll"); plt.grid(); plt.legend()

plt.subplot(1, 3, 2)
plt.plot(time, euler_angles[:, 1], label="Pitch (θ)")
plt.xlabel("Time (s)"); plt.ylabel("Degrees"); plt.title("Pitch"); plt.grid(); plt.legend()

plt.subplot(1, 3, 3)
plt.plot(time, euler_angles[:, 2], label="Yaw (ψ)")
plt.xlabel("Time (s)"); plt.ylabel("Degrees"); plt.title("Yaw"); plt.grid(); plt.legend()

plt.suptitle("Euler Angle Integration from Gyroscope (Nonlinear)")
plt.tight_layout()
plt.savefig("euler_gyro_integrated_plot.png")
plt.show()


