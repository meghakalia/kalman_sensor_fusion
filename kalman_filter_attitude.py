import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======= Quaternion Utilities =======
def euler_to_quaternion(psi, theta, phi):
    cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
    cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
    cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)

    q = np.zeros(4)
    q[0] = cr * cp * cy + sr * sp * sy
    q[1] = sr * cp * cy - cr * sp * sy
    q[2] = cr * sp * cy + sr * cp * sy
    q[3] = cr * cp * sy - sr * sp * cy
    return q

def quaternion_to_euler(q):
    w, x, y, z = q
    t0, t1 = +2.0 * (w*z + x*y), +1.0 - 2.0 * (y*y + z*z)
    psi = np.arctan2(t0, t1)

    t2 = +2.0 * (w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    theta = np.arcsin(t2)

    t3, t4 = +2.0 * (w*x + y*z), +1.0 - 2.0 * (x*x + y*y)
    phi = np.arctan2(t3, t4)

    return np.array([psi, theta, phi])

# ======= Kalman Filter =======
class QuaternionKalmanFilter:
    def __init__(self):
        self.x = np.array([1, 0, 0, 0])
        self.P = np.eye(4)
        self.Q = 0.0001 * np.eye(4)
        self.R = 10 * np.eye(4)
        self.H = np.eye(4)

    def predict(self, omega, dt):
        A = np.eye(4) + 0.5 * dt * np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        self.x = A @ self.x
        self.x /= np.linalg.norm(self.x)
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x += K @ (z - self.H @ self.x)
        self.x /= np.linalg.norm(self.x)
        self.P = self.P - K @ self.H @ self.P

    def get_euler(self):
        return quaternion_to_euler(self.x)

# ======= Main Function =======
def run_kalman_with_plot(accel_csv="ArsAccel.csv", gyro_csv="ArsGyro.csv", dt=0.01, output_csv="FilteredAngles.csv"):
    accel_df = pd.read_csv(accel_csv)
    gyro_df = pd.read_csv(gyro_csv)
    assert len(accel_df) == len(gyro_df), "Mismatch in accel and gyro data length"
    N = len(accel_df)

    kf = QuaternionKalmanFilter()
    filtered = []
    raw_pitch = []
    raw_roll = []

    for k in range(N):
        fx, fy, fz = accel_df.loc[k, ['fx', 'fy', 'fz']]
        wx, wy, wz = gyro_df.loc[k, ['wx', 'wy', 'wz']]

        g = np.linalg.norm([fx, fy, fz])
        theta = np.arcsin(fx / g)
        phi = np.arcsin(-fy / (g * np.cos(theta)))
        psi = 0.0

        raw_pitch.append(theta * 180 / np.pi)
        raw_roll.append(phi * 180 / np.pi)

        z = euler_to_quaternion(psi, theta, phi)
        kf.predict([wx, wy, wz], dt)
        kf.update(z)
        filtered.append(kf.get_euler() * 180 / np.pi)

    filtered_df = pd.DataFrame(filtered, columns=["yaw", "pitch", "roll"])
    filtered_df.to_csv(output_csv, index=False)

    # === Plot ===
    t = np.arange(N) * dt
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t, raw_pitch, label='Raw Pitch (accel)', alpha=0.5)
    plt.plot(t, filtered_df['pitch'], label='Kalman Pitch')
    plt.xlabel('Time (s)'); plt.ylabel('Pitch (°)')
    plt.title('Pitch Angle'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, raw_roll, label='Raw Roll (accel)', alpha=0.5)
    plt.plot(t, filtered_df['roll'], label='Kalman Roll')
    plt.xlabel('Time (s)'); plt.ylabel('Roll (°)')
    plt.title('Roll Angle'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kalman_with_plot()
