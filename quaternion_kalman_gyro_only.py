import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Helper: Quaternion to Euler angles ===
def quaternion_to_euler(q):
    w, x, y, z = q
    t0 = +2.0 * (w * z + x * y)
    t1 = +1.0 - 2.0 * (y**2 + z**2)
    psi = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    theta = np.arcsin(t2)

    t3 = +2.0 * (w * x + y * z)
    t4 = +1.0 - 2.0 * (x**2 + y**2)
    phi = np.arctan2(t3, t4)

    return np.array([psi, theta, phi])  # yaw, pitch, roll

# === Quaternion Kalman Filter (gyro only) ===
class QuaternionKalmanGyroOnly:
    def __init__(self):
        self.x = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion [w, x, y, z]
        self.P = np.eye(4)
        self.Q = 0.0001 * np.eye(4)  # process noise covariance

    def predict(self, omega, dt):
        wx, wy, wz = omega
        A = np.eye(4) + 0.5 * dt * np.array([
            [ 0,   -wx, -wy, -wz],
            [ wx,   0,   wz, -wy],
            [ wy,  -wz,  0,   wx],
            [ wz,   wy, -wx,  0 ]
        ])
        self.x = A @ self.x
        self.x /= np.linalg.norm(self.x)
        self.P = A @ self.P @ A.T + self.Q
        return self.x

# === Main Execution ===
def run_kalman_gyro_only(gyro_csv="ArsGyro.csv", dt=0.01):
    gyro_df = pd.read_csv(gyro_csv)
    N = len(gyro_df)
    kf = QuaternionKalmanGyroOnly()
    euler_angles = []

    for i in range(N):
        omega = gyro_df.loc[i, ['wx', 'wy', 'wz']].values
        q = kf.predict(omega, dt)
        angles = quaternion_to_euler(q)
        euler_angles.append(angles)

    euler_deg = np.rad2deg(np.array(euler_angles))
    time = np.arange(N) * dt

    # Plotting
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plt.plot(time, euler_deg[:, 2], label='Roll (ϕ)')
    plt.title('Roll'); plt.xlabel('Time (s)'); plt.ylabel('Degrees'); plt.grid(); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(time, euler_deg[:, 1], label='Pitch (θ)')
    plt.title('Pitch'); plt.xlabel('Time (s)'); plt.ylabel('Degrees'); plt.grid(); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(time, euler_deg[:, 0], label='Yaw (ψ)')
    plt.title('Yaw'); plt.xlabel('Time (s)'); plt.ylabel('Degrees'); plt.grid(); plt.legend()

    plt.suptitle("Quaternion Kalman Filter (Gyro Only, No Correction)")
    plt.tight_layout()
    plt.savefig("kalman_gyro_only_orientation.png")
    plt.show()

if __name__ == "__main__":
    run_kalman_gyro_only()
