import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / l) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0, 100), y0=[0, 0]):
    sol = solve_ivp(
        fun=forced_pendulum_ode,
        t_span=t_span,
        y0=y0,
        args=(l, g, C, Omega),
        t_eval=np.linspace(t_span[0], t_span[1], 2000),
        method='RK45'
    )
    return sol.t, sol.y[0]

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0, 200), y0=[0, 0]):
    if Omega_range is None:
        Omega0 = np.sqrt(g / l)
        Omega_range = np.linspace(Omega0 / 2, 2 * Omega0, 50)
    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l=l, g=g, C=C, Omega=Omega, t_span=t_span, y0=y0)
        mask = t >= 50  # 忽略前50秒的暂态过程
        if np.sum(mask) == 0:
            A = 0.0
        else:
            steady_theta = theta[mask]
            A = np.max(np.abs(steady_theta))
        amplitudes.append(A)
    return np.array(Omega_range), np.array(amplitudes)

def plot_results(t, theta, title):
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def main():
    # 任务1: 特定参数下的数值解与可视化
    t1, theta1 = solve_pendulum()
    plot_results(t1, theta1, "Task 1: θ(t) vs t (Ω=5 rad/s)")
    
    # 任务2: 探究共振现象
    Omega_range, amplitudes = find_resonance()
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes, 'b-')
    plt.xlabel('Driving Frequency Ω (rad/s)')
    plt.ylabel('Steady State Amplitude (rad)')
    plt.title('Amplitude vs Driving Frequency')
    plt.grid(True)
    plt.show()
    
    # 找到共振频率并绘制共振情况
    max_index = np.argmax(amplitudes)
    Omega_res = Omega_range[max_index]
    print(f"Resonance frequency found at Ω = {Omega_res:.2f} rad/s")
    
    t_res, theta_res = solve_pendulum(Omega=Omega_res, t_span=(0, 200))
    plot_results(t_res, theta_res, f"Resonance Case: θ(t) vs t (Ω={Omega_res:.2f} rad/s)")

if __name__ == '__main__':
    main()
