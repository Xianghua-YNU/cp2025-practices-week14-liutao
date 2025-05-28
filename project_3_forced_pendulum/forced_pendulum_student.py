import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def pendulum_equation(t, state, l, g, C, Omega):
    """受迫单摆运动方程"""
    theta, omega = state
    return [omega, -(g/l)*np.sin(theta) + C*np.cos(theta)*np.sin(Omega*t)]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0, 100), y0=None):
    """求解单摆运动"""
    if y0 is None:
        y0 = [0.0, 0.0]
    
    try:
        sol = solve_ivp(
            fun=lambda t, y: pendulum_equation(t, y, l, g, C, Omega),
            t_span=t_span,
            y0=y0,
            t_eval=np.linspace(t_span[0], t_span[1], 2000),
            method='RK45'
        )
        return sol.t, sol.y[0]
    except Exception as e:
        print(f"求解错误: {e}")
        return None, None

def find_resonance(l=0.1, g=9.81, C=2, Omega_min=3, Omega_max=15, steps=30, t_span=(0, 200)):
    """寻找共振频率"""
    Omega_values = np.linspace(Omega_min, Omega_max, steps)
    amplitudes = []
    
    for Omega in Omega_values:
        t, theta = solve_pendulum(l=l, g=g, C=C, Omega=Omega, t_span=t_span)
        if t is not None:
            # 取后1/3时间作为稳态
            steady_theta = theta[len(theta)//3:]
            amplitudes.append(np.max(np.abs(steady_theta)))
        else:
            amplitudes.append(0)
    
    return Omega_values, np.array(amplitudes)

def plot_angle_vs_time(t, theta, title=""):
    """绘制角度-时间图"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def plot_amplitude_vs_frequency(Omega_values, amplitudes, l, g):
    """绘制振幅-频率图"""
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_values, amplitudes, 'b-')
    plt.axvline(np.sqrt(g/l), color='r', linestyle='--', label='Natural frequency')
    plt.title('Amplitude vs Driving Frequency')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 任务1: 基本参数求解
    print("任务1: 基本参数求解")
    t, theta = solve_pendulum()
    if t is not None:
        plot_angle_vs_time(t, theta, "Pendulum Motion (Ω=5 rad/s)")
    
    # 任务2: 寻找共振频率
    print("\n任务2: 寻找共振频率")
    Omega_values, amplitudes = find_resonance()
    
    # 绘制共振曲线
    plot_amplitude_vs_frequency(Omega_values, amplitudes, l=0.1, g=9.81)
    
    # 找到并绘制共振情况
    max_idx = np.argmax(amplitudes)
    Omega_res = Omega_values[max_idx]
    print(f"\n共振频率: {Omega_res:.2f} rad/s")
    
    t_res, theta_res = solve_pendulum(Omega=Omega_res, t_span=(0, 200))
    if t_res is not None:
        plot_angle_vs_time(t_res, theta_res, f"Resonance Case (Ω={Omega_res:.2f} rad/s)")

if __name__ == "__main__":
    main()
