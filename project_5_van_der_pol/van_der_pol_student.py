import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """van der Pol振子的一阶微分方程组"""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """四阶龙格-库塔单步积分"""
    k1 = dt * ode_func(state, t, **kwargs)
    k2 = dt * ode_func(state + 0.5*k1, t + 0.5*dt, **kwargs)
    k3 = dt * ode_func(state + 0.5*k2, t + 0.5*dt, **kwargs)
    k4 = dt * ode_func(state + k3, t + dt, **kwargs)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ODE求解器"""
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + dt, dt)
    states = np.zeros((len(t_values), len(initial_state)))
    states[0] = initial_state
    for i in range(len(t_values)-1):
        states[i+1] = rk4_step(ode_func, states[i], t_values[i], dt, **kwargs)
    return t_values, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """绘制时间演化图"""
    plt.figure(figsize=(10, 4))
    plt.plot(t, states[:, 0], label='Position (x)')
    plt.plot(t, states[:, 1], label='Velocity (v)')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """绘制相空间轨迹"""
    plt.figure(figsize=(6, 6))
    plt.plot(states[:, 0], states[:, 1], linewidth=0.8)
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """计算系统能量"""
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(t: np.ndarray, states: np.ndarray) -> Tuple[float, float]:
    """分析极限环特征"""
    x = states[:, 0]
    # 寻找波峰位置
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    if len(peaks) < 2:
        return (0.0, 0.0)
    # 计算平均周期
    peak_times = t[peaks]
    periods = np.diff(peak_times)
    avg_period = np.mean(periods)
    # 计算振幅
    amplitude = np.max(np.abs(x[peaks]))
    return amplitude, avg_period

def main():
    # 基础参数设置
    base_mu = 1.0
    omega = 1.0
    t_span = (0, 40)  # 延长观察时间
    dt = 0.01
    initial_state = np.array([1.0, 0.0])

    # ===== 任务1：基本实现 =====
    print("Running Task 1: Basic Implementation...")
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=base_mu)
    plot_time_evolution(t, states, f"Time Evolution (μ={base_mu})")
    plot_phase_space(states, f"Phase Space (μ={base_mu})")

    # ===== 任务2：参数影响分析 =====
    print("\nRunning Task 2: Parameter Analysis...")
    mu_values = [1.0, 2.0, 4.0]
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu)
        plot_phase_space(states, f"Phase Space (μ={mu})")
        amplitude, period = analyze_limit_cycle(t, states)
        print(f"μ={mu}: Amplitude={amplitude:.2f}, Period={period:.2f}")

    # ===== 任务3：能量分析 =====
    print("\nRunning Task 3: Energy Analysis...")
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=base_mu)
    energies = np.array([calculate_energy(s, omega) for s in states])
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, energies)
    plt.title("Energy Evolution (μ=1.0)")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ===== 扩展分析：不同初始条件 =====
    print("\nAdditional Analysis: Different Initial Conditions")
    initial_conditions = [
        [2.0, 0.0],
        [0.5, 0.0],
        [1.0, 2.0]
    ]
    for ic in initial_conditions:
        t, states = solve_ode(van_der_pol_ode, np.array(ic), t_span, dt, mu=base_mu)
        plot_phase_space(states, f"Initial Condition {ic}")

if __name__ == "__main__":
    main()
