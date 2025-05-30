import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable

# 修正1：调整参数顺序 (t在前，state在后)
def van_der_pol_ode(t: float, state: np.ndarray, mu: float, omega: float) -> np.ndarray:
    """实现van der Pol振子的一阶微分方程组"""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

# 修正2：调整RK4参数顺序 (t在前)
def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """四阶龙格-库塔方法单步积分"""
    k1 = dt * ode_func(t, state, **kwargs)  # 修正调用顺序
    k2 = dt * ode_func(t + 0.5*dt, state + 0.5*k1, **kwargs)
    k3 = dt * ode_func(t + 0.5*dt, state + 0.5*k2, **kwargs)
    k4 = dt * ode_func(t + dt, state + k3, **kwargs)
    return state + (k1 + 2*k2 + 2*k3 + k4)/6

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ODE求解器"""
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + dt, dt)
    states = [initial_state.copy()]
    for i in range(len(t_values)-1):
        new_state = rk4_step(ode_func, states[-1], t_values[i], dt, **kwargs)
        states.append(new_state)
    return t_values, np.array(states)

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """绘制时间演化图"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, states[:, 0], label='Displacement (x)')
    plt.plot(t, states[:, 1], label='Velocity (v)')
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """绘制相空间轨迹图"""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], lw=0.5)
    plt.xlabel('Displacement (x)')
    plt.ylabel('Velocity (v)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """计算系统能量"""
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """分析极限环特征 (返回振幅和周期步数)"""
    x = states[:, 0]
    # 振幅：取稳态部分的最大绝对值
    steady_state = x[-1000:] if len(x) > 1000 else x
    amplitude = np.max(np.abs(steady_state))
    
    # 周期：通过过零点计算周期步数 (返回步数而非实际时间)
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    if len(zero_crossings) > 2:
        # 计算相邻过零点之间的步数 (半个周期)
        half_period_steps = np.diff(zero_crossings)
        # 完整周期步数 = 2 * 半周期步数
        period_steps = 2 * np.mean(half_period_steps)
    else:
        period_steps = np.nan
        
    return amplitude, period_steps

def main():
    # 基本参数设置
    params = {
        'omega': 1.0,
        't_span': (0, 20),
        'dt': 0.01,
        'initial_state': np.array([1.0, 0.0])
    }
    
    # 任务1：基本实现 (μ=1)
    print("Running basic simulation (μ=1)...")
    t, states = solve_ode(van_der_pol_ode, **params, mu=1.0)
    plot_time_evolution(t, states, 'Time Evolution (μ=1)')
    plot_phase_space(states, 'Phase Space (μ=1)')

    # 任务2：参数影响分析（独立绘图）
    print("\nAnalyzing parameter effects...")
    for mu in [1.0, 2.0, 4.0]:
        # 数值求解
        t, states = solve_ode(van_der_pol_ode, **params, mu=mu)
        
        # 时间演化图
        plot_time_evolution(t, states, f'Time Evolution (μ={mu})')
        
        # 相空间图
        plot_phase_space(states, f'Phase Space (μ={mu})')
    
    # 任务3：能量分析
    print("\nAnalyzing energy evolution...")
    plt.figure(figsize=(10, 5))
    energies = np.array([calculate_energy(state) for state in states])
    plt.plot(t, energies)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution (μ=1)')
    plt.grid(True)
    plt.show()
    
    # 极限环特征分析
    print("\nLimit cycle characteristics:")
    for mu in [1.0, 2.0, 4.0]:
        t, states = solve_ode(van_der_pol_ode, **params, mu=mu)
        amp, period = analyze_limit_cycle(states, t)
        print(f"μ={mu}: Amplitude={amp:.2f}, Period={period:.2f}")

if __name__ == "__main__":
    main()
