import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable

def van_der_pol_ode(t: float, state: np.ndarray, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """van der Pol振子的一阶微分方程组 (solve_ivp兼容版本)"""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """使用solve_ivp求解常微分方程"""
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(
        fun=ode_func,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        args=tuple(kwargs.values()),
        method='RK45'
    )
    return sol.t, sol.y.T

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

def analyze_limit_cycle(t: np.ndarray, states: np.ndarray) -> Tuple[float, float]:
    """分析极限环特征"""
    x = states[:, 0]
    
    # 跳过前40%的瞬态过程（由于总时间较短）
    skip = int(len(x) * 0.4)
    if skip < 100:  # 确保至少有100个点用于分析
        skip = min(100, len(x) - 10)
    
    x_steady = x[skip:]
    t_steady = t[skip:]
    
    # 振幅：取稳态部分的最大绝对值
    amplitude = np.max(np.abs(x_steady))
    
    # 检测过零点（从正到负或负到正）
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    
    if len(zero_crossings) >= 2:
        # 使用实际时间计算周期
        periods = np.diff(t[zero_crossings])
        
        # 只考虑稳态部分的过零点
        steady_zero_crossings = zero_crossings[zero_crossings > skip]
        if len(steady_zero_crossings) >= 2:
            steady_periods = np.diff(t[steady_zero_crossings])
            period = np.mean(steady_periods) * 2  # 两次过零点为一个完整周期
        else:
            period = np.mean(periods) * 2
    else:
        period = np.nan
    
    return amplitude, period

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
