import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(t, state, mu=1.0, omega=1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    """
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])
    
def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    """
    k1 = dt * ode_func(state, t, **kwargs)
    k2 = dt * ode_func(state + 0.5*k1, t + 0.5*dt, **kwargs)
    k3 = dt * ode_func(state + 0.5*k2, t + 0.5*dt, **kwargs)
    k4 = dt * ode_func(state + k3, t + dt, **kwargs)
    return state + (k1 + 2*k2 + 2*k3 + k4)/6
    
def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ODE求解器"""
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + dt, dt)
    states = [initial_state.copy()]
    for t in t_values[:-1]:
        new_state = rk4_step(ode_func, states[-1], t, dt, **kwargs)
        states.append(new_state)
    return t_values, np.array(states)
    
def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    """
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
    """
    绘制相空间轨迹。
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], lw=0.5)
    plt.xlabel('Displacement (x)')
    plt.ylabel('Velocity (v)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算系统能量
    """
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2
    
def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """分析极限环特征（内部定义时间参数t）"""
    # 定义时间参数（假设dt=0.01，与主函数一致）
    dt = 0.01
    t = np.arange(0, len(states)) * dt
    
    x = states[:, 0]
    
    # 振幅计算（取稳态部分的最大绝对值）
    skip = int(len(x) * 0.5)
    x_steady = x[skip:]
    amplitude = np.max(np.abs(x_steady))
    
    # 周期计算（基于内部生成的t）
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    if len(zero_crossings) >= 2:
        periods = np.diff(t[zero_crossings])
        period = np.mean(periods) * 2  # 两次过零点为一个完整周期
    else:
        period = np.nan
    
    return amplitude, period
    
def main():
    # 设置基本参数
    params = {
        'omega': 1.0,
        't_span': (0, 20),
        'dt': 0.01,
        'initial_state': np.array([1.0, 0.0])
    }
    
    # TODO: 任务1 - 基本实现
    print("Running basic simulation (μ=1)...")
    t, states = solve_ode(van_der_pol_ode, **params, mu=1.0)
    plot_time_evolution(t, states, 'Time Evolution (μ=1)')
    plot_phase_space(states, 'Phase Space (μ=1)')
    
    # 1. 求解van der Pol方程
    # 2. 绘制时间演化图
    print("\nAnalyzing parameter effects...")
    for mu in [1.0, 2.0, 4.0]:
        # 数值求解
        t, states = solve_ode(van_der_pol_ode, **params, mu=mu)
        
        # 时间演化图
        plot_time_evolution(t, states, f'Time Evolution (μ={mu})')
        
        # 相空间图
        plot_phase_space(states, f'Phase Space (μ={mu})')
        
    # TODO: 任务2 - 参数影响分析
    # 1. 尝试不同的mu值
    # 2. 比较和分析结果
    
    # TODO: 任务3 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 分析极限环特征
    
    # TODO: 任务4 - 能量分析
    print("\nAnalyzing energy evolution...")
    plt.figure(figsize=(10, 5))
    energies = np.array([calculate_energy(state) for state in states])
    plt.plot(t, energies)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution (μ=1)')
    plt.grid(True)
    plt.show()
    
    # 1. 计算和绘制能量随时间的变化
    # 2. 分析能量的耗散和补充
    
    # 极限环特征分析
    print("\nLimit cycle characteristics:")
    for mu in [1.0, 2.0, 4.0]:
        t, states = solve_ode(van_der_pol_ode, **params, mu=mu)
        amp, period = analyze_limit_cycle(states, t)
        print(f"μ={mu}: Amplitude={amp:.2f}, Period={period:.2f}")

if __name__ == "__main__":
    main()
