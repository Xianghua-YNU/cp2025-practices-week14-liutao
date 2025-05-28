import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """实现van der Pol振子的一阶微分方程组"""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """四阶龙格-库塔方法单步积分"""
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

def plot_time_evolution(t: np.ndarray, states: np.ndarray, mu: float) -> None:
    """绘制时间演化图"""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], 'b-', label='Displacement (x)')
    plt.plot(t, states[:, 1], 'r--', label='Velocity (v)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('State Variables', fontsize=12)
    plt.title(f'Time Evolution of van der Pol Oscillator (μ={mu})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, mu: float) -> None:
    """绘制相空间轨迹图"""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 'g-', linewidth=1)
    plt.xlabel('Displacement (x)', fontsize=12)
    plt.ylabel('Velocity (v)', fontsize=12)
    plt.title(f'Phase Space Trajectory (μ={mu})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_energy_evolution(t: np.ndarray, states: np.ndarray, mu: float, omega: float) -> None:
    """绘制能量随时间变化图"""
    # 计算能量
    energies = 0.5 * states[:, 1]**2 + 0.5 * omega**2 * states[:, 0]**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, energies, 'm-')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(f'Energy Evolution (μ={mu})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """分析极限环特征（内部定义时间参数t）"""
    # 定义时间参数（假设dt=0.01）
    dt = 0.01
    t = np.arange(0, len(states)) * dt
    
    x = states[:, 0]
    # 跳过前50%的瞬态过程
    skip = int(len(x) * 0.5)
    x_steady = x[skip:]
    
    # 振幅：取稳态部分的最大绝对值
    amplitude = np.max(np.abs(x_steady))
    
    # 周期：通过过零点计算
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    if len(zero_crossings) >= 2:
        periods = np.diff(t[zero_crossings])
        period = np.mean(periods) * 2  # 两次过零点为一个完整周期
    else:
        period = np.nan
    
    return amplitude, period

def main():
    # 基本参数设置（必须与analyze_limit_cycle中的dt一致）
    params = {
        'omega': 1.0,
        't_span': (0, 20),  # 延长模拟时间以获得稳定解
        'dt': 0.01,
        'initial_state': np.array([1.0, 0.0])
    }
    
    # 任务1：基本实现 (μ=1)
    print("="*20)
    print("Running basic simulation (μ=1.0)...")
    t, states = solve_ode(van_der_pol_ode, **params, mu=1.0)
    
    # 绘制时间演化图
    plot_time_evolution(t, states, mu=1.0)
    
    # 绘制相空间图
    plot_phase_space(states, mu=1.0)
    
    # 绘制能量图
    plot_energy_evolution(t, states, mu=1.0, omega=params['omega'])
    
    # 分析极限环
    amp, period = analyze_limit_cycle(states)
    print(f"μ=1.0: Amplitude={amp:.3f}, Period={period:.3f}s")
    
    # 任务2：参数影响分析
    mu_values = [2.0, 4.0]
    for mu in mu_values:
        print("\n" + "="*20)
        print(f"Analyzing parameter effect (μ={mu})...")
        t, states = solve_ode(van_der_pol_ode, **params, mu=mu)
        
        # 绘制时间演化图
        plot_time_evolution(t, states, mu=mu)
        
        # 绘制相空间图
        plot_phase_space(states, mu=mu)
        
        # 绘制能量图
        plot_energy_evolution(t, states, mu=mu, omega=params['omega'])
        
        # 分析极限环
        amp, period = analyze_limit_cycle(states)
        print(f"μ={mu}: Amplitude={amp:.3f}, Period={period:.3f}s")
    
    # 任务3：初始条件影响分析
    print("\n" + "="*20)
    print("Analyzing initial conditions...")
    initial_conditions = [
        np.array([0.5, 0.0]),
        np.array([2.0, 0.0]),
        np.array([0.0, 1.0])
    ]
    
    plt.figure(figsize=(8, 8))
    for i, ic in enumerate(initial_conditions):
        params['initial_state'] = ic
        t, states = solve_ode(van_der_pol_ode, **params, mu=1.0)
        plt.plot(states[:, 0], states[:, 1], label=f'Initial: ({ic[0]}, {ic[1]})')
    
    plt.xlabel('Displacement (x)', fontsize=12)
    plt.ylabel('Velocity (v)', fontsize=12)
    plt.title('Phase Space Trajectory for Different Initial Conditions (μ=1.0)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
