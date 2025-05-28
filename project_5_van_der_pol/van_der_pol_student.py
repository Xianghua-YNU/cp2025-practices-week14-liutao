import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(t: float, state: np.ndarray, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """van der Pol振子的一阶微分方程组"""
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

def plot_time_evolution(t: np.ndarray, states: np.ndarray, mu: float) -> None:
    """绘制时间演化图"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, states[:, 0], color='#1f77b4', label='Position x(t)')
    plt.plot(t, states[:, 1], color='#ff7f0e', linestyle='--', label='Velocity v(t)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('State Variables', fontsize=12)
    plt.title(f'Time Evolution (μ={mu})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, mu: float) -> None:
    """绘制相空间轨迹图"""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 
             color='#2ca02c', 
             linewidth=1,
             alpha=0.8)
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel('Velocity v', fontsize=12)
    plt.title(f'Phase Space (μ={mu})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def analyze_limit_cycle(t: np.ndarray, states: np.ndarray) -> Tuple[float, float]:
    """分析极限环特征（增强版峰值检测）"""
    # 跳过前50%的瞬态过程
    skip = int(len(states) * 0.5)
    x = states[skip:, 0]
    t_steady = t[skip:]
    
    # 改进的峰值检测算法
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > 0.5*np.max(x):
            peaks.append((t_steady[i], x[i]))
    
    # 计算振幅和周期
    if len(peaks) >= 3:
        peak_times = [p[0] for p in peaks]
        periods = np.diff(peak_times)
        amplitude = np.mean([p[1] for p in peaks])
        period = np.mean(periods)
    else:
        amplitude = period = np.nan
    
    return amplitude, period

def main():
    # 统一参数设置
    params = {
        'omega': 1.0,
        't_span': (0, 50),  # 延长模拟时间以获得稳定解
        'dt': 0.01,
        'initial_state': np.array([1.0, 0.0])
    }
    
    # 参数分析
    mu_values = [1.0, 2.0, 4.0]
    
    # 独立绘制每个参数的图表
    for mu in mu_values:
        print(f"\n正在分析 μ={mu} 的情况...")
        
        # 数值求解
        t, states = solve_ode(van_der_pol_ode, **params, mu=mu)
        
        # 时间演化图
        plot_time_evolution(t, states, mu)
        
        # 相空间图
        plot_phase_space(states, mu)
        
        # 极限环分析
        amplitude, period = analyze_limit_cycle(t, states)
        print(f"μ={mu}: 稳态振幅 = {amplitude:.3f}, 振荡周期 = {period:.3f}s")
        
        # 能量分析
        energy = 0.5 * states[:,1]**2 + 0.5 * params['omega']**2 * states[:,0]**2
        plt.figure(figsize=(10, 5))
        plt.plot(t, energy, color='#d62728')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title(f'Energy Evolution (μ={mu})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
