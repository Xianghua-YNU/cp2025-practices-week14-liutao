import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Optional

def van_der_pol_ode(t, state, mu=1.0, omega=1.0):
    """van der Pol振子的一阶微分方程组。"""
    x, v = state
    return np.array([v, mu*(1-x**2)*v - omega**2*x])

def solve_ode(ode_func, initial_state, t_span, dt, **kwargs):
    """使用solve_ivp求解常微分方程组"""
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(ode_func, t_span, initial_state, 
                   t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """Plot the time evolution of states."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """Plot the phase space trajectory."""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_limit_cycle(
    t: np.ndarray,
    states: np.ndarray,
    skip_ratio: float = 0.3,
    min_peaks: int = 3,
    debug: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """
    分析极限环特征（振幅和周期），使用明确的时间参数t
    
    参数:
        t: 时间序列数组
        states: 状态数组，形状为(N, 2)
        skip_ratio: 跳过初始瞬态的比例（默认跳过前30%）
        min_peaks: 需要的最小有效波峰数（至少2个峰才能计算周期）
        debug: 是否显示波峰检测验证图
    
    返回:
        (振幅, 周期) 或 (None, None)（当分析失败时）
    """
    # 参数校验
    if len(t) != len(states):
        raise ValueError("时间数组t和状态数组states的长度必须一致")
    if states.shape[1] != 2:
        raise ValueError("状态数组应为(N, 2)形状")
    
    # 跳过初始瞬态
    skip = int(len(states) * skip_ratio)
    x = states[skip:, 0]
    t_steady = t[skip:]
    
    # 波峰检测
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    
    # 有效性检查
    if len(peaks) < min_peaks:
        print(f"警告：只找到{len(peaks)}个波峰，至少需要{min_peaks}个")
        return (None, None)
    
    # 使用实际时间计算特征
    peak_indices = np.array(peaks) + skip  # 转换为全局索引
    peak_times = t[peak_indices]
    
    # 振幅（取稳态峰值的平均值）
    amplitude = np.mean(states[peak_indices, 0])
    
    # 周期（取最后几个周期的平均值）
    periods = np.diff(peak_times)
    if len(periods) >= 2:
        period = np.mean(periods[-3:])  # 取最后3个周期
    else:
        period = periods[0]
    
    # 调试可视化
    if debug:
        plt.figure(figsize=(12, 4))
        plt.plot(t_steady, x, label="Steady State")
        plt.plot(peak_times, states[peak_indices, 0], 'ro', markersize=4)
        plt.title(f"Peak Detection (Found {len(peaks)} peaks)")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.show()
    
    return (abs(amplitude), period)

def main():
    # 基本参数
    mu = 1.0
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基本实现
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'van der Pol振子时间演化 (μ={mu})')
    
    # 任务2 - 参数影响分析
    mu_values = [1.0, 2.0, 4.0]
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'相空间轨迹 (μ={mu})')
        amplitude, period = analyze_limit_cycle(t, states, debug=True)
        if amplitude and period:
            print(f"μ = {mu}: 振幅 = {amplitude:.3f}, 周期 = {period:.3f}秒")
    
    # 任务3 - 初始条件影响
    initial_conditions = [
        [2.0, 0.0],
        [0.5, 0.0],
        [1.0, 2.0]
    ]
    for ic in initial_conditions:
        t, states = solve_ode(van_der_pol_ode, ic, t_span, dt, mu=2.0)
        plot_phase_space(states, f'初始条件 {ic} 相空间')

if __name__ == "__main__":
    main()
