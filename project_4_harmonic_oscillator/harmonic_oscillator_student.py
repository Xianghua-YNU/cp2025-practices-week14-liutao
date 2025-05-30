import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x**3
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    k1 = dt * ode_func(state, t, **kwargs)
    k2 = dt * ode_func(state + 0.5*k1, t + 0.5*dt, **kwargs)
    k3 = dt * ode_func(state + 0.5*k2, t + 0.5*dt, **kwargs)
    k4 = dt * ode_func(state + k3, t + dt, **kwargs)
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
    return new_state

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, num_steps)
    state_array = np.zeros((num_steps, len(initial_state)))
    state_array[0] = initial_state
    
    for i in range(1, num_steps):
        state_array[i] = rk4_step(ode_func, state_array[i-1], t_array[i-1], dt, **kwargs)
    
    return t_array, state_array

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='位移 (x)')
    plt.plot(t, states[:, 1], label='速度 (v)')
    plt.xlabel('时间 (t)')
    plt.ylabel('状态')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('位移 (x)')
    plt.ylabel('速度 (v)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    # 寻找位移过零点（由正变负）
    zero_crossings = []
    for i in range(1, len(states)):
        if states[i-1, 0] >= 0 and states[i, 0] < 0:
            # 线性插值得到精确的过零点时间
            t0 = t[i-1]
            t1 = t[i]
            x0 = states[i-1, 0]
            x1 = states[i, 0]
            t_cross = t0 - x0 * (t1 - t0) / (x1 - x0)
            zero_crossings.append(t_cross)
    
    # 计算相邻过零点的时间差（半周期）
    if len(zero_crossings) < 2:
        return None
    
    half_periods = np.diff(zero_crossings)
    # 取平均值并乘以2得到完整周期
    period = 2 * np.mean(half_periods)
    return period

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # 任务1: 简谐振子的数值求解 (初始条件 x(0)=1, v(0)=0)
    initial_state = np.array([1.0, 0.0])
    t_harmonic, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t_harmonic, states_harmonic, "简谐振子: 时间演化 (x(0)=1, v(0)=0)")
    
    # 任务2: 振幅对周期的影响分析 (简谐振子)
    amplitudes = [1.0, 2.0, 3.0]
    harmonic_periods = []
    
    for amp in amplitudes:
        initial_state = np.array([amp, 0.0])
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        harmonic_periods.append(period)
        print(f"简谐振子: 初始振幅={amp}, 周期={period:.4f} (理论值={2*np.pi/omega:.4f})")
    
    # 任务3: 非谐振子的数值分析
    anharmonic_periods = []
    
    for amp in amplitudes:
        initial_state = np.array([amp, 0.0])
        t_anharmonic, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t_anharmonic, states_anharmonic)
        anharmonic_periods.append(period)
        print(f"非谐振子: 初始振幅={amp}, 周期={period:.4f}")
        
        if amp == 1.0 or amp == 2.0:
            plot_time_evolution(t_anharmonic, states_anharmonic, f"非谐振子: 时间演化 (x(0)={amp}, v(0)=0)")
    
    # 任务4: 相空间分析
    # 简谐振子 (振幅=1)
    initial_state = np.array([1.0, 0.0])
    _, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_harmonic, "简谐振子: 相空间轨迹 (x(0)=1, v(0)=0)")
    
    # 非谐振子 (振幅=1)
    _, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_anharmonic, "非谐振子: 相空间轨迹 (x(0)=1, v(0)=0)")
    
    # 绘制振幅对周期的影响
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes, harmonic_periods, 'o-', label='简谐振子')
    plt.plot(amplitudes, anharmonic_periods, 's-', label='非谐振子')
    plt.axhline(y=2*np.pi/omega, color='r', linestyle='--', label='理论周期 (简谐)')
    plt.xlabel('初始振幅')
    plt.ylabel('振动周期')
    plt.title('振幅对振动周期的影响')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
