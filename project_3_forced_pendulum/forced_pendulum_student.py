import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, List

def forced_pendulum_ode(t: float, state: List[float], l: float, g: float, 
                       C: float, Omega: float) -> List[float]:
    """
    受驱单摆的常微分方程
    
    参数:
        t: 当前时间
        state: [theta(角度), omega(角速度)]
        l: 摆长(m)
        g: 重力加速度(m/s²)
        C: 驱动力强度(s⁻²)
        Omega: 驱动力角频率(rad/s)
    
    返回:
        [dtheta/dt, domega/dt]
    """
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/l) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l: float = 0.1, g: float = 9.81, C: float = 2, 
                  Omega: float = 5, t_span: Tuple[float, float] = (0, 100), 
                  y0: List[float] = [0, 0], n_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解受迫单摆运动方程
    
    参数:
        l: 摆长(m)，0.1
        g: 重力加速度(m/s²)，9.81
        C: 驱动力强度(s⁻²)，2
        Omega: 驱动力角频率(rad/s)，5
        t_span: 时间范围(s)，(0,100)
        y0: 初始条件[theta0, omega0]，[0,0]
        n_points: 时间点数量，2000
    
    返回:
        (时间数组, 角度数组)
    """
    # 参数验证
    assert l > 0, "摆长必须为正数"
    assert g > 0, "重力加速度必须为正数"
    assert t_span[1] > t_span[0], "时间范围无效"
    
    sol = solve_ivp(
        fun=forced_pendulum_ode,
        t_span=t_span,
        y0=y0,
        args=(l, g, C, Omega),
        t_eval=np.linspace(t_span[0], t_span[1], n_points),
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    return sol.t, sol.y[0]

def find_resonance(l: float = 0.1, g: float = 9.81, C: float = 2, 
                  Omega_range: Optional[np.ndarray] = None, 
                  t_span: Tuple[float, float] = (0, 200), 
                  y0: List[float] = [0, 0], transient_time: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    寻找共振频率
    
    参数:
        l: 摆长(m)，0.1
        g: 重力加速度(m/s²)，9.81
        C: 驱动力强度(s⁻²)，2
        Omega_range: 扫描的频率范围(rad/s)，None(自动生成)
        t_span: 时间范围(s)，(0,200)
        y0: 初始条件，[0,0]
        transient_time: 暂态时间(s)，50.0
    
    返回:
        (频率数组, 振幅数组)
    """
    if Omega_range is None:
        Omega0 = np.sqrt(g/l)  # 自然频率
        Omega_range = np.linspace(0.5*Omega0, 2*Omega0, 50)
    
    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l=l, g=g, C=C, Omega=Omega, t_span=t_span, y0=y0)
        
        # 计算稳态振幅(忽略暂态)
        mask = t >= transient_time
        if np.sum(mask) > 0:
            steady_theta = theta[mask]
            # 使用峰值检测代替简单的最大值
            peaks, _ = find_peaks(np.abs(steady_theta))
            if len(peaks) > 0:
                A = np.mean(np.abs(steady_theta[peaks]))  # 取峰值平均值
            else:
                A = np.max(np.abs(steady_theta))
        else:
            A = 0.0
            
        amplitudes.append(A)
    
    return np.array(Omega_range), np.array(amplitudes)

def find_peaks(x: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, dict]:
    """
    简单的峰值检测函数
    
    参数:
        x: 输入信号
        threshold: 峰值最小高度(相对最大值比例)
    
    返回:
        (峰值索引, 峰值属性)
    """
    from scipy.signal import find_peaks
    min_height = threshold * np.max(x)
    peaks, properties = find_peaks(x, height=min_height)
    return peaks, properties

def plot_time_series(t: np.ndarray, theta: np.ndarray, title: str = "") -> None:
    """绘制时间序列图"""
    plt.figure(figsize=(12, 6))
    plt.plot(t, theta, label=r'$\theta(t)$')
    plt.title(title + "\n" + r"$\theta(t)$ vs $t$", fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (rad)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_amplitude_response(Omega_range: np.ndarray, amplitudes: np.ndarray, 
                          l: float, g: float) -> None:
    """绘制振幅-频率响应曲线"""
    Omega0 = np.sqrt(g/l)
    plt.figure(figsize=(12, 6))
    plt.plot(Omega_range, amplitudes, 'b-', linewidth=2, label='Response')
    plt.axvline(Omega0, color='r', linestyle='--', 
               label=rf'Natural frequency $\Omega_0 = {Omega0:.2f}$ rad/s')
    
    # 标记共振频率
    max_idx = np.argmax(amplitudes)
    plt.scatter(Omega_range[max_idx], amplitudes[max_idx], color='red', s=100,
               label=f'Resonance: {Omega_range[max_idx]:.2f} rad/s')
    
    plt.title('Amplitude-Frequency Response', fontsize=14)
    plt.xlabel('Driving Frequency $\Omega$ (rad/s)', fontsize=12)
    plt.ylabel('Steady State Amplitude (rad)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    # 任务1: 特定参数下的数值解与可视化
    print("Running Task 1...")
    t1, theta1 = solve_pendulum()
    plot_time_series(t1, theta1, "Task 1: Forced Pendulum (Ω=5 rad/s)")
    
    # 任务2: 探究共振现象
    print("\nRunning Task 2...")
    Omega_range, amplitudes = find_resonance()
    plot_amplitude_response(Omega_range, amplitudes, l=0.1, g=9.81)
    
    # 找到共振频率并绘制共振情况
    max_idx = np.argmax(amplitudes)
    Omega_res = Omega_range[max_idx]
    print(f"\nResonance frequency found at Ω = {Omega_res:.2f} rad/s")
    print(f"Maximum amplitude = {amplitudes[max_idx]:.2f} rad")
    
    t_res, theta_res = solve_pendulum(Omega=Omega_res, t_span=(0, 200))
    plot_time_series(t_res, theta_res, 
                   f"Resonance Case (Ω={Omega_res:.2f} rad/s)")

if __name__ == '__main__':
    main()
