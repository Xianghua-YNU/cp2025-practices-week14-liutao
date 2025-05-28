import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, List

def forced_pendulum_ode(t: float, state: List[float], l: float, g: float, 
                       C: float, Omega: float) -> List[float]:
    """受驱单摆的常微分方程"""
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/l)*np.sin(theta) + C*np.cos(theta)*np.sin(Omega*t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l: float = 0.1, g: float = 9.81, C: float = 2, 
                  Omega: float = 5, t_span: Tuple[float, float] = (0, 100), 
                  y0: Optional[List[float]] = None, n_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """求解受迫单摆运动方程"""
    # 参数验证
    if y0 is None:
        y0 = [0.0, 0.0]
    if l <= 0:
        raise ValueError("摆长必须为正数")
    if t_span[1] <= t_span[0]:
        raise ValueError("时间范围无效")
    
    try:
        sol = solve_ivp(
            fun=forced_pendulum_ode,
            t_span=t_span,
            y0=y0,
            args=(l, g, C, Omega),
            t_eval=np.linspace(t_span[0], t_span[1], n_points),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        return sol.t, sol.y[0]
    except Exception as e:
        raise RuntimeError(f"求解失败: {str(e)}")

def find_resonance(l: float = 0.1, g: float = 9.81, C: float = 2, 
                  Omega_range: Optional[np.ndarray] = None, 
                  t_span: Tuple[float, float] = (0, 200), 
                  y0: Optional[List[float]] = None, 
                  transient_time: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """寻找共振频率"""
    if y0 is None:
        y0 = [0.0, 0.0]
    if Omega_range is None:
        Omega0 = np.sqrt(g/l)  # 自然频率
        Omega_range = np.linspace(0.5*Omega0, 2*Omega0, 50)
    
    amplitudes = []
    valid_omegas = []
    
    for Omega in Omega_range:
        try:
            t, theta = solve_pendulum(l=l, g=g, C=C, Omega=Omega, 
                                    t_span=t_span, y0=y0)
            
            # 计算稳态振幅(忽略暂态)
            mask = t >= transient_time
            if np.sum(mask) > 10:  # 确保有足够的数据点
                steady_theta = theta[mask]
                A = np.max(np.abs(steady_theta))  # 使用简单最大值法
                amplitudes.append(A)
                valid_omegas.append(Omega)
        except:
            continue  # 跳过失败的计算
    
    return np.array(valid_omegas), np.array(amplitudes)

def plot_time_series(t: np.ndarray, theta: np.ndarray, title: str = "") -> plt.Figure:
    """绘制时间序列图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, theta, 'b-', linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Angle (rad)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_amplitude_response(Omega_range: np.ndarray, amplitudes: np.ndarray, 
                          l: float, g: float) -> plt.Figure:
    """绘制振幅-频率响应曲线"""
    Omega0 = np.sqrt(g/l)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(Omega_range, amplitudes, 'b-', linewidth=1.5, label='Response')
    ax.axvline(Omega0, color='r', linestyle='--', linewidth=1.2,
              label=f'Natural freq: {Omega0:.2f} rad/s')
    
    if len(Omega_range) > 0:
        max_idx = np.argmax(amplitudes)
        ax.plot(Omega_range[max_idx], amplitudes[max_idx], 'ro', markersize=8,
               label=f'Resonance: {Omega_range[max_idx]:.2f} rad/s')
    
    ax.set_title('Amplitude-Frequency Response', fontsize=12)
    ax.set_xlabel('Driving Frequency (rad/s)', fontsize=10)
    ax.set_ylabel('Amplitude (rad)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def main():
    """主执行函数"""
    print("=== 受迫单摆共振现象研究 ===")
    
    # 任务1: 基本参数求解
    print("\n任务1: 基本参数求解 (Ω=5 rad/s)...")
    try:
        t1, theta1 = solve_pendulum()
        fig1 = plot_time_series(t1, theta1, "Pendulum Motion (Ω=5 rad/s)")
        plt.show()
    except Exception as e:
        print(f"任务1失败: {str(e)}")
    
    # 任务2: 共振分析
    print("\n任务2: 寻找共振频率...")
    try:
        Omega_range, amps = find_resonance()
        if len(Omega_range) == 0:
            raise RuntimeError("未找到有效共振频率")
        
        fig2 = plot_amplitude_response(Omega_range, amps, l=0.1, g=9.81)
        plt.show()
        
        max_idx = np.argmax(amps)
        Omega_res = Omega_range[max_idx]
        print(f"\n共振频率: {Omega_res:.2f} rad/s")
        print(f"最大振幅: {amps[max_idx]:.4f} rad")
        
        # 共振情况绘图
        print("\n绘制共振情况...")
        t_res, theta_res = solve_pendulum(Omega=Omega_res, t_span=(0, 200))
        fig3 = plot_time_series(t_res, theta_res, 
                              f"Resonance Case (Ω={Omega_res:.2f} rad/s)")
        plt.show()
        
    except Exception as e:
        print(f"任务2失败: {str(e)}")

if __name__ == '__main__':
    main()
