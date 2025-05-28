import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """受驱单摆的常微分方程"""
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/l)*np.sin(theta) + C*np.cos(theta)*np.sin(Omega*t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, 
                  t_start=0, t_end=100, y0=None, n_points=2000):
    """求解受迫单摆运动方程"""
    if y0 is None:
        y0 = [0.0, 0.0]
    
    # 基本参数检查
    if l <= 0:
        raise ValueError("摆长必须大于0")
    if t_end <= t_start:
        raise ValueError("结束时间必须大于开始时间")
    
    try:
        sol = solve_ivp(
            fun=lambda t, y: forced_pendulum_ode(t, y, l, g, C, Omega),
            t_span=[t_start, t_end],
            y0=y0,
            t_eval=np.linspace(t_start, t_end, n_points),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        return sol.t, sol.y[0]
    except Exception as e:
        raise RuntimeError(f"求解失败: {e}")

def find_resonance(l=0.1, g=9.81, C=2, 
                  Omega_min=None, Omega_max=None, num_points=50,
                  t_start=0, t_end=200, y0=None, 
                  transient_time=50.0):
    """寻找共振频率"""
    if y0 is None:
        y0 = [0.0, 0.0]
    
    # 自动确定频率范围
    Omega0 = np.sqrt(g/l)
    if Omega_min is None:
        Omega_min = 0.5 * Omega0
    if Omega_max is None:
        Omega_max = 2.0 * Omega0
    
    Omega_range = np.linspace(Omega_min, Omega_max, num_points)
    amplitudes = []
    valid_omegas = []
    
    for Omega in Omega_range:
        try:
            t, theta = solve_pendulum(l=l, g=g, C=C, Omega=Omega,
                                     t_start=t_start, t_end=t_end, y0=y0)
            
            # 计算稳态振幅(忽略暂态)
            mask = t >= transient_time
            if sum(mask) > 10:  # 确保有足够的数据点
                steady_theta = theta[mask]
                A = np.max(np.abs(steady_theta))
                amplitudes.append(A)
                valid_omegas.append(Omega)
        except:
            continue
    
    return np.array(valid_omegas), np.array(amplitudes)

def plot_results(t, theta, title=""):
    """绘制时间序列结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta, 'b-', linewidth=1.5)
    plt.title(title, fontsize=12)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Angle (rad)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_resonance_curve(Omega_range, amplitudes, l, g):
    """绘制共振曲线"""
    Omega0 = np.sqrt(g/l)
    plt.figure(figsize=(10, 5))
    
    plt.plot(Omega_range, amplitudes, 'b-', linewidth=1.5, label='Response')
    plt.axvline(Omega0, color='r', linestyle='--', linewidth=1.2,
               label=f'Natural freq: {Omega0:.2f} rad/s')
    
    if len(Omega_range) > 0:
        max_idx = np.argmax(amplitudes)
        plt.plot(Omega_range[max_idx], amplitudes[max_idx], 'ro', markersize=8,
                label=f'Resonance: {Omega_range[max_idx]:.2f} rad/s')
    
    plt.title('Amplitude-Frequency Response', fontsize=12)
    plt.xlabel('Driving Frequency (rad/s)', fontsize=10)
    plt.ylabel('Amplitude (rad)', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    print("=== 受迫单摆共振现象研究 ===")
    
    # 任务1: 基本参数求解
    print("\n任务1: 基本参数求解 (Ω=5 rad/s)...")
    try:
        t1, theta1 = solve_pendulum()
        plot_results(t1, theta1, "Pendulum Motion (Ω=5 rad/s)")
    except Exception as e:
        print(f"任务1失败: {e}")
        return
    
    # 任务2: 共振分析
    print("\n任务2: 寻找共振频率...")
    try:
        Omega_range, amps = find_resonance()
        if len(Omega_range) == 0:
            raise RuntimeError("未找到有效共振频率")
        
        plot_resonance_curve(Omega_range, amps, l=0.1, g=9.81)
        
        max_idx = np.argmax(amps)
        Omega_res = Omega_range[max_idx]
        print(f"\n共振频率: {Omega_res:.2f} rad/s")
        print(f"最大振幅: {amps[max_idx]:.4f} rad")
        
        # 共振情况绘图
        print("\n绘制共振情况...")
        t_res, theta_res = solve_pendulum(Omega=Omega_res, t_end=200)
        plot_results(t_res, theta_res, 
                    f"Resonance Case (Ω={Omega_res:.2f} rad/s)")
        
    except Exception as e:
        print(f"任务2失败: {e}")

if __name__ == '__main__':
    main()
