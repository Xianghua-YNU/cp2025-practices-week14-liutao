import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ========================
# 核心函数实现
# ========================

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """受迫单摆运动方程"""
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/l) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0, 100), y0=(0, 0)):
    """数值求解单摆运动"""
    try:
        sol = solve_ivp(
            fun=pendulum_ode,
            t_span=t_span,
            y0=y0,
            args=(l, g, C, Omega),
            t_eval=np.linspace(t_span[0], t_span[1], 3000),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        return sol.t, sol.y[0]
    except Exception as e:
        print(f"求解失败: {str(e)}")
        return None, None

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_total=200):
    """寻找共振频率"""
    # 自动生成频率范围
    if Omega_range is None:
        Omega0 = np.sqrt(g/l)  # 理论自然频率
        Omega_range = np.linspace(0.4*Omega0, 1.6*Omega0, 50)
    
    amplitudes = []
    valid_omegas = []
    
    for Omega in Omega_range:
        t, theta = solve_pendulum(
            l=l, g=g, C=C, Omega=Omega,
            t_span=(0, t_total),
            y0=(0, 0)
        )
        
        if t is None:
            continue  # 跳过失败情况
        
        # 稳态分析：取后1/3数据
        steady_start = int(2*len(t)/3)
        steady_theta = theta[steady_start:]
        
        if len(steady_theta) > 0:
            A = np.max(np.abs(steady_theta))
            amplitudes.append(A)
            valid_omegas.append(Omega)
    
    return np.array(valid_omegas), np.array(amplitudes)

# ========================
# 可视化函数
# ========================

def plot_angle_time(t, theta, title=""):
    """绘制角度-时间图"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta, color='#1f77b4', linewidth=1.2)
    plt.title(f"{title}\n(时间范围: {t[0]:.1f}-{t[-1]:.1f}s", fontsize=12)
    plt.xlabel('时间 (秒)', fontsize=10)
    plt.ylabel('摆角 (弧度)', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_amplitude_response(Omega_values, amplitudes, l, g):
    """绘制振幅-频率响应曲线"""
    plt.figure(figsize=(10, 5))
    
    # 理论自然频率
    Omega0 = np.sqrt(g/l)
    
    # 绘制响应曲线
    plt.plot(Omega_values, amplitudes, 
            marker='o', markersize=4,
            linestyle='-', linewidth=1.5,
            color='#2ca02c', 
            label='数值解')
    
    # 标记理论值和共振点
    plt.axvline(Omega0, color='#d62728', linestyle='--',
              label=f'理论自然频率: {Omega0:.2f} rad/s')
    
    if len(amplitudes) > 0:
        max_idx = np.argmax(amplitudes)
        plt.scatter(Omega_values[max_idx], amplitudes[max_idx],
                   color='#d62728', s=100, zorder=5,
                   label=f'实测共振频率: {Omega_values[max_idx]:.2f} rad/s')
    
    plt.title('振幅-频率响应曲线', fontsize=12)
    plt.xlabel('驱动力频率 (rad/s)', fontsize=10)
    plt.ylabel('稳态振幅 (rad)', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========================
# 主程序
# ========================

def main():
    print("="*40)
    print("受迫单摆共振现象研究")
    print("="*40)
    
    # 任务1: 基本参数求解
    print("\n>> 任务1: 基本参数求解 (Ω=5 rad/s)")
    t, theta = solve_pendulum()
    if t is not None:
        plot_angle_time(t, theta, "受迫单摆运动 (Ω=5 rad/s)")
    
    # 任务2: 共振分析
    print("\n>> 任务2: 共振频率检测")
    Omega_values, amplitudes = find_resonance()
    
    if len(Omega_values) == 0:
        print("错误: 未找到有效数据")
        return
    
    # 绘制共振曲线
    plot_amplitude_response(Omega_values, amplitudes, l=0.1, g=9.81)
    
    # 找到共振频率
    max_idx = np.argmax(amplitudes)
    Omega_res = Omega_values[max_idx]
    print(f"\n共振频率: {Omega_res:.2f} rad/s")
    print(f"最大振幅: {amplitudes[max_idx]:.2f} rad")
    
    # 绘制共振情况
    print("\n>> 绘制共振情况...")
    t_res, theta_res = solve_pendulum(
        Omega=Omega_res,
        t_span=(0, 200),
        y0=(0, 0)
    )
    if t_res is not None:
        plot_angle_time(t_res, theta_res, 
                       f"共振状态 (Ω={Omega_res:.2f} rad/s)")

if __name__ == "__main__":
    main()
