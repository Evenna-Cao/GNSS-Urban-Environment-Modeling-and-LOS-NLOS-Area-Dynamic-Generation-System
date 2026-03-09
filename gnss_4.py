import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import os
import georinex as gr
import xarray as xr

# ==========================================
# 0. 核心配置 (在这里修改推演参数)
# ==========================================
SIM_STEPS = 5  # 推演步数 (例如：推演 5 个时刻)
STEP_MINUTES = 15  # 推演步长 (例如：每隔 15 分钟)
OUTPUT_ROOT = "simulation_results_5_15"  # 结果根目录

# ==========================================
# 第一部分：GPS 轨道解算与坐标转换 (核心数学库)
# ==========================================

# WGS84 物理常数
MU = 3.986005e14  # 地球引力常数 (m^3/s^2)
OMEGA_E = 7.2921151467e-5  # 地球自转角速度 (rad/s)


def gps_orbit_calculation(eph, t_input):
    """
    基于广播星历参数计算卫星 ECEF 坐标
    :param t_input: 目标时刻 (GPS周内秒)
    """
    # 1. 计算时间差 (t - toe)
    t_k = t_input - eph['toe']

    # 处理跨周问题 (如果时间差超过半周，说明跨周了)
    if t_k > 302400: t_k -= 604800
    if t_k < -302400: t_k += 604800

    # 2. 计算平均运动
    A = eph['sqrt_a'] ** 2
    n0 = np.sqrt(MU / A ** 3)
    n = n0 + eph['delta_n']

    # 3. 平均近点角 M_k (随时间 t_k 线性变化，这就是运动学的核心)
    M_k = eph['m0'] + n * t_k

    # 4. 偏近点角 (开普勒方程迭代求解)
    E_k = M_k
    for _ in range(10):
        E_new = M_k + eph['e'] * np.sin(E_k)
        if abs(E_new - E_k) < 1e-12:
            break
        E_k = E_new

    # 5. 真近点角
    sin_v = (np.sqrt(1 - eph['e'] ** 2) * np.sin(E_k)) / (1 - eph['e'] * np.cos(E_k))
    cos_v = (np.cos(E_k) - eph['e']) / (1 - eph['e'] * np.cos(E_k))
    v_k = np.arctan2(sin_v, cos_v)

    # 6. 升交点角距
    phi_k = v_k + eph['omega']

    # 7. 摄动校正
    sin_2phi = np.sin(2 * phi_k)
    cos_2phi = np.cos(2 * phi_k)

    du_k = eph['cus'] * sin_2phi + eph['cuc'] * cos_2phi
    dr_k = eph['crs'] * sin_2phi + eph['crc'] * cos_2phi
    di_k = eph['cis'] * sin_2phi + eph['cic'] * cos_2phi

    u_k = phi_k + du_k
    r_k = A * (1 - eph['e'] * np.cos(E_k)) + dr_k
    i_k = eph['i0'] + eph['idot'] * t_k + di_k

    # 8. 轨道平面坐标
    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    # 9. 升交点经度 (校正后的，包含地球自转修正)
    omega_k = eph['omega0'] + (eph['omega_dot'] - OMEGA_E) * t_k - OMEGA_E * eph['toe']

    # 10. ECEF 坐标转换
    x = x_k_prime * np.cos(omega_k) - y_k_prime * np.cos(i_k) * np.sin(omega_k)
    y = x_k_prime * np.sin(omega_k) + y_k_prime * np.cos(i_k) * np.cos(omega_k)
    z = y_k_prime * np.sin(i_k)

    return np.array([x, y, z])


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    """ ECEF 转 局部 ENU """
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a ** 2 - b ** 2) / a ** 2

    phi = np.radians(lat0)
    lam = np.radians(lon0)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    N = a / np.sqrt(1 - e2 * sin_phi ** 2)
    X0 = (N + h0) * cos_phi * cos_lam
    Y0 = (N + h0) * cos_phi * sin_lam
    Z0 = (N * (1 - e2) + h0) * sin_phi

    dx, dy, dz = x - X0, y - Y0, z - Z0

    t = cos_lam * dx + sin_lam * dy
    u = -sin_lam * dx + cos_lam * dy
    w = dz

    east = u
    north = -sin_phi * t + cos_phi * w
    up = cos_phi * t + sin_phi * w

    return np.array([east, north, up])


# ==========================================
# 第二部分：仿真模型类
# ==========================================

class Building:
    def __init__(self, name, x, y, w, d, h):
        self.name = name
        self.center = np.array([x, y, h / 2.0])
        self.dims = np.array([w, d, h])
        self.mesh = trimesh.creation.box(extents=self.dims)
        transform = np.eye(4)
        transform[:3, 3] = self.center
        self.mesh.apply_transform(transform)


class CityScene:
    def __init__(self):
        self.buildings = []
        self.scene_mesh = None
        self.intersector = None

    def add_building(self, b: Building):
        self.buildings.append(b)

    def build(self):
        if not self.buildings: return
        self.scene_mesh = trimesh.util.concatenate([b.mesh for b in self.buildings])
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector
            self.intersector = RayMeshIntersector(self.scene_mesh)
            print("成功加载内置射线引擎。")
        except Exception as e:
            self.intersector = trimesh.ray.list_engines(self.scene_mesh)[0]


class Satellite:
    def __init__(self, prn, position_enu):
        self.prn = prn
        self.position = position_enu


class GNSS_Simulator:
    def __init__(self, city, x_lim, y_lim, step):
        self.city = city
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(x_lim[0], x_lim[1] + step, step),
            np.arange(y_lim[0], y_lim[1] + step, step)
        )
        self.receivers = np.column_stack((self.grid_x.flatten(), self.grid_y.flatten(), np.full(self.grid_x.size, 1.5)))

    def check_visibility(self, sat_pos):
        if self.city.intersector is None: return np.ones_like(self.grid_x)
        vectors = sat_pos - self.receivers
        distances = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        is_occluded = self.city.intersector.intersects_any(ray_origins=self.receivers,
                                                           ray_directions=vectors / distances)
        return np.where(is_occluded, 0, 1).reshape(self.grid_x.shape)


# ==========================================
# 第三部分：数据加载与结果输出 (升级版)
# ==========================================

def load_ephemeris_with_library(file_path):
    try:
        return gr.load(file_path)
    except Exception as e:
        print(f"解析失败: {e}")
        return None


def get_eph_dict_from_dataset(nav_ds, prn):
    # 提取并转换参数
    sat_data = nav_ds.sel(sv=prn).dropna(dim='time', how='all').isel(time=-1)
    return {
        'PRN': str(prn),
        'toe': float(sat_data['Toe']), 'sqrt_a': float(sat_data['sqrtA']), 'e': float(sat_data['Eccentricity']),
        'm0': float(sat_data['M0']), 'delta_n': float(sat_data['DeltaN']), 'omega': float(sat_data['omega']),
        'omega0': float(sat_data['Omega0']), 'omega_dot': float(sat_data['OmegaDot']), 'i0': float(sat_data['Io']),
        'idot': float(sat_data['IDOT']), 'cus': float(sat_data['Cus']), 'cuc': float(sat_data['Cuc']),
        'crs': float(sat_data['Crs']), 'crc': float(sat_data['Crc']), 'cis': float(sat_data['Cis']),
        'cic': float(sat_data['Cic'])
    }


def plot_and_save_results(output_dir, sim, sat_obj, vis_matrix, time_info):
    """
    生成可视化结果
    :param time_info: 字符串，用于显示当前推演时间
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. 保存矩阵
    mat_filename = os.path.join(output_dir, f"{sat_obj.prn}_matrix.txt")
    np.savetxt(mat_filename, vis_matrix, fmt='%d', delimiter=',')

    # 2. 绘图
    sx, sy, sz = sat_obj.position
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Sat: {sat_obj.prn} | Time: {time_info}", fontsize=16)

    # 子图1: Top View
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("1. Top View (X-Y)")
    for b in sim.city.buildings:
        ax1.add_patch(
            Rectangle((b.center[0] - b.dims[0] / 2, b.center[1] - b.dims[1] / 2), b.dims[0], b.dims[1], color='gray',
                      alpha=0.5))
    ax1.scatter(sx, sy, c='red', marker='*', s=200)
    ax1.set_xlim(sim.x_lim);
    ax1.set_ylim(sim.y_lim);
    ax1.grid(True)

    # 子图2: Side View (X-Z)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("2. Side View (X-Z)")
    for b in sim.city.buildings:
        ax2.add_patch(Rectangle((b.center[0] - b.dims[0] / 2, 0), b.dims[0], b.dims[2], color='gray', alpha=0.5))
    ax2.arrow(sim.x_lim[1] / 2, 0, sx / (abs(sz) + 1) * 50, 50, head_width=5, color='red')
    ax2.text(sim.x_lim[1] / 2, 50, " To Sat", color='red')
    ax2.set_xlim(sim.x_lim);
    ax2.set_ylim(0, 150);
    ax2.grid(True)

    # 子图3: Side View (Y-Z)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("3. Side View (Y-Z)")
    for b in sim.city.buildings:
        ax3.add_patch(Rectangle((b.center[1] - b.dims[1] / 2, 0), b.dims[1], b.dims[2], color='gray', alpha=0.5))
    ax3.arrow(sim.y_lim[1] / 2, 0, sy / (abs(sz) + 1) * 50, 50, head_width=5, color='red')
    ax3.text(sim.y_lim[1] / 2, 50, " To Sat", color='red')
    ax3.set_xlim(sim.y_lim);
    ax3.set_ylim(0, 150);
    ax3.grid(True)

    # 子图4: 3D View
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_title("4. 3D Scene")
    for b in sim.city.buildings:
        ax4.bar3d(b.center[0] - b.dims[0] / 2, b.center[1] - b.dims[1] / 2, 0, b.dims[0], b.dims[1], b.dims[2],
                  color='cyan', alpha=0.6, edgecolor='k')
    norm_vec = sat_obj.position / np.linalg.norm(sat_obj.position) * 150
    center = [(sim.x_lim[1] - sim.x_lim[0]) / 2, (sim.y_lim[1] - sim.y_lim[0]) / 2, 0]
    ax4.plot([center[0], center[0] + norm_vec[0]], [center[1], center[1] + norm_vec[1]],
             [center[2], center[2] + norm_vec[2]], 'r--', lw=2)
    ax4.set_zlim(0, 150)

    # 子图5: Heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("5. Visibility Heatmap")
    im = ax5.imshow(vis_matrix, origin='lower', extent=[*sim.x_lim, *sim.y_lim], cmap='bwr', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax5, ticks=[0, 1], label='0: NLOS, 1: LOS')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sat_obj.prn}_analysis.png"))
    plt.close()


# ==========================================
# 第四部分：主程序执行 (时间序列推演版)
# ==========================================
if __name__ == "__main__":
    # 0. 清理环境
    import os

    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

    # 1. 搭建城市
    city = CityScene()
    buildings_data = [
        {'name': 'B1', 'x': 30, 'y': 30, 'w': 20, 'd': 20, 'h': 40},
        {'name': 'B2', 'x': 70, 'y': 60, 'w': 15, 'd': 40, 'h': 60}
    ]
    for b in buildings_data: city.add_building(Building(**b))
    city.build()
    sim = GNSS_Simulator(city, x_lim=(0, 100), y_lim=(0, 100), step=1.0)

    # 2. 读取星历
    nav_file = "hkcl356u.25n"
    nav_ds = load_ephemeris_with_library(nav_file)

    if nav_ds is not None:
        all_svs = nav_ds.sv.values
        print(f"--- 开始推演: 共 {SIM_STEPS} 步，每步 {STEP_MINUTES} 分钟 ---")

        # 3. 遍历每一颗卫星
        for sv in all_svs:
            if not str(sv).startswith('G'): continue  # 仅处理 GPS

            try:
                eph = get_eph_dict_from_dataset(nav_ds, sv)
                prn = eph['PRN']

                # 获取该卫星的参考时间 Toe (作为推演的起始时间 t0)
                t_start = eph['toe']
                print(f"\n[Sat {prn}] 基准时间 Toe: {t_start}")

                # 4. 时间推演循环 (Time Loop)
                for step in range(SIM_STEPS):
                    # 计算当前时刻: t = t0 + step * minutes * 60
                    dt_seconds = step * STEP_MINUTES * 60
                    current_time = t_start + dt_seconds

                    # 4.1 轨道计算 (传入变化的 current_time)
                    # 这里的 gps_orbit_calculation 会利用开普勒公式自动推演新位置
                    sat_ecef = gps_orbit_calculation(eph, current_time)

                    # 4.2 坐标转换
                    sat_enu = ecef_to_enu(sat_ecef[0], sat_ecef[1], sat_ecef[2], 22.3, 114.2, 50.0)

                    # 4.3 准备输出文件夹: simulation_results/Step_0/
                    # 我们按步骤创建文件夹，把同一时刻的所有卫星放在一起，或者按卫星分
                    # 根据你的需求：“输出每一时刻（文件夹）”
                    step_folder_name = f"Step_{step:02d}_Tplus_{int(dt_seconds / 60)}min"
                    current_output_dir = os.path.join(OUTPUT_ROOT, step_folder_name)

                    if sat_enu[2] > 0:  # 仅处理地平线以上
                        sat_obj = Satellite(prn, sat_enu)

                        # 4.4 计算遮挡
                        vis_matrix = sim.check_visibility(sat_obj.position)

                        # 4.5 绘图并保存
                        time_label = f"Toe + {int(dt_seconds / 60)} min"
                        plot_and_save_results(current_output_dir, sim, sat_obj, vis_matrix, time_label)

                        # 简单的进度条
                        print(f"  -> Step {step}: {time_label} | Az/El计算完成")
                    else:
                        print(f"  -> Step {step}: 地平线下 (不可见)")

            except Exception as e:
                print(f"处理 {sv} 异常: {e}")

    print(f"\n全部推演完成。请查看 {os.path.abspath(OUTPUT_ROOT)}")