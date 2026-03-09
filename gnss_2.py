import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import math
import os

# ==========================================
# 第一部分：GPS 轨道解算与坐标转换 (核心数学库)
# ==========================================

# WGS84 物理常数
MU = 3.986005e14  # 地球引力常数 (m^3/s^2)
OMEGA_E = 7.2921151467e-5  # 地球自转角速度 (rad/s)
PI = np.pi


def gps_orbit_calculation(eph, t_input):
    """
    基于广播星历参数计算卫星 ECEF 坐标
    参考: IS-GPS-200 Interface Specification
    :param eph: 包含星历参数的字典 (key需与下列变量名一致)
    :param t_input: 当前计算时刻 (周内秒)
    :return: (x, y, z) ECEF 坐标 (米)
    """
    # 1. 计算时间
    t_k = t_input - eph['toe']

    # 处理跨周问题 (简易处理)
    if t_k > 302400: t_k -= 604800
    if t_k < -302400: t_k += 604800

    # 2. 计算平均运动
    A = eph['sqrt_a'] ** 2
    n0 = np.sqrt(MU / A ** 3)
    n = n0 + eph['delta_n']

    # 3. 平均近点角
    M_k = eph['m0'] + n * t_k

    # 4. 偏近点角 (开普勒方程迭代求解 E_k = M_k + e * sin(E_k))
    E_k = M_k
    for _ in range(10):  # 10次迭代通常足够收敛
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

    # 9. 升交点经度 (校正后的)
    omega_k = eph['omega0'] + (eph['omega_dot'] - OMEGA_E) * t_k - OMEGA_E * eph['toe']

    # 10. ECEF 坐标转换
    x = x_k_prime * np.cos(omega_k) - y_k_prime * np.cos(i_k) * np.sin(omega_k)
    y = x_k_prime * np.sin(omega_k) + y_k_prime * np.cos(i_k) * np.cos(omega_k)
    z = y_k_prime * np.sin(i_k)

    return np.array([x, y, z])


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    """
    将 ECEF 坐标转换为 局部 ENU 坐标 (以城市中心为原点)
    :param lat0, lon0: 城市中心纬度、经度 (度)
    :param h0: 城市中心高度
    """
    # WGS84 椭球参数
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

    # 计算参考点的 ECEF 坐标 (X0, Y0, Z0)
    N = a / np.sqrt(1 - e2 * sin_phi ** 2)
    X0 = (N + h0) * cos_phi * cos_lam
    Y0 = (N + h0) * cos_phi * sin_lam
    Z0 = (N * (1 - e2) + h0) * sin_phi

    # 差值
    dx, dy, dz = x - X0, y - Y0, z - Z0

    # 旋转矩阵 (ECEF -> ENU)
    t = cos_lam * dx + sin_lam * dy
    u = -sin_lam * dx + cos_lam * dy
    w = dz

    east = u
    north = -sin_phi * t + cos_phi * w
    up = cos_phi * t + sin_phi * w

    return np.array([east, north, up])


# ==========================================
# 第二部分：仿真模型类 (Building & Simulator)
# ==========================================

class Building:
    def __init__(self, name, x, y, w, d, h):
        self.name = name
        self.center = np.array([x, y, h / 2.0])
        self.dims = np.array([w, d, h])  # width(x), depth(y), height(z)
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
        # # 使用 trimesh 的射线求交引擎
        # self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.scene_mesh)
        # --- 修复逻辑开始 ---
        try:
            # 优先尝试内置的 ray_triangle 引擎，它不依赖外部 embree.dll
            from trimesh.ray.ray_triangle import RayMeshIntersector
            self.intersector = RayMeshIntersector(self.scene_mesh)
            print("成功加载内置射线引擎。")
        except Exception as e:
            # 如果还报错，使用最通用的接口
            self.intersector = trimesh.ray.list_engines(self.scene_mesh)[0]
            print(f"转换引擎，当前使用: {type(self.intersector)}")
        # --- 修复逻辑结束 ---



class Satellite:
    def __init__(self, prn, position_enu):
        self.prn = prn
        self.position = position_enu  # [x, y, z] in ENU (meters)


class GNSS_Simulator:
    def __init__(self, city, x_lim, y_lim, step):
        self.city = city
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.step = step

        # 生成网格
        self.xs = np.arange(x_lim[0], x_lim[1] + step, step)
        self.ys = np.arange(y_lim[0], y_lim[1] + step, step)
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)

        # 接收机坐标 (Z=1.5m)
        self.flat_x = self.grid_x.flatten()
        self.flat_y = self.grid_y.flatten()
        self.z_h = np.full_like(self.flat_x, 1.5)
        self.receivers = np.column_stack((self.flat_x, self.flat_y, self.z_h))

    def check_visibility(self, sat_pos):
        """核心射线追踪"""
        if self.city.intersector is None:
            return np.ones_like(self.grid_x)  # 如果没有楼，全可见

        ray_origins = self.receivers
        vectors = sat_pos - ray_origins
        distances = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        ray_dirs = vectors / distances

        # 射线检测
        is_occluded = self.city.intersector.intersects_any(
            ray_origins=ray_origins,
            ray_directions=ray_dirs
        )

        # 0=遮挡(NLOS), 1=可见(LOS)
        # 注意：intersects_any 返回 True 表示相交(被遮挡)
        vis_flat = np.where(is_occluded, 0, 1)
        return vis_flat.reshape(self.grid_x.shape)


# ==========================================
# 第三部分：数据加载与结果输出
# ==========================================

def load_buildings_from_list(data_list):
    """
    data_list: list of dict [{'name':'A', 'x':30, 'y':30, 'w':20, 'd':20, 'h':40}, ...]
    """
    city = CityScene()
    for d in data_list:
        b = Building(d['name'], d['x'], d['y'], d['w'], d['d'], d['h'])
        city.add_building(b)
    city.build()
    return city


def load_ephemeris_data():
    """
    模拟读取文件。实际使用时可以使用 pd.read_csv('rinex_params.csv')
    这里直接返回一个包含多颗卫星参数的列表。
    参数单位需严格对应：角度为弧度，距离为米，时间为秒。
    """
    # 模拟两颗卫星的星历参数 (简化数据，仅供测试几何关系)
    # 为了演示效果，我们构造两个位置：一个高仰角(可见)，一个低仰角(遮挡)
    # 实际项目中，这些 key 对应 RINEX 文件里的参数

    # 构造一个假星历生成器 (为了产生想要的角度，反推大致参数，实际应读取真实文件)
    # 这里我们简化逻辑：主循环中我们先算出卫星位置，再根据星历参数算其实是一样的。
    # 为了代码严谨性，这里定义真实星历结构。

    sats = []

    # 卫星 G01 (高仰角)
    sats.append({
        'PRN': 'G01',
        'toe': 400000, 'sqrt_a': 5153.79, 'e': 0.01, 'm0': 0.5,
        'omega': 1.0, 'i0': 0.96, 'omega0': 0.5, 'delta_n': 4e-9,
        'idot': 0, 'omega_dot': -2.6e-9,
        'cuc': 0, 'cus': 0, 'crc': 0, 'crs': 0, 'cic': 0, 'cis': 0
    })

    # 卫星 G15 (低仰角，容易被挡)
    sats.append({
        'PRN': 'G15',
        'toe': 400000, 'sqrt_a': 5153.79, 'e': 0.01, 'm0': 2.5,  # 修改 m0 改变位置
        'omega': 1.2, 'i0': 0.96, 'omega0': 0.5, 'delta_n': 4e-9,
        'idot': 0, 'omega_dot': -2.6e-9,
        'cuc': 0, 'cus': 0, 'crc': 0, 'crs': 0, 'cic': 0, 'cis': 0
    })

    return sats


def plot_and_save_results(output_dir, sim, sat_obj, vis_matrix):
    """生成5张图并保存矩阵"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存 0/1 矩阵
    mat_filename = os.path.join(output_dir, f"{sat_obj.prn}_matrix.txt")
    np.savetxt(mat_filename, vis_matrix, fmt='%d', delimiter=',')
    print(f"  -> Matrix saved to {mat_filename}")

    # 准备绘图数据
    b_patches_xy = []
    b_patches_xz = []
    b_patches_yz = []

    # 遍历楼房生成投影矩形
    for b in sim.city.buildings:
        # XOY: x, y, w, d
        b_patches_xy.append(
            Rectangle((b.center[0] - b.dims[0] / 2, b.center[1] - b.dims[1] / 2), b.dims[0], b.dims[1], color='gray',
                      alpha=0.5))
        # XOZ: x, z, w, h (z的中心是h/2，底边是0)
        b_patches_xz.append(Rectangle((b.center[0] - b.dims[0] / 2, 0), b.dims[0], b.dims[2], color='gray', alpha=0.5))
        # YOZ: y, z, d, h
        b_patches_yz.append(Rectangle((b.center[1] - b.dims[1] / 2, 0), b.dims[1], b.dims[2], color='gray', alpha=0.5))

    # 卫星位置 (用于投影)
    sx, sy, sz = sat_obj.position

    # 创建画布
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Satellite Visibility Analysis: {sat_obj.prn}", fontsize=16)

    # 1. XOY Projection
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("1. X-Y Projection (Top View)")
    for p in b_patches_xy: ax1.add_patch(p)
    ax1.scatter(sx, sy, c='red', marker='*', s=200, label='Satellite')
    # 画一个箭头指向卫星方向 (如果卫星太远，画在图外)
    ax1.set_xlim(sim.x_lim[0] - 10, sim.x_lim[1] + 10)
    ax1.set_ylim(sim.y_lim[0] - 10, sim.y_lim[1] + 10)
    ax1.set_aspect('equal')
    ax1.grid(True)

    # 2. XOZ Projection
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("2. X-Z Projection (Side View)")
    for p in b_patches_xz: ax2.add_patch(p)
    # 为了能在图上看到卫星方向，我们需要根据比例绘制，或者只画方向
    # 这里我们简单地画出卫星的位置（注意：卫星很高，可能导致图被压缩，这里做截断处理或示意）
    # 策略：如果卫星太高，就不画点，画箭头。
    arrow_len = 50
    ax2.arrow(sim.x_lim[1] / 2, 0, sx / (np.abs(sz) + 1) * arrow_len, arrow_len, head_width=5, color='red')
    ax2.text(sim.x_lim[1] / 2, arrow_len, "  To Sat", color='red')
    ax2.set_xlim(sim.x_lim[0] - 10, sim.x_lim[1] + 10)
    ax2.set_ylim(0, 150)  # 高度限制以便看清楼
    ax2.grid(True)

    # 3. YOZ Projection
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("3. Y-Z Projection (Side View)")
    for p in b_patches_yz: ax3.add_patch(p)
    ax3.arrow(sim.y_lim[1] / 2, 0, sy / (np.abs(sz) + 1) * arrow_len, arrow_len, head_width=5, color='red')
    ax3.text(sim.y_lim[1] / 2, arrow_len, "  To Sat", color='red')
    ax3.set_xlim(sim.y_lim[0] - 10, sim.y_lim[1] + 10)
    ax3.set_ylim(0, 150)
    ax3.grid(True)

    # 4. 3D Scene View
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_title("4. 3D Building Scene")
    for b in sim.city.buildings:
        # 简单画法，画box
        x_c, y_c, z_c = b.center
        dx, dy, dz = b.dims
        ax4.bar3d(x_c - dx / 2, y_c - dy / 2, 0, dx, dy, dz, color='cyan', alpha=0.6, edgecolor='k')

    # 画一条线指向卫星
    center_map = [(sim.x_lim[1] - sim.x_lim[0]) / 2, (sim.y_lim[1] - sim.y_lim[0]) / 2, 0]
    # 归一化方向向量以便画图
    norm_vec = sat_obj.position / np.linalg.norm(sat_obj.position) * 150  # 长度150的指示线
    ax4.plot([center_map[0], center_map[0] + norm_vec[0]],
             [center_map[1], center_map[1] + norm_vec[1]],
             [center_map[2], center_map[2] + norm_vec[2]], 'r--', lw=2)
    ax4.text(center_map[0] + norm_vec[0], center_map[1] + norm_vec[1], center_map[2] + norm_vec[2], "Sat Direction",
             color='red')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlim(0, 150)

    # 5. Heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("5. Visibility Heatmap (Blue=LOS, Red=NLOS)")
    # cmap='RdYlBu' -> Red(0) to Blue(1)
    # extent: [left, right, bottom, top]
    im = ax5.imshow(vis_matrix, origin='lower',
                    extent=[sim.x_lim[0], sim.x_lim[1], sim.y_lim[0], sim.y_lim[1]],
                    cmap='bwr', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax5, ticks=[0, 1], label='0: Blocked, 1: Visible')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"{sat_obj.prn}_analysis.png")
    plt.savefig(plot_filename)
    # plt.show() # 如果跑大批量循环，建议注释掉show，只保存图片
    plt.close()
    print(f"  -> Plots saved to {plot_filename}")


# ==========================================
# 第四部分：主程序执行逻辑
# ==========================================
if __name__ == "__main__":
    # --- 1. 设置城市中心 (用于 ECEF 转 ENU) ---
    # 假设在北京
    CITY_LAT, CITY_LON, CITY_H = 39.9, 116.3, 50.0

    # --- 2. 定义/读取 楼房数据 (Building Loop Input) ---
    # 你可以把这里改成 pd.read_csv('buildings.csv').to_dict('records')
    buildings_data = [
        {'name': 'Build_A', 'x': 30, 'y': 30, 'w': 20, 'd': 20, 'h': 40},
        {'name': 'Build_B', 'x': 70, 'y': 60, 'w': 15, 'd': 40, 'h': 60},
        {'name': 'Build_C', 'x': 20, 'y': 80, 'w': 20, 'd': 10, 'h': 30},
        {'name': 'Build_D', 'x': 50, 'y': 50, 'w': 10, 'd': 10, 'h': 20},  # 新增
        {'name': 'Build_E', 'x': 85, 'y': 20, 'w': 15, 'd': 15, 'h': 50},  # 新增
    ]

    # 初始化仿真环境
    print("正在构建城市模型...")
    city_scene = load_buildings_from_list(buildings_data)

    # 设置仿真范围 (0-100m, 精度1m)
    sim = GNSS_Simulator(city_scene, x_lim=(0, 100), y_lim=(0, 100), step=1.0)

    # --- 3. 定义/读取 星历数据 (Satellite Loop Input) ---
    # 读取星历参数列表
    ephemeris_list = load_ephemeris_data()

    # 设定当前观测时间 (周内秒)
    current_time_sec = 400000 + 3600  # 假设过了一小时

    output_folder = "simulation_results_0"

    # --- 4. 主循环：遍历所有卫星进行计算 ---
    print(f"开始处理 {len(ephemeris_list)} 颗卫星...")

    for eph in ephemeris_list:
        prn = eph['PRN']
        print(f"\nProcessing Satellite {prn}...")

        # A. 轨道解算 (Math)
        # 1. 算出 ECEF 坐标
        sat_ecef = gps_orbit_calculation(eph, current_time_sec)

        # 2. 转换到 城市局部坐标 (ENU)
        sat_enu = ecef_to_enu(sat_ecef[0], sat_ecef[1], sat_ecef[2],
                              CITY_LAT, CITY_LON, CITY_H)

        # 3. 创建卫星对象
        sat_obj = Satellite(prn, sat_enu)

        # B. 运行可见性分析 (Simulation)
        visibility_matrix = sim.check_visibility(sat_obj.position)

        # C. 绘图与保存 (Visualization)
        plot_and_save_results(output_folder, sim, sat_obj, visibility_matrix)

    print("\n所有计算完成！请查看 simulation_results_0 文件夹。")