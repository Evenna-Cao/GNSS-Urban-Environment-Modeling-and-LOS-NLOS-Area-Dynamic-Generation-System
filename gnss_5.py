import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import georinex as gr
import xarray as xr
import pandas as pd

# ==========================================
# 核心数学库 (保持不变)
# ==========================================
MU = 3.986005e14
OMEGA_E = 7.2921151467e-5


def gps_orbit_calculation(eph, t_input):
    # 这里 t_input 必须等于 eph['toe']，因为我们不推演
    t_k = t_input - eph['toe']

    # 即使不推演，跨周检查也是必要的
    if t_k > 302400: t_k -= 604800
    if t_k < -302400: t_k += 604800

    A = eph['sqrt_a'] ** 2
    n0 = np.sqrt(MU / A ** 3)
    n = n0 + eph['delta_n']
    M_k = eph['m0'] + n * t_k
    E_k = M_k
    for _ in range(10):
        E_new = M_k + eph['e'] * np.sin(E_k)
        if abs(E_new - E_k) < 1e-12: break
        E_k = E_new

    sin_v = (np.sqrt(1 - eph['e'] ** 2) * np.sin(E_k)) / (1 - eph['e'] * np.cos(E_k))
    cos_v = (np.cos(E_k) - eph['e']) / (1 - eph['e'] * np.cos(E_k))
    v_k = np.arctan2(sin_v, cos_v)

    phi_k = v_k + eph['omega']
    sin_2phi = np.sin(2 * phi_k)
    cos_2phi = np.cos(2 * phi_k)

    du_k = eph['cus'] * sin_2phi + eph['cuc'] * cos_2phi
    dr_k = eph['crs'] * sin_2phi + eph['crc'] * cos_2phi
    di_k = eph['cis'] * sin_2phi + eph['cic'] * cos_2phi

    u_k = phi_k + du_k
    r_k = A * (1 - eph['e'] * np.cos(E_k)) + dr_k
    i_k = eph['i0'] + eph['idot'] * t_k + di_k

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)
    omega_k = eph['omega0'] + (eph['omega_dot'] - OMEGA_E) * t_k - OMEGA_E * eph['toe']

    x = x_k_prime * np.cos(omega_k) - y_k_prime * np.cos(i_k) * np.sin(omega_k)
    y = x_k_prime * np.sin(omega_k) + y_k_prime * np.cos(i_k) * np.cos(omega_k)
    z = y_k_prime * np.sin(i_k)
    return np.array([x, y, z])


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a ** 2 - b ** 2) / a ** 2
    phi = np.radians(lat0);
    lam = np.radians(lon0)
    sin_phi = np.sin(phi);
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam);
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
# 仿真类与绘图 (保持不变，省略部分细节)
# ==========================================
class Building:
    def __init__(self, name, x, y, w, d, h):
        self.center = np.array([x, y, h / 2.0])
        self.dims = np.array([w, d, h])
        self.mesh = trimesh.creation.box(extents=self.dims)
        transform = np.eye(4);
        transform[:3, 3] = self.center
        self.mesh.apply_transform(transform)


class CityScene:
    def __init__(self):
        self.buildings = []
        self.intersector = None

    def add_building(self, b):
        self.buildings.append(b)

    def build(self):
        if not self.buildings: return
        mesh = trimesh.util.concatenate([b.mesh for b in self.buildings])
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector
            self.intersector = RayMeshIntersector(mesh)
        except:
            self.intersector = trimesh.ray.list_engines(mesh)[0]


class GNSS_Simulator:
    def __init__(self, city, x_lim, y_lim, step):
        self.city = city
        self.x_lim = x_lim;
        self.y_lim = y_lim
        xs = np.arange(x_lim[0], x_lim[1] + step, step)
        ys = np.arange(y_lim[0], y_lim[1] + step, step)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)
        self.receivers = np.column_stack((self.grid_x.flatten(), self.grid_y.flatten(), np.full(self.grid_x.size, 1.5)))

    def check_visibility(self, sat_pos):
        if self.city.intersector is None: return np.ones_like(self.grid_x)
        vectors = sat_pos - self.receivers
        dists = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        is_occ = self.city.intersector.intersects_any(ray_origins=self.receivers, ray_directions=vectors / dists)
        return np.where(is_occ, 0, 1).reshape(self.grid_x.shape)


def plot_and_save(output_dir, sim, prn, pos, vis, timestamp_str):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, f"{prn}_matrix.txt"), vis, fmt='%d', delimiter=',')

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Sat: {prn} | Raw Epoch: {timestamp_str}")

    # 仅画一张热力图示例 (为了代码简洁，你可以把之前完整的5张图代码粘回来)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(vis, origin='lower', extent=[*sim.x_lim, *sim.y_lim], cmap='bwr', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.savefig(os.path.join(output_dir, f"{prn}_analysis.png"))
    plt.close()


def extract_eph(sat_ds):
    # 提取单条记录的参数
    return {
        'toe': float(sat_ds['Toe']), 'sqrt_a': float(sat_ds['sqrtA']), 'e': float(sat_ds['Eccentricity']),
        'm0': float(sat_ds['M0']), 'delta_n': float(sat_ds['DeltaN']), 'omega': float(sat_ds['omega']),
        'omega0': float(sat_ds['Omega0']), 'omega_dot': float(sat_ds['OmegaDot']), 'i0': float(sat_ds['Io']),
        'idot': float(sat_ds['IDOT']), 'cus': float(sat_ds['Cus']), 'cuc': float(sat_ds['Cuc']),
        'crs': float(sat_ds['Crs']), 'crc': float(sat_ds['Crc']), 'cis': float(sat_ds['Cis']),
        'cic': float(sat_ds['Cic'])
    }


# ==========================================
# 主程序：直接读取模式
# ==========================================
if __name__ == "__main__":
    OUTPUT_ROOT = "results_raw_mode"
    nav_file = "hkcl0270.26n"  # 假设你的文件名

    # 1. 场景
    city = CityScene()
    city.add_building(Building('B1', 50, 50, 20, 20, 50))
    city.build()
    sim = GNSS_Simulator(city, (0, 100), (0, 100), 1.0)

    # 2. 读取
    try:
        nav_ds = gr.load(nav_file)
    except Exception as e:
        print(f"读取错误: {e}")
        exit()

    print("开始处理原始星历记录...")

    # 3. 遍历每一颗卫星
    for sv in nav_ds.sv.values:
        sv_str = str(sv)
        if not sv_str.startswith('G'): continue  # 只看GPS

        # 提取该卫星所有有效的时间点
        sat_all_data = nav_ds.sel(sv=sv).dropna(dim='time', how='all')

        # 4. 遍历该卫星的每一个时间刻度 (不推演，有多少条读多少条)
        for t in sat_all_data.time.values:
            # 获取该具体时刻的数据切片
            data_slice = sat_all_data.sel(time=t)

            # 转换为字典
            try:
                eph = extract_eph(data_slice)
            except:
                continue

            # 使用 Toe 作为计算时刻 (不推演)
            current_time = eph['toe']

            # 计算
            ecef = gps_orbit_calculation(eph, current_time)
            enu = ecef_to_enu(ecef[0], ecef[1], ecef[2], 22.3, 114.2, 50.0)

            if enu[2] > 0:
                vis = sim.check_visibility(enu)

                # 创建按时间命名的文件夹
                # timestamp 格式转换，例如: 2026-01-28T06:00:00
                ts = pd.to_datetime(t)
                time_folder = f"{ts.strftime('%Y%m%d_%H%M%S')}"
                save_dir = os.path.join(OUTPUT_ROOT, time_folder)

                plot_and_save(save_dir, sim, sv_str, enu, vis, str(ts))
                print(f"[{sv_str}] {ts} 处理完成 -> {save_dir}")