import numpy as np
import trimesh
import matplotlib.pyplot as plt
from typing import List, Tuple


# --- 模块1: 城市建筑生成器 ---
class Building:
    def __init__(self, center_x, center_y, width, depth, height):
        """
        定义一个长方体建筑物
        :param center_x, center_y: 建筑物底面中心坐标
        :param width: X轴方向的长度
        :param depth: Y轴方向的长度
        :param height: 高度 (Z轴)
        """
        self.center = np.array([center_x, center_y, height / 2.0])
        self.extents = np.array([width, depth, height])

        # 使用 trimesh 创建长方体几何体
        self.mesh = trimesh.creation.box(extents=self.extents)

        # 将几何体移动到指定位置 (默认box生成在原点)
        transform = np.eye(4)
        transform[:3, 3] = self.center
        self.mesh.apply_transform(transform)


class CityScene:
    def __init__(self):
        self.buildings = []
        self.scene_mesh = None

    def add_building(self, building: Building):
        self.buildings.append(building)

    def build_scene(self):
        """将所有单独的楼房合并为一个大的场景Mesh，加速运算"""
        if not self.buildings:
            raise ValueError("没有添加任何建筑物！")
        # 将所有楼房的 mesh 合并
        self.scene_mesh = trimesh.util.concatenate([b.mesh for b in self.buildings])
        # 创建射线查询器 (Ray Intersector)
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.scene_mesh)
        print(f"城市模型构建完成，包含 {len(self.buildings)} 栋建筑。")


# --- 模块2: 卫星位置模拟器 ---
class Satellite:
    def __init__(self, prn, azimuth, elevation, distance=20000000):
        """
        :param prn: 卫星编号 (如 'G01')
        :param azimuth: 方位角 (度, 0=北, 90=东)
        :param elevation: 高度角 (度, 90=头顶)
        :param distance: 卫星距离 (米)，假设很远
        """
        self.prn = prn
        self.az = np.radians(azimuth)
        self.el = np.radians(elevation)
        self.dist = distance
        self.position = self._calculate_position()

    def _calculate_position(self):
        """将方位角/高度角转换为局部坐标系(x, y, z)"""
        # 简单的球坐标转直角坐标
        z = self.dist * np.sin(self.el)
        r_xy = self.dist * np.cos(self.el)
        x = r_xy * np.sin(self.az) # 注意：导航中通常X轴朝东，Y轴朝北，这里简化处理
        y = r_xy * np.cos(self.az)
        return np.array([x, y, z])

def ephemeris_mock_processor(ephemeris_data):
    """
    模拟：接收广播星历参数，计算出卫星位置
    实际上这里应该写复杂的开普勒轨道公式。
    现在为了跑通流程，我们假设 ephemeris_data 包含直接的方位角信息。
    """
    # 假设输入是字典: {'PRN': 'G01', 'Az': 135, 'El': 45}
    return Satellite(ephemeris_data['PRN'], ephemeris_data['Az'], ephemeris_data['El'])


# --- 模块3: 仿真引擎 ---
class GNSS_Simulator:
    def __init__(self, city_scene: CityScene, x_range, y_range, step):
        self.city = city_scene
        self.step = step
        # 生成网格点
        self.xs = np.arange(x_range[0], x_range[1], step)
        self.ys = np.arange(y_range[0], y_range[1], step)
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)

        # 将网格展平为点列表 (N, 3)，高度设为1.5米(模拟手持接收机)
        self.flat_x = self.grid_x.flatten()
        self.flat_y = self.grid_y.flatten()
        self.z_height = np.full_like(self.flat_x, 1.5)

        self.receivers = np.column_stack((self.flat_x, self.flat_y, self.z_height))
        print(f"生成测试点网格: {self.receivers.shape[0]} 个点")

    def run_analysis(self, satellite: Satellite):
        """
        核心函数：计算指定卫星在整个区域的遮挡情况
        :return: 遮挡矩阵 (0=NLOS/遮挡, 1=LOS/可见)
        """
        num_points = self.receivers.shape[0]

        # 1. 构建射线源 (所有地面接收机)
        ray_origins = self.receivers

        # 2. 构建射线方向 (从接收机指向卫星)
        # 卫星位置是 (3,)，接收机是 (N, 3)，numpy会自动广播
        vectors = satellite.position - ray_origins
        # 归一化向量 (trimesh 需要方向向量是单位向量)
        norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        ray_directions = vectors / norms

        # 3. 批量射线检测 (Ray Tracing)
        # intersects_any 返回布尔数组，True表示射线碰到了物体（即被遮挡）
        print(f"正在计算卫星 {satellite.prn} 的可见性...")
        is_occluded = self.city.intersector.intersects_any(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        # 4. 转换结果
        # is_occluded: True(遮挡/NLOS), False(可见/LOS)
        # 我们想要: 0(NLOS), 1(LOS)
        visibility_flat = np.where(is_occluded, 0, 1)

        # 恢复成矩阵形状以便绘图
        visibility_matrix = visibility_flat.reshape(self.grid_x.shape)

        return visibility_matrix


# --- 模块4: 主程序执行 ---
if __name__ == "__main__":
    # 1. 搭建城市环境
    city = CityScene()

    # 添加几栋楼 (参数: center_x, center_y, width, depth, height)
    # 假设区域是 0-100米
    city.add_building(Building(30, 30, 20, 20, 40))  # 楼A
    city.add_building(Building(70, 60, 15, 40, 60))  # 楼B
    city.add_building(Building(20, 80, 20, 10, 30))  # 楼C

    city.build_scene()  # 合并模型

    # 2. 初始化仿真器 (区域 0-100米，步长1米)
    sim = GNSS_Simulator(city, x_range=(0, 100), y_range=(0, 100), step=1.0)

    # 3. 输入卫星参数 (这里模拟从星历解算出的方位/高度)
    # 场景1：卫星在头顶偏东 (Az=90, El=60) -> 应该大部分可见
    sat1_params = {'PRN': 'G01', 'Az': 90, 'El': 60}
    sat1 = ephemeris_mock_processor(sat1_params)

    # 场景2：卫星在低空 (Az=225, El=15) -> 应该被楼房大量遮挡
    sat2_params = {'PRN': 'G12', 'Az': 225, 'El': 15}
    sat2 = ephemeris_mock_processor(sat2_params)

    # 4. 运行计算
    vis_map_1 = sim.run_analysis(sat1)
    vis_map_2 = sim.run_analysis(sat2)

    # 5. 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制卫星 G01
    ax1 = axes[0]
    c1 = ax1.imshow(vis_map_1, origin='lower', extent=[0, 100, 0, 100], cmap='bwr', vmin=0, vmax=1)
    ax1.set_title(f"Sat: {sat1.prn} (Az:{sat1_params['Az']} El:{sat1_params['El']})")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    plt.colorbar(c1, ax=ax1, label='0=NLOS, 1=LOS')

    # 绘制卫星 G12
    ax2 = axes[1]
    c2 = ax2.imshow(vis_map_2, origin='lower', extent=[0, 100, 0, 100], cmap='bwr', vmin=0, vmax=1)
    ax2.set_title(f"Sat: {sat2.prn} (Az:{sat2_params['Az']} El:{sat2_params['El']})")
    ax2.set_xlabel("X (meters)")

    plt.tight_layout()
    plt.show()