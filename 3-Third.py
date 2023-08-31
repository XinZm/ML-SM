import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon

# 设置文件名的基本路径和文件名格式
base_file_path = "hexagon_data_"  # 文件名的基本路径
file_extension = ".csv"  # 文件扩展名

i = 9  # 六边形的数量
m = 14  # 数据所涵盖的天数
cluster_centers_all = []

# 循环处理多个文件
for file_number in range(1, i):  # 假设您有文件从 1 到 i 编号
    # 构建完整的文件路径
    file_name = f"{base_file_path}{file_number}{file_extension}"
    df = pd.read_csv(file_name)

    # 获取数据点数量，如果数量大于0才进行聚类
    total_points = len(df) - 1  # 总行数减1
    Demarcation_point = total_points - (m * 160)  # 减去Macro cell 对应的人数 800Mbps ————5Mbps
    if Demarcation_point > 0:  # 判断是否需要Micro cell进行支持
        num_clusters = int(np.ceil(Demarcation_point / (m * 200)))  # 计算聚类的数量，取最小大于等于Demarcation_point / (m * 200)的正整数

        converted_x = df['X Coordinate']
        converted_y = df['Y Coordinate']

        # 转换为聚类所需的坐标矩阵
        data = np.column_stack((converted_x, converted_y))

        # 使用 K-Means 算法进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data)

        # 存储聚类中心
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_all.extend(cluster_centers)

# 创建一个 DataFrame 来存储所有聚类中心点坐标信息
cluster_centers_df = pd.DataFrame(cluster_centers_all, columns=['X Coordinate', 'Y Coordinate'])

# 将所有聚类中心点信息保存到 CSV 文件
cluster_centers_df.to_csv('all_Micro cell_centers.csv', index=False)

# 读取点的坐标
csv_data = cluster_centers_df
new_centers = [(row['X Coordinate'], row['Y Coordinate']) for index, row in csv_data.iterrows()]

# 绘制散点图和正六边形边界在同一张图上
plt.figure(figsize=(10, 8))

# 绘制新的正六边形 边界为粉色
l_1 = 1500
for file_counter, center in enumerate(new_centers, start=1):  # 使用enumerate生成序号
    hexagon = Polygon([
        (center[0] - l_1 * 1, center[1]),
        (center[0] - l_1 * 0.5, center[1] + np.sqrt(3) * l_1 * 0.5),
        (center[0] + l_1 * 0.5, center[1] + np.sqrt(3) * l_1 * 0.5),
        (center[0] + l_1 * 1, center[1]),
        (center[0] + l_1 * 0.5, center[1] - np.sqrt(3) * l_1 * 0.5),
        (center[0] - l_1 * 0.5, center[1] - np.sqrt(3) * l_1 * 0.5),
    ])

    x_1, y_1 = hexagon.exterior.xy
    plt.plot(x_1, y_1, color='black')

    # 在图上显示编号
    plt.text(center[0], center[1], str(file_counter), ha='center', va='center', fontsize=20, color='green')

# 读取数据点的坐标信息
data_points = pd.read_csv('converted_coordinates.csv')

x_coordinates = data_points['X coordinate']
y_coordinates = data_points['Y coordinate']

# 绘制散点图
plt.scatter(cluster_centers_df['X Coordinate'], cluster_centers_df['Y Coordinate'], s=2, marker='x', color='red',
            label='Cluster Centers')
plt.scatter(x_coordinates, y_coordinates, s=1, marker='o', color='blue')
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.title('Cluster Centers on Map')
plt.grid(True)

plt.legend()
plt.show()
