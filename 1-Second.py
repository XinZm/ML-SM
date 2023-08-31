# 规划区域范围，用Macro cell实现全面覆盖，并读取每个Macro cell 的中心点坐标及涵盖信息的坐标和数量

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point

# 读取CSV文件
file_path = "converted_coordinates.csv"  # 替换成您的CSV文件路径
df = pd.read_csv(file_path)

# 假设CSV文件中有'X-coordinate和'Y-coordinate这两列，请根据实际情况修改列名

x_coordinates = df['X coordinate']
y_coordinates = df['Y coordinate']

# 计算最大和最小横纵坐标
max_x = x_coordinates.max()
min_x = x_coordinates.min()
max_y = y_coordinates.max()
min_y = y_coordinates.min()

# 2. 将最大最小国际单位长度坐标作为四个点，并将这四个点减去最小坐标得到新的坐标
corner_points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
print(f"坐标范围：({max_x - min_x}, {max_y - min_y})")


# 在图中添加分界线
x_split_points = [10000, max_x]
y_split_points = [0, 13000]


"""plt.figure(figsize=(10, 10))  # 只调用一次 plt.figure()"""

# 4. 在图中添加分界线
"""plt.plot(x_split_points, y_split_points, color='red', linestyle='--')"""

# 5. 以新的四个最大最小的国际单位长度坐标作为二维图的范围，将得到的坐标输入到该二维图进行打点，最终输出图像
"""plt.scatter(converted_x, converted_y, s=1, marker='o', color='blue')"""

# 添加标签到 (max_x, min_y) 点
"""plt.text(max_x - min_x, 0, f'({max_x}, {min_y})', fontsize=10, ha='right', va='bottom')"""

# 五边形点
polygon_points = [
    (0, 0),
    (10000, 0),
    (max_x, 13000),
    (max_x, max_y),
    (0, max_y)]

polygon = Polygon(polygon_points)

# 正六边形的边长
l = 6000

# 高度（hexagon height）
h = np.sqrt(3) * l

# 横向和纵向的间距
dx = 1.5 * l
dy = h

hexagons = []
# 获取五边形的边界
minx, miny, maxx, maxy = polygon.bounds

# 扩展范围
expand_distance = 8 * l  # 选择适当的扩展距离以包括额外的六边形
x_range = np.arange(minx - expand_distance, maxx + expand_distance, dx)
y_range = np.arange(miny - expand_distance, maxy + expand_distance, dy)

hexagons = []

# 开始创建六边形
for x in x_range:
    for y in y_range:
        # Offset every second row
        if (x // dx) % 2 == 1:
            y += dy / 2

        hexagon = Polygon([
            (x - l * 1, y),
            (x - l * 0.5, y + h * 0.5),
            (x + l * 0.5, y + h * 0.5),
            (x + l * 1, y),
            (x + l * 0.5, y - h * 0.5),
            (x - l * 0.5, y - h * 0.5),
        ])

        hexagons.append(hexagon)

# 初始化编号变量
hexagon_number = 1
for hexagon in hexagons:
    center = hexagon.centroid  # 获取六边形的中心点
    #     print(f"正六边形编号: {hexagon_number}, 中心坐标: ({center.x}, {center.y})")
    hexagon_number += 1  # 编号递增

polygon_points


def rotate_around_point(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point."""
    x, y = xy
    ox, oy = origin

    qx = ox + np.cos(radians) * (x - ox) - np.sin(radians) * (y - oy)
    qy = oy + np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)

    return qx, qy


# 计算分界线的斜率
split_line_slope = (y_split_points[1] - y_split_points[0]) / (x_split_points[1] - x_split_points[0])

# 获取第一个六边形的两个顶点来计算其对角线的斜率
hexagon_top = hexagons[0].exterior.coords[1]
hexagon_bottom = hexagons[0].exterior.coords[4]
hex_diagonal_slope = (hexagon_top[1] - hexagon_bottom[1]) / (hexagon_top[0] - hexagon_bottom[0])

# 计算两个斜率之间的角度差异
angle_difference_rad = np.arctan((split_line_slope - hex_diagonal_slope) / (1 + split_line_slope * hex_diagonal_slope))

# 获取五边形的中心点作为旋转中心
rotation_center = (polygon.centroid.x, polygon.centroid.y)

# 使用先前提供的rotate_around_point函数对所有的六边形进行旋转
rotated_hexagons = []
for hexagon in hexagons:
    rotated_coords = [rotate_around_point((x, y), angle_difference_rad, rotation_center) for x, y in
                      hexagon.exterior.coords[:-1]]  # 注意去除最后一个点，因为它与第一个点重复
    rotated_hexagons.append(Polygon(rotated_coords))
# 获取最接近分界线上面点的六边形
closest_hexagon = min(rotated_hexagons, key=lambda hexagon: abs(hexagon.centroid.y - y_split_points[1]))

# 计算与分界线最接近的那一排六边形的对角线中点的x坐标
mid_point_x = (closest_hexagon.exterior.coords[1][0] + closest_hexagon.exterior.coords[4][0]) / 2

# 使用分界线的方程找到该x坐标对应的y坐标
m = (y_split_points[1] - y_split_points[0]) / (x_split_points[1] - x_split_points[0])
c = y_split_points[0] - m * x_split_points[0]
target_y = m * mid_point_x + c

# 计算移动距离
move_distance = target_y - (closest_hexagon.exterior.coords[1][1] + closest_hexagon.exterior.coords[4][1]) / 2

# 移动所有六边形
moved_hexagons = []
for hexagon in rotated_hexagons:
    moved_coords = [(x, y + move_distance) for x, y in hexagon.exterior.coords[:-1]]
    moved_hexagons.append(Polygon(moved_coords))

# 获取与五边形有实际面积交集的六边形
intersecting_hexagons = [hexagon for hexagon in moved_hexagons if polygon.intersection(hexagon).area > 0]

# 获取交集六边形的中心坐标
intersecting_centers = [(hexagon.centroid.x, hexagon.centroid.y) for hexagon in intersecting_hexagons]

# 打印中心坐标
for center in intersecting_centers:
    print(center)

# 打印总数
print(f"Total number of intersecting hexagons with actual area: {len(intersecting_centers)}")

# 假设您的散点坐标列表是scatter_points
scatter_points = [(x, y) for x, y in zip(x_coordinates, y_coordinates)]
# 定义一个空列表来保存包含散点的六边形的中心坐标
valid_intersecting_centers = []
points_in_hexagon = {(hexagon.centroid.x, hexagon.centroid.y): [] for hexagon in intersecting_hexagons}

# 遍历所有散点，查找它们所在的六边形
for point in scatter_points:
    point_obj = Point(point)
    contains_point = False  # 用于标记六边形是否包含散点
    for hexagon in intersecting_hexagons:
        center = (hexagon.centroid.x, hexagon.centroid.y)
        if hexagon.contains(point_obj):
            points_in_hexagon[center].append(point)
            contains_point = True
            break  # 当找到所属的六边形时，就不需要继续查找
    if contains_point:
        if center not in valid_intersecting_centers:
            valid_intersecting_centers.append(center)
valid_intersecting_centers

valid_intersecting_centers

intersecting_centers
# Visualization
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.set_xlabel('X-coordinate / meter')
ax.set_ylabel('Y-coordinate / meter')
ax.set_title('Macro cell overlay')

# Plotting the scatter plot again
plt.scatter(x_coordinates, y_coordinates, s=1, marker='o', color='blue')

# Drawing the points of the polygon
for point in corner_points:
    plt.scatter(point[0], point[1], color='purple', s=100)  # 使用较大的散点表示五边形的顶点

# Drawing the polygon boundary with a thicker line
polygon_x, polygon_y = polygon.exterior.xy
plt.plot(polygon_x, polygon_y, color='purple', label='Boundary', linewidth=3)  # 使用linewidth参数增加线条粗细

# Plotting each hexagon
for hexagon in moved_hexagons:
    x, y = hexagon.exterior.xy
    """plt.fill(x, y, alpha=0.3)"""  # This will fill the hexagon with a transparent color
    plt.plot(x, y, color='black')  # This will draw the outline of the hexagon
# 打印中心坐标并在图上显示编号
for idx, center in enumerate(intersecting_centers, start=1):
    plt.text(center[0], center[1], str(idx), ha='center', va='center', fontsize=50, color='red')  # 使用红色标出编号
plt.xlim(0, max_x - min_x)
plt.ylim(0, max_y - min_y)

plt.grid(True)

x_interval = 1000
y_interval = 1000
plt.xticks(range(-5000, int(max_x - min_x) + 4000, x_interval))
plt.yticks(range(-5000, int(max_y - min_y) + 5000, y_interval))

plt.legend()

plt.show()

# # 只保留value不为空的条目

# # 打印每个六边形中的点
# for key, value in points_in_hexagon.items():
#     print(f"Hexagon {key} contains points: {value}")


# 获取交叉的六边形中心，计算每个六边形中的点数
hexagon_info = []
for idx, center in enumerate(intersecting_centers, start=1):
    num_points = len(points_in_hexagon[center])
    hexagon_info.append(
        {'area number': idx, 'total included': num_points})

# 创建一个DataFrame来存储六边形信息
hexagon_df = pd.DataFrame(hexagon_info)

# 将六边形信息保存到CSV文件中
hexagon_df.to_csv('a1-hexagon.csv', index=False)

# 创建一个文件夹来存储导出的CSV文件
import os

output_folder = "hexagon_data_csv"
os.makedirs(output_folder, exist_ok=True)

# 遍历不同区域的数据，导出为CSV文件
for idx, points_list in enumerate(points_in_hexagon.values(), start=1):
    x_coords = [point[0] for point in points_list]
    y_coords = [point[1] for point in points_list]

    data = {

        "X Coordinate": x_coords,
        "Y Coordinate": y_coords

    }

    df = pd.DataFrame(data)

    df.loc[-1] = ["total included", "", len(points_list)]  # 添加信息行
    df.index = df.index + 1  # 更新索引
    df = df.sort_index()  # 按索引排序
    filename = f"{output_folder}/hexagon_data_{idx}.csv"
    df.to_csv(filename, index=False)
