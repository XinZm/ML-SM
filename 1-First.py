# 将经纬度坐标转化到长度的映射二维坐标图中
import pandas as pd
import pyproj
import matplotlib.pyplot as plt

# 1. 将CSV文件中的经纬度坐全部转为国际单位长度坐标，并读取最大最小国际单位长度坐标
file_path = "Barcelona_41k.csv"  # 文件路径，请替换成你的CSV文件路径
df = pd.read_csv(file_path)

wgs84 = pyproj.CRS("EPSG:4326")
utm = pyproj.CRS("EPSG:32633")  # 选择适合你所在区域的EPSG代码
project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)

converted_x, converted_y = project.transform(df['longitude'], df['latitude'])
min_x, min_y = converted_x.min(), converted_y.min()
max_x, max_y = converted_x.max(), converted_y.max()

# 2. 将最大最小国际单位长度坐标作为四个点，并将这四个点减去最小坐标得到新的坐标
corner_points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
corner_points = [(x - min_x, y - min_y) for x, y in corner_points]

# 3. 将每个坐标都减去最小的国际单位长度坐标得到新的坐标
converted_x -= min_x
converted_y -= min_y

# 创建新的 DataFrame 来保存新的坐标
min_coordinate_df = pd.DataFrame({
    'X coordinate': [min_x],
    'Y coordinate': [min_y]
})

# 创建新的 DataFrame 来保存新的坐标
new_df = pd.DataFrame({
    'X coordinate': converted_x,
    'Y coordinate': converted_y
})

# 将新的坐标保存到 CSV 文件
output_file_path = "converted_coordinates.csv"
new_df.to_csv(output_file_path, index=False)

# 将最小的坐标保存到 CSV 文件
output_file_path = "min_coordinates.csv"
min_coordinate_df.to_csv(output_file_path, index=False)

# 4. 以新的四个最大最小的国际单位长度坐标作为二维图的范围，将得到的坐标输入到该二维图进行打点，最终输出图像
plt.figure(figsize=(10, 10))
plt.scatter(converted_x, converted_y, s=1, marker='o', color='blue')
plt.xlim(0, max_x - min_x)
plt.ylim(0, max_y - min_y)
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.title('Scatter Plot of Longitude and Latitude in International Units')
plt.grid(True)

# 设置横纵坐标的刻度间隔（自定义值）
x_interval = 1000  # 横坐标刻度间隔 单位m
y_interval = 1000  # 纵坐标刻度间隔
plt.xticks(range(0, int(max_x - min_x), x_interval))
plt.yticks(range(0, int(max_y - min_y), y_interval))

#plt.show()
plt.legend()
plt.show()
