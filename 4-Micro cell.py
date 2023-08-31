import csv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_hexagon_area(center_x, center_y, side_length, ax):
    angle = np.linspace(0, 2 * np.pi, 7)
    hexagon_x = center_x + side_length * np.cos(angle)
    hexagon_y = center_y + side_length * np.sin(angle)

    hexagon_verts = list(zip(hexagon_x, hexagon_y))

    hexagon_area = patches.Polygon(hexagon_verts, closed=True, edgecolor='black', linewidth=1, fill=False)
    ax.add_patch(hexagon_area)

    return hexagon_verts  # 返回顶点坐标


def find_nearest_hexagon(point, hexagon_centers):
    nearest_hexagon_index = None
    min_distance = float('inf')

    for idx, (center_x, center_y) in enumerate(hexagon_centers, start=1):
        distance = np.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_hexagon_index = idx

    return nearest_hexagon_index


def main():
    with open('all_Micro cell_centers.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        hexagon_centers = [(float(row[0]), float(row[1])) for row in reader]

    with open('converted_coordinates.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        scatter_points = [(float(row[0]), float(row[1])) for row in reader]

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X-coordinate / meter')
    ax.set_ylabel('Y-coordinate / meter')
    ax.set_title('Schematic diagram of Micro cell')

    side_length = 1500

    hexagon_results = []

    for hexagon_index, (center_x, center_y) in enumerate(hexagon_centers, start=1):
        hexagon_verts = plot_hexagon_area(center_x, center_y, side_length, ax)

        hexagon_path = patches.Path(np.array(hexagon_verts))
        contained_points = []

        for point in scatter_points:
            if hexagon_path.contains_point(point):
                nearest_hexagon_index = find_nearest_hexagon(point, hexagon_centers)
                if nearest_hexagon_index == hexagon_index:
                    contained_points.append(point)

        hexagon_csv_filename = f'Micro cell_{hexagon_index}.csv'

        with open(hexagon_csv_filename, 'w', newline='') as hexagon_csvfile:
            writer = csv.writer(hexagon_csvfile)
            writer.writerow(['X Coordinate', 'Y Coordinate'])
            writer.writerows(contained_points)

        num_points = len(contained_points)
        hexagon_results.append((hexagon_index, num_points))
        ax.text(center_x, center_y, str(hexagon_index), ha='center', va='center', fontsize=20, color='black')

        color = (np.random.random(), np.random.random(), np.random.random())  # Generate a random color
        plt.scatter(*zip(*contained_points), marker='.', color=color, s=10, label=f'Micro cell {hexagon_index}')

    # plt.scatter(*zip(*scatter_points), marker='.', color='blue', s=1)

    plt.legend()
    plt.show()

    total_results = [(hexagon_index, num_points) for hexagon_index, num_points in hexagon_results]
    with open('total_Microcell_points.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Hexagon_Index', 'Total_Points'])
        writer.writerows(total_results)


if __name__ == '__main__':
    main()
