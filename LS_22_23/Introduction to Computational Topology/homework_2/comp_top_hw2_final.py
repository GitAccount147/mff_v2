import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sc


# create graph where edges connect only vertices that are close enough, then get the number of components
def get_components(max_distance, distance_matrix):
    connected = np.where(distance_matrix <= max_distance, 1, 0)  # max_dist in Cech complex is <= 2*radius
    components, labels = sc.sparse.csgraph.connected_components(csgraph=connected, directed=False)
    return components, labels


# calculate f_vector of a simplex tree
def get_f_vector(simplex_tree):
    max_dim = simplex_tree.dimension()
    f_vector = np.zeros(shape=(max_dim + 1), dtype="int64")
    for filtered_value in simplex_tree.get_filtration():
        f_vector[len(filtered_value[0]) - 1] += 1
    return f_vector


# calculate euler characteristic of a simplex tree based on its f_vector
def euler_characteristic(f_vector):
    sign = np.array([(-1)**i for i in range(len(f_vector))])
    euler_char = np.sum(sign * f_vector)
    return euler_char


# returns data_params about pokemons that suffice condition (selection_values in selection_params)
def get_data(data_params, selection_values, selection_params, max_examples=None):
    param_types = ["#", "Name", "Type 1", "Type 2", "Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed",
                   "Generation", "Legendary"]
    selection_param_indices, data_param_indices = [], []
    for param in selection_params:
        if param not in param_types:
            raise Exception("No parameter like this in database.")
        else:
            selection_param_indices.append(param_types.index(param))
    for param in data_params:
        if param not in param_types:
            raise Exception("No parameter like this in database.")
        else:
            data_param_indices.append(param_types.index(param))

    data, data_size = {}, {}
    for value in selection_values:
        data[value] = []
        data_size[value] = 0

    poke_txt_1 = open('pokemon_text_1.txt', 'r')  # the csv file turned into a txt file

    for line in poke_txt_1:
        pokemon_info = (line.strip()).split(sep=',')
        current_param_values, current_data_values = [], []
        for index in selection_param_indices:
            current_param_values.append(pokemon_info[index])
        for index in data_param_indices:
            current_data_values.append(int(pokemon_info[index]))
        for value in selection_values:
            if max_examples is None or data_size[value] < max_examples:
                if value in current_param_values:
                    data[value].append(current_data_values)
                    data_size[value] += 1
        if max_examples is not None:
            if all(data_size[value] >= max_examples for value in data_size.keys()):
                break

    poke_txt_1.close()
    return data


# simple visualization of points in dataset
def visualize(dataset):
    data_np = np.array(dataset)
    data_np_x = np.transpose(data_np)[0]
    data_np_y = np.transpose(data_np)[1]
    plt.scatter(data_np_x, data_np_y)
    plt.show()


# creates table with values (radius; ratio=(# max dim cells)/(# pts in dataset); connected components; euler char)
def create_table(dataset, dist_matrix, radii, max_dim=2, verbosity=0):
    table, component_labels = [], []

    dataset_size = len(dataset)
    for radius in radii:
        if verbosity == 1:
            print("Calculating radius:", radius)
        connected_components, labels = get_components(2 * radius, dist_matrix)
        rips_complex = gd.RipsComplex(points=dataset, max_edge_length=(2 * radius))
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        f_vector = get_f_vector(simplex_tree)
        euler_char = euler_characteristic(f_vector)
        top_dim_cells = f_vector[-1]
        ratio = top_dim_cells / dataset_size
        table.append([np.round(radius, decimals=2), ratio, connected_components, euler_char])
        component_labels.append(labels)  # for visualization
    return table, component_labels


# main function - creates table and removes rows where nothing changes
def driver(data_params, selection_values, selection_params, max_examples=None, step=0.01, max_dim=2):
    tables = {}
    labels = {}
    point_clouds = get_data(data_params, selection_values, selection_params, max_examples=max_examples)

    for key in point_clouds.keys():
        point_cloud = point_clouds[key]

        # calculate distance matrix and range of radii that can affect the table values (larger values of radius will
        # not have impact on the table values):
        dist_matrix = sc.spatial.distance_matrix(x=point_cloud, y=point_cloud)
        max_dist = np.ceil(np.max(dist_matrix) / 2)
        min_dist = np.floor(np.min(dist_matrix + np.eye(len(dist_matrix)) * max_dist) / 2)
        radii = np.arange(min_dist, max_dist, step)

        raw_table, raw_labels = create_table(point_cloud, dist_matrix, radii, max_dim=max_dim)

        # filter repeating table rows
        unique = []  # can occur more than once (so actually not unique)
        unique_labels = []  # for visualization
        last_val = [-1, -1, -1]
        last_comps = -1
        for i in range(len(raw_table)):
            if last_val != raw_table[i][1:]:
                unique.append(raw_table[i])
                last_val = raw_table[i][1:]
            if last_comps != raw_table[i][2]:
                unique_labels.append([raw_table[i][0], raw_table[i][2], raw_labels[i]])
                last_comps = raw_table[i][2]
        tables[key] = unique
        labels[key] = unique_labels

    return tables, labels, point_clouds


# visualize the balls in Cech complex for every row where the number of connected components changes
def visualize_components(dataset, labels):
    for i in range(len(labels)):
        radius, comps_num, curr_labels = labels[i]  # current data

        fig = plt.gcf()
        ax = fig.gca()
        plt.title("Pokemon")
        ax.set_xlabel("Attack")
        ax.set_ylabel("Defense")

        np_array = np.transpose(np.array(dataset))

        # set the figure size:
        min_x, min_y = np.min(np_array, axis=1) - radius - 5
        max_x, max_y = np.max(np_array, axis=1) + radius + 5
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))
        ax.axis('equal')

        colormap = plt.get_cmap('hsv')  # other choices: 'gnuplot'

        for j in range(len(curr_labels)):
            x, y = dataset[j][0], dataset[j][1]
            color = colormap((1 / comps_num) * curr_labels[j])  # color depends on the component the point/circle is in

            circle = plt.Circle((x, y), radius=radius, fill=False, color=color)
            plt.scatter(x, y, color=color, marker='.')  # point
            ax.add_patch(circle)
        plt.show()
    return None


# Run the code on the choices from exercise.
# - Use step=0.01 as the smallest length possible (we could be more precise by looking at the data from the distance
#   matrix but the complexity would increase because of multi-intersections).
# - Use either max_examples=10 for smaller dataset od max_examples=None for the full dataset.
# - Restrict the Rips complex to 2 dimensions (the complexity of the calculations would increase dramatically for full
#   dataset if we would not restrict it).

final_tables, final_labels, final_point_clouds = driver(["Attack", "Defense"], ["Fire", "Water"], ["Type 1", "Type 2"],
                                                        max_examples=None, step=0.01, max_dim=2)
final_fire, final_water = final_tables["Fire"], final_tables["Water"]
labels_fire, labels_water = final_labels["Fire"], final_labels["Water"]

# print the results:
#print(*final_fire, sep='\n')
#print(*final_water, sep='\n')

# for visualization:
#visualize_components(final_point_clouds["Fire"], labels_fire)
#visualize_components(final_point_clouds["Water"], labels_water)



# create a gif for Fire Pokemon:
fig = plt.figure()

data_fire_vis, labels_fire_vis = final_point_clouds["Fire"], labels_fire
frames_fire = len(labels_fire_vis)


def fire_animation_func(i):
    plt.clf()
    radius, comps_num, curr_labels = labels_fire_vis[i]  # current data

    fig = plt.gcf()
    ax = fig.gca()
    plt.title("Fire Pokemon")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Defense")

    np_array = np.transpose(np.array(data_fire_vis))

    # set the figure size:
    min_x, min_y = np.min(np_array, axis=1) - radius - 5
    max_x, max_y = np.max(np_array, axis=1) + radius + 5
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    ax.axis('equal')

    colormap = plt.get_cmap('hsv')  # other choices: 'gnuplot'

    res = []

    for j in range(len(curr_labels)):
        x, y = data_fire_vis[j][0], data_fire_vis[j][1]
        color = colormap((1 / comps_num) * curr_labels[j])  # color depends on the component the point/circle is in

        circle = plt.Circle((x, y), radius=radius, fill=False, color=color)
        point = plt.scatter(x, y, color=color, marker='.')  # point
        ax.add_patch(circle)
        res.append(circle)
        res.append(point)

    return res


fire_animation = FuncAnimation(fig, fire_animation_func, frames=frames_fire)

#fire_animation.save('fire_animation_all_ex_1.gif', writer='PillowWriter', fps=2)  # ffmpeg



# create a gif for Water Pokemon:

data_water_vis, labels_water_vis = final_point_clouds["Water"], labels_water
frames_water = len(labels_water_vis)


def water_animation_func(i):
    plt.clf()
    radius, comps_num, curr_labels = labels_water_vis[i]  # current data

    fig = plt.gcf()
    ax = fig.gca()
    plt.title("Water Pokemon")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Defense")

    np_array = np.transpose(np.array(data_water_vis))

    # set the figure size:
    min_x, min_y = np.min(np_array, axis=1) - radius - 5
    max_x, max_y = np.max(np_array, axis=1) + radius + 5
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    ax.axis('equal')

    colormap = plt.get_cmap('hsv')  # other choices: 'gnuplot'

    res = []

    for j in range(len(curr_labels)):
        x, y = data_water_vis[j][0], data_water_vis[j][1]
        color = colormap((1 / comps_num) * curr_labels[j])  # color depends on the component the point/circle is in

        circle = plt.Circle((x, y), radius=radius, fill=False, color=color)
        point = plt.scatter(x, y, color=color, marker='.')  # point
        ax.add_patch(circle)
        res.append(circle)
        res.append(point)

    return res


water_animation = FuncAnimation(fig, water_animation_func, frames=frames_water)

#water_animation.save('water_animation_all_ex.gif', writer='PillowWriter', fps=2)


# get the csv file (probably could be done more effectively):
file_csv_fire = open("fire_10_ex.txt", "w")
file_csv_fire.write("r,X_r,Conn_r,euler_rips_r\n")
for i in range(len(final_fire)):
    line = str(final_fire[i][0]) + ',' + str(final_fire[i][1]) + ',' + str(final_fire[i][2]) + ',' + str(final_fire[i][3]) + '\n'
    file_csv_fire.write(line)
file_csv_fire.close()

file_csv_water = open("water_10_ex.txt", "w")
file_csv_water.write("r,X_r,Conn_r,euler_rips_r\n")
for i in range(len(final_water)):
    line = str(final_water[i][0]) + ',' + str(final_water[i][1]) + ',' + str(final_water[i][2]) + ',' + str(final_water[i][3]) + '\n'
    file_csv_water.write(line)
file_csv_water.close()

