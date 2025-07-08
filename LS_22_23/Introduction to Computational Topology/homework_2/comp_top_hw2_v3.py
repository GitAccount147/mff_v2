import gudhi as gd
#from gudhi import AlphaComplex
#import pandas as pd
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
import scipy as sc


def get_components(radius, dist_matrix):
    connected = np.where(dist_matrix <= radius, 1, 0)  # connect points with an edge if they are close enough
    #print(connected)
    comps, labels = sc.sparse.csgraph.connected_components(csgraph=connected, directed=False)
    return comps, labels


def get_f_vector(simplex_tree):
    max_dim = simplex_tree.dimension()
    #f_vector = [0] * (max_dim + 1)
    f_vector = np.zeros(shape=(max_dim + 1), dtype="int64")
    for filtered_value in simplex_tree.get_filtration():
        f_vector[len(filtered_value[0]) - 1] += 1
    return f_vector


def euler_characteristic(f_vector):
    sign = np.array([(-1)**i for i in range(len(f_vector))])
    euler_char = np.sum(sign * f_vector)
    return euler_char


def get_data(pokemon_types, both_types=True, max_examples=None):
    data = {}
    for pokemon_type in pokemon_types:
        data[pokemon_type] = [[], 0]
    #data_size = [0] * len(pokemon_types)

    poke_txt_1 = open('pokemon_text_1.txt', 'r')

    for line in poke_txt_1:
        pokemon_info = (line.strip()).split(sep=',')
        type_1, type_2, attack, defense = pokemon_info[2], pokemon_info[3], pokemon_info[6], pokemon_info[7]
        for pokemon_type in pokemon_types:
            if data[pokemon_type][1] != max_examples:
                if type_1 == pokemon_type or (both_types and type_2 == pokemon_type):
                    data[pokemon_type][0].append([int(attack), int(defense)])
                    data[pokemon_type][1] += 1
        if max_examples is not None:
            #print("smth")
            if all(data[pokemon_type][1] == max_examples for pokemon_type in data.keys()):
                break

    poke_txt_1.close()
    return data


def visualize(dataset):
    # Creating a numpy array
    pc_fire_np = np.array(dataset)
    pc_fire_att = np.transpose(pc_fire_np)[0]
    pc_fire_def = np.transpose(pc_fire_np)[1]
    # X = np.array([1,2,3,-1,-2])
    # Y = np.array([6,1,-4,2,5])
    # Plotting point using scatter method
    plt.scatter(pc_fire_att, pc_fire_def)
    plt.show()


def create_table(dataset, dist_matrix, radii, max_dim=2):
    table = []
    component_labels = []

    dataset_size = len(dataset)
    print("dataset size:", dataset_size)
    #print("max distance:", max_dist)
    #print(radii)
    for radius in radii:
        #print(radius)
        connected_components, labels = get_components(radius, dist_matrix)
        rips_complex = gd.RipsComplex(points=dataset, max_edge_length=radius)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        f_vector = get_f_vector(simplex_tree)
        euler_char = euler_characteristic(f_vector)
        top_dim_cells = f_vector[-1]
        ratio = top_dim_cells / dataset_size
        #print("(r, X_r, Conn_r, Euler, f_vec):", radius, ratio, connected_components, euler_char, f_vector)
        table.append([np.round(radius, decimals=2), ratio, connected_components, euler_char])
        component_labels.append(labels)
    return table, component_labels


def driver():
    step = 0.5
    tables = {}  # dict of dicts?
    labels = {}
    point_clouds = get_data(["Fire", "Water"], max_examples=11)
    #print(point_clouds)

    for key in point_clouds.keys():
        point_cloud = point_clouds[key][0]
        #visualize(point_cloud)
        dist_matrix = sc.spatial.distance_matrix(x=point_cloud, y=point_cloud)
        max_dist = np.ceil(np.max(dist_matrix))
        min_dist = np.floor(np.min(dist_matrix + np.eye(len(dist_matrix)) * max_dist))  # shape instead of len
        radii = np.arange(min_dist, max_dist, step)
        raw_table, raw_labels = create_table(point_cloud, dist_matrix, radii)

        unique = []  # can occur more than once so not really unique
        unique_labels = []
        last_val = [-1, -1, -1]
        last_comps = -1
        for i in range(len(raw_table)):
            if last_val != raw_table[i][1:]:
                unique.append(raw_table[i])
                #unique_labels.append(raw_labels[i])
                last_val = raw_table[i][1:]
            if last_comps != raw_table[i][2]:
                unique_labels.append([raw_table[i][0], raw_table[i][2], raw_labels[i]])
                last_comps = raw_table[i][2]
        tables[key] = unique
        labels[key] = unique_labels

    return tables, labels, point_clouds

def visualize_components(dataset, labels):
    for i in range(len(labels)):
        radius, comps_num, curr_labels = labels[i]

        #fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
        # (or if you have an existing figure)
        fig = plt.gcf()
        ax = fig.gca()
        #np_arr = np.array(dataset)
        min_x = np.min(np.transpose(np.array(dataset))[0]) - radius - 5
        min_y = np.min(np.transpose(np.array(dataset))[1]) - radius - 5
        max_x = np.max(np.transpose(np.array(dataset))[0]) + radius + 5
        max_y = np.max(np.transpose(np.array(dataset))[1]) + radius + 5
        #print(dataset)
        #print(min_x, max_x, min_y, max_y)
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))
        ax.axis('equal')

        #colormap = matplotlib.colors.Colormap('hsv', N=256)
        #print(colormap())
        #gradient = np.linspace(0, 1, 256)
        #gradient = np.vstack((gradient, gradient))
        #ax.imshow(gradient, cmap='hsv')
        colormap = plt.get_cmap('hsv')
        #print(p(0.1))

        for j in range(len(curr_labels)):
            x, y = dataset[j][0], dataset[j][1]

            #print(x, y)
            #circle = matplotlib.patches.Circle((x, y), radius=radius)
            #fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
            # (or if you have an existing figure)
            # fig = plt.gcf()
            # ax = fig.gca()

            #ax.add_patch(circle)
            #color = [0.3+(1/comps_num)*curr_labels[j]/2, 0, 0.3+(1/comps_num)*curr_labels[j]/4]
            color = colormap((1/comps_num)*curr_labels[j])
            #print(color)
            circle = plt.Circle((x, y), radius=radius/2, fill=False, color=color)
            plt.scatter(x, y, color=color, marker='.')
            ax.add_patch(circle)
        #fig.savefig('comps_test_1.png')
        plt.show()
        #plt.show()
        #break

    return None


final_tables, final_labels, point_clouds = driver()
final_fire, final_water = final_tables["Fire"], final_tables["Water"]
labels_fire, labels_water = final_labels["Fire"], final_labels["Water"]
print(*final_fire, sep='\n')
#print(*labels_fire, sep='\n')
visualize_components(point_clouds["Fire"][0], labels_fire)
#print(np.array(final_fire))



# hyper-params:
#both_types = True  # consider Type1 and Type2 when deciding type  (is it different?...yes (e.g. Houndour))

#point_clouds = get_data(["Fire", "Water"], max_examples=10)
#pc_fire, pc_water = point_clouds["Fire"][0], point_clouds["Water"][0]

#visualize(pc_fire)
#visualize(pc_water)



# number of components:
#print(get_components(10, distance_matrix))  # !!!!!!!!!!!!!!!!!!!!!! is it radius or diameter?




