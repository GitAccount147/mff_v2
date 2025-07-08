import gudhi as gd
#from gudhi import AlphaComplex
import pandas as pd
import numpy as np
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

    dataset_size = len(dataset)
    #print("max distance:", max_dist)
    #print(radii)
    for radius in radii:
        #print(radius)
        connected_components = get_components(radius, dist_matrix)[0]
        rips_complex = gd.RipsComplex(points=dataset, max_edge_length=radius)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        f_vector = get_f_vector(simplex_tree)
        euler_char = euler_characteristic(f_vector)
        top_dim_cells = f_vector[-1]
        ratio = top_dim_cells / dataset_size
        #print("(r, X_r, Conn_r, Euler, f_vec):", radius, ratio, connected_components, euler_char, f_vector)
        table.append([np.round(radius, decimals=2), ratio, connected_components, euler_char])
    return table


def driver():
    point_clouds = get_data(["Fire", "Water"], max_examples=10)
    pc_fire, pc_water = point_clouds["Fire"][0], point_clouds["Water"][0]

    dist_matrix_fire = sc.spatial.distance_matrix(x=pc_fire, y=pc_fire)
    dist_matrix_water = sc.spatial.distance_matrix(x=pc_water, y=pc_water)

    max_dist_fire = np.ceil(np.max(dist_matrix_fire))
    max_dist_water = np.ceil(np.max(dist_matrix_water))

    min_dist_fire = np.floor(np.min(dist_matrix_fire + np.eye(len(dist_matrix_fire)) * max_dist_fire))
    min_dist_water = np.floor(np.min(dist_matrix_water + np.eye(len(dist_matrix_water)) * max_dist_water))
    #print("max dist fire:", max_dist_fire)

    radii_fire = np.arange(min_dist_fire, max_dist_fire, 0.01)
    radii_water = np.arange(min_dist_water, max_dist_water, 0.01)
    #print("radii_fire:", radii_fire)

    raw_table_fire = create_table(pc_fire, dist_matrix_fire, radii_fire)
    raw_table_water = create_table(pc_water, dist_matrix_water, radii_water)

    #unique_fire = np.unique(np.array(raw_table_fire)[:, 1:], axis=0)
    #unique_water = np.unique(np.array(raw_table_water)[:, 1:], axis=0)

    unique_fire = []
    last_val = [-1, -1, -1]
    for i in range(len(raw_table_fire)):
        if last_val != raw_table_fire[i][1:]:
            unique_fire.append(raw_table_fire[i])
            last_val = raw_table_fire[i][1:]

    unique_water = []
    last_val = [-1, -1, -1]
    for i in range(len(raw_table_water)):
        if last_val != raw_table_water[i][1:]:
            unique_water.append(raw_table_water[i])
            last_val = raw_table_water[i][1:]

    return unique_fire, unique_water


final_fire, final_water = driver()
print(*final_fire, sep='\n')
#print(np.array(final_fire))



# hyper-params:
#both_types = True  # consider Type1 and Type2 when deciding type  (is it different?...yes (e.g. Houndour))

#point_clouds = get_data(["Fire", "Water"], max_examples=10)
#pc_fire, pc_water = point_clouds["Fire"][0], point_clouds["Water"][0]

#visualize(pc_fire)
#visualize(pc_water)


# count the distances of all points:
test_set = [[0, 1], [2, 2], [0, 0], [1, 0]]
#distance_matrix = sc.spatial.distance_matrix(x=test_set, y=test_set)
#distance_matrix = sc.spatial.distance_matrix(x=pc_fire, y=pc_fire)
#print("max distance:", np.max(distance_matrix))
#print(distance_matrix)

# number of components:
#print(get_components(10, distance_matrix))  # !!!!!!!!!!!!!!!!!!!!!! is it radius or diameter?


"""
diam = 10
rips_complex_fire = gd.RipsComplex(points=pc_fire, max_edge_length=diam)
#cc_fire = gd.cech_complex_from_points(points=pc_fire_list, max_radius=radius)
simplex_tree = rips_complex_fire.create_simplex_tree(max_dimension=2)
print('Rips complex is of dimension ', simplex_tree.dimension(), ' - ',
      simplex_tree.num_simplices(), ' simplices - ', simplex_tree.num_vertices(), ' vertices.')

f_vec = get_f_vector(simplex_tree)
e_char = euler_characteristic(f_vec)
print("f_vector:", f_vec)
print("euler char:", e_char)

fmt = '%s -> %.2f'
for filtered_value in simplex_tree.get_filtration():
    placeholder = 0
    #print(fmt % tuple(filtered_value))
    #print(filtered_value)

#print(np.full((3, 3), [1, 2]))
#print([(-1)**i for i in range(5)])
"""

"""
fire_table = create_table(pc_fire, np.arange(0.01, 10, 4))
print("radius, ratio, connected_components, euler_char:")
print(fire_table)
arr = np.array(fire_table)
print(arr)
nparr = arr[:, 1:]
print(nparr)
print(np.unique(nparr, axis=0))
"""

#arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [-1, -2, -3, -4, -5]])
#print(arr[0:3, 1:4])
#print(fire_table)
