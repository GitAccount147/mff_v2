import gudhi as gd
#from gudhi import AlphaComplex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
#from scipy.spatial import distance_matrix


def get_components(radius, dist_matrix):
    connected = np.where(dist_matrix <= radius, 0, 1)
    comps, labels = sc.sparse.csgraph.connected_components(csgraph=connected, directed=False)
    return comps, labels

def get_f_vector(simplex_tree):
    max_dim = simplex_tree.dimension
    f_vector = [0] * max_dim




# hyper-params:
both_types = True  # consider Type1 and Type2 when deciding type  (is it different?...yes (e.g. Houndour))

#gd.alpha_complex()
#pc_fire = gd.read_lower_triangular_matrix_from_csv_file("Pokemon.csv")
#print(pc_fire)

poke_txt_1 = open('pokemon_text_1.txt', 'r')
poke_fire_txt_1 = open('pokemon_fire_text_1.txt', 'w')
count = 0

pc_fire_str = ""
pc_fire_list = []

for line in poke_txt_1:
    count += 1
    #print("Line{}: {}".format(count, line.strip()))
    pokemon_info = (line.strip()).split(sep=',')
    #print(split)
    if pokemon_info[2] == 'Fire' or (both_types and pokemon_info[3] == 'Fire'):
        #print(line)
        #print(line.strip())
        attack, defense = pokemon_info[6], pokemon_info[7]
        pc_fire_str += attack + ',' + defense + '\n'
        pc_fire_list.append([int(attack), int(defense)])
        #print(pokemon_info)



poke_fire_txt_1.write(pc_fire_str)
    #if count > 10:
    #    break

poke_txt_1.close()
poke_fire_txt_1.close()


#read_file = pd.read_csv("pokemon_fire_text_1.txt")
#read_file.to_csv(r'C:\Users\Ron\Desktop\Test\New_Products.csv', index=None)
#read_file.to_csv("pokemon_fire_text_1.csv", index=None)

#pc_fire = gd.read_lower_triangular_matrix_from_csv_file("pokemon_fire_text_1.csv")
#print(pc_fire)


# Creating a numpy array
pc_fire_np = np.array(pc_fire_list)
pc_fire_att = np.transpose(pc_fire_np)[0]
pc_fire_def = np.transpose(pc_fire_np)[1]
#X = np.array([1,2,3,-1,-2])
#Y = np.array([6,1,-4,2,5])
# Plotting point using scatter method
#plt.scatter(pc_fire_att, pc_fire_def)
#plt.show()


# count the distances of all points:
#pts = np.array([[0, 1], [1, 1], [1, 0]])
#print(np.outer())
#distances = np.linalg.norm(test_data[i] - train_data, axis=1)

test_set = [[0, 1], [1, 1], [0, 0], [1, 0]]
distance_matrix = sc.spatial.distance_matrix(x=test_set, y=test_set)
print(distance_matrix)

# number of components:
print(get_components(1, distance_matrix))


#print(pc_fire_list)
ac_fire = gd.alpha_complex.AlphaComplex(points=pc_fire_list)
#gd.import_module("gudhi.cech_complex")

diam = 1.5
rips_complex_fire = gd.RipsComplex(points=pc_fire_list, max_edge_length=diam)
#cc_fire = gd.cech_complex_from_points(points=pc_fire_list, max_radius=radius)
simplex_tree = rips_complex_fire.create_simplex_tree(max_dimension=10)
print('Alpha complex is of dimension ', simplex_tree.dimension(), ' - ',
      simplex_tree.num_simplices(), ' simplices - ', simplex_tree.num_vertices(), ' vertices.')

fmt = '%s -> %.2f'
for filtered_value in simplex_tree.get_filtration():
    print(fmt % tuple(filtered_value))
    #print(filtered_value)
