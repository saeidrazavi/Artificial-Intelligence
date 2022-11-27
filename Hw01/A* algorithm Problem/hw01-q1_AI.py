import numpy as np
import time
import heapq


def update(i, j, list):
    """
    this function get two index and replace their elements by each other ! 

    inputs : 
      i : first index
      j : second index
      list : list before updating 

    output : 
          updated list after replacing   
    """
    updated_list = list.copy()
    updated_list[i], updated_list[j] = updated_list[j], updated_list[i]
    return updated_list

# -----------------------------------------------
# ------------------------------------------------


def distance_cal(i_indice, j_indice, adjacency_dic):
    """
    this function, calculate distance between two vertices 
    with respect to the edges using BFS

    inputs : 
      i_indice : vertex i-th
      j_indice : vertex j-th
      adjacency_dic : list before updating 

    output : 
          minimum distance between i_th and j_th vertex  
    """

    if(i_indice == j_indice):
        return 0
    dist = 1
    goal = np.copy(i_indice)
    queue_dic = {k: [] for k in range(len(adjacency_dic[i_indice]))}
    for i, candidate_vertex in enumerate(adjacency_dic[i_indice]):
        queue_dic[i].append(candidate_vertex)
        queue_dic[i].append(dist)
        if(candidate_vertex == j_indice):
            return 1

    while(i_indice == goal):

        key = list(queue_dic)[0]
        length = max(queue_dic)+1
        current_index = queue_dic[key][0]
        dist = queue_dic[key][1]

        for i, candidate_vertex in enumerate(adjacency_dic[current_index]):

            if(candidate_vertex == j_indice):
                return dist+1

            queue_dic_temp = {i+length: []}
            queue_dic_temp[i+length].append(candidate_vertex)
            queue_dic_temp[i+length].append(dist+1)
            queue_dic.update(queue_dic_temp)

        queue_dic.pop(key, None)

# --------------------------------


def cost_func(state):

    global distance_list

    n = len(state)
    cost = np.sum([distance_list[i+n*k]
                   for i, k in enumerate(np.array(state))])

    return cost

# --------------------------
# --------------------------


def a_star_algo(current_index, updated_list, adjacency_dic):

    global indicator
    depth = 1
    queue_dic = {i: [] for i in range(len(adjacency_dic[current_index]))}
    hp = []

    visited_list = [updated_list]
    for i, candidate_vertex in enumerate(adjacency_dic[current_index]):

        current_list = update(
            current_index, candidate_vertex, updated_list)
        queue_dic[i].append(cost_func(current_list)+depth)
        queue_dic[i].append(
            update(current_index, candidate_vertex, updated_list))
        queue_dic[i].append(depth)
        queue_dic[i].append(candidate_vertex)
        visited_list.append(current_list)

        if(queue_dic[i][0] == 0):
            indicator = True
            break

    for i in range(len(queue_dic)):
        hp.append((queue_dic[i][0], queue_dic[i][1],
                  queue_dic[i][2], queue_dic[i][3]))
    heapq.heapify(hp)

    while indicator is not True:

        updated_list = hp[0][1]
        current_index = hp[0][3]
        depth = hp[0][2]+1
        heapq.heappop(hp)

        for i, candidate_vertex in enumerate(adjacency_dic[current_index]):

            current_list = update(
                current_index, candidate_vertex, updated_list)
            cost_current = cost_func(current_list)

            if(current_list in visited_list):
                continue
            heapq.heappush(
                hp, (cost_current+depth, current_list, depth, candidate_vertex))
            visited_list.append(current_list)

            if(cost_current == 0):
                indicator = True
                break

    return depth

# --------------------------
# -------------------------


# --- getting vertexs and edges
x = input()
num_vertex, num_edges = [int(i) for i in x.split()]
list_of_key = [i for i in range(num_vertex)]
adjacency_dic = {key: [] for key in list_of_key}

# ----------------------------------------------
# --- getting edges and store it in a dictionary
for _ in range(num_edges):
    x = input()
    v, u = [int(i) for i in x.split()]
    adjacency_dic[v].append(u)
    adjacency_dic[u].append(v)

list_of_current_state = []
ideal_state = [i for i in range(num_vertex)]

# -----------------------------------------------
# -------getting state
x = input()
list_of_current_state = [int(i) for i in x.split()]


depth = 0
current_index = list_of_current_state.index(0)
indicator = False

# -----------------------------------------------
# ---------make list form distance of any vertexs

distance_list = []

power_list = [len(adjacency_dic[k]) for k in range(num_vertex)]

for i in range(num_vertex):
    for j in range(num_vertex):
        distance_list.append(distance_cal(i, j, adjacency_dic))


# ----------------------------------------------
# --------------------printing the answer

if(list_of_current_state == ideal_state):
    print(0)

else:
    answer = a_star_algo(current_index, list_of_current_state,
                         adjacency_dic)
    print(answer)
