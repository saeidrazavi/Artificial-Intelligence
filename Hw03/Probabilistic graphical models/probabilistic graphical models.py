# import required libraries
import numpy as np

# -------------------------
# -------------------------


def dfs(path: list, direction_list: list, visited_list: list, current_node):

    global independecy
    global active_path

    for node, direction in adj_dic[current_node]:

        if(independecy == False):
            break

        if(node in visited_list):
            continue

        else:
            visited_list_temp = visited_list.copy()
            visited_list_temp.append(node)
            temp_path = path.copy()
            temp_path.append(node)
            direction_list_temp = direction_list.copy()
            direction_list_temp.append(direction)

        if(node == end_node):

            answer = is_inactive(temp_path, direction_list_temp)
            if(answer == False):
                independecy = False
                active_path = temp_path.copy()
                break

        dfs(temp_path, direction_list_temp, visited_list_temp, node)

    if(independecy == False):
        return active_path
    else:
        return 'independent'

# -------------------------
# -------------------------


def adjacency_dic(adj_list: list, node_numbers) -> dict:

    adj_dic = {key: [] for key in range(1, node_numbers+1, 1)}

    for path in adj_list:

        adj_dic[path[0]].append((path[1], 0))
        adj_dic[path[1]].append((path[0], -1))

    return adj_dic

# -----------------------------------
# -----------------------------------


def triple_type(triple: list) -> str:

    if(triple[0] == 0 and triple[1] == 0):
        return "causal"

    if(triple[0] == -1 and triple[1] == -1):
        return "causal"

    if(triple[0] == -1 and triple[1] == 0):
        return "common"

    if(triple[0] == 0 and triple[1] == -1):
        return "v_structure"

# ----------------------------------
# ----------------------------------


def descendent_finder(current_node, visited_list: list) -> list:

    for node, direction in adj_dic[current_node]:

        if(node in visited_list):
            continue

        if(node not in visited_list and direction == 0):
            visited_list.append(node)
            visited_list = descendent_finder(node, visited_list)

    return visited_list


def is_inactive(path: list, direction_list: list):

    answer = False
    for i in range(len(direction_list)-1):
        direction = direction_list[i:i+2]
        triple_path = path[i:i+3]
        type_ = triple_type(direction)

        if(type_ == 'causal' and triple_path[1] in observed_nodes):
            answer = True
            return answer

        if(type_ == 'common' and triple_path[1] in observed_nodes):
            answer = True
            return answer

        if(type_ == 'v_structure'):
            descendent_list = descendent_finder(
                triple_path[1], [triple_path[1]])
            if(len(set(observed_nodes)-set(descendent_list)) == observed_num):
                answer = True
                return answer

    return answer
# --------------------------


def print_ans(array_):
    for i, x in enumerate(array_):
        if(i != len(array_)-1):
            print(x, end=", ")
        else:
            print(x, end="")


# ----------------------------
node_num, edge_num, observed_num = np.array(input().split(), dtype=int)
adjacency_list = []
for _ in range(edge_num):

    path = np.array(input().split(), dtype=int)
    adjacency_list.append(path)
# -----------------------
observed_nodes = []
for _ in range(observed_num):
    observed_nodes .append(int(input()))
start_node, end_node = np.array(input().split(), dtype=int)
adj_dic = adjacency_dic(adjacency_list, node_num)
independecy = True
active_path = []


# ------------------------------
# --------print answer----------
# ------------------------------

if(start_node != end_node):
    final_ans = dfs([start_node], direction_list=[], visited_list=[
        start_node], current_node=start_node)
    if(type(final_ans) == str):
        print(final_ans)
    else:
        print_ans(final_ans)
else:
    print(start_node)
