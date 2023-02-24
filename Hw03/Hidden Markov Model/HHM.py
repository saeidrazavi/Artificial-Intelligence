# hidden markov model implementation
import numpy as np


def viterbi(depth):

    # --------------------------------
    cost_matrix = np.zeros((hidden_var_num, depth))
    index_matrix = np.zeros((hidden_var_num, depth))
    cost_matrix[:, 0] = np.multiply(
        initial_hidden_prob, hidden_observed_prob_matrix[:, observed_list[0]-1])

    for i in range(depth):

        if(i >= 1):
            for current_col in range(hidden_var_num):
                prob_list = []
                for prev_col in range(hidden_var_num):
                    prob = 1 * \
                        hidden_prob_matrix[prev_col,
                                           current_col]
                    prob *= hidden_observed_prob_matrix[current_col,
                                                        observed_list[i]-1]

                    prob *= cost_matrix[prev_col, i-1]

                    prob_list.append(prob)
                cost_matrix[current_col, i] = max(prob_list)
                index_matrix[current_col, i] = np.argmax(
                    np.array(prob_list))

    # ------------------------------------

    max_prob_index = np.argmax(cost_matrix[:, depth-1])
    max_probe_state = [max_prob_index]

    for i in range(depth-1, 0, -1):
        max_probe_state.append(int(index_matrix[max_prob_index, i]))
        max_prob_index = int(index_matrix[max_prob_index, i])

    return max_probe_state, np.max(cost_matrix[:, depth-1])


def predict(last_state, occurance_num):

    best_prob = 0
    best_color = 0
    for color in range(observed_var_num):
        prob_temp = 0
        for next_state in range(hidden_var_num):
            prob = 1*hidden_observed_prob_matrix[next_state,
                                                 color]*hidden_prob_matrix[last_state, next_state] if occurance_num != 0 else 1*hidden_observed_prob_matrix[next_state,
                                                                                                                                                            color]
            prob_temp += prob
        best_color = color if prob_temp > best_prob else best_color
        best_prob = prob_temp if prob_temp > best_prob else best_prob

    return best_color, best_prob


def decimal_length(number):
    length = str(number)[::-1].find('.')
    return length


# ---------------------
# -----get informations
occurrance_num = int(input())
hidden_var_num = int(input())
observed_var_num = int(input())

observed_list = np.array(
    input().split(), dtype=int) if occurrance_num != 0 else 0
initial_hidden_prob = np.array(input().split(), dtype=float)

# ---make hidden variables matrix
hidden_prob_matrix = np.zeros((hidden_var_num, hidden_var_num), dtype=float)
for i in range(hidden_var_num):
    hidden_prob_matrix[i, :] = np.array(input().split(), dtype=float)

# ---make observed_hidden probabilty matrix
hidden_observed_prob_matrix = np.zeros(
    (hidden_var_num, observed_var_num), dtype=float)
for i in range(hidden_var_num):
    hidden_observed_prob_matrix[i, :] = np.array(input().split(), dtype=float)


ans, prob1 = viterbi(occurrance_num)
best_color, prob2 = predict(ans[0], occurrance_num)
print(ans[0])
print(prob1)
print(prob2)
# -------------------------------
prob = round(prob2, 2) if decimal_length(prob2) >= 2 else prob2
print(best_color+1, prob)
