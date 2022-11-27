import numpy as np
import math
import pandas as pd
from tqdm import tqdm


class SimulatedAnnealing:

    """
    NOTE:
        - S is the set of members.
        - T is the target value.
        - Chromosomes are represented as an array of 0 and 1 with the same length as the set.
        (0 means the member is not included in the subset, 1 means the member is included in the subset)

        Feel free to add any other function you need.
    """

    def __init__(self):
        pass

    def random_state_generator(self, n: int) -> np.ndarray:
        """
        Generate initial state: This function is used to generate the initial state.

        Inputs:
        - n: number of members in S

        Outputs:
        - initial state
        """
        random_sequence = np.random.randint(2, size=n)

        return random_sequence

    def is_feasible(self, state: np.ndarray, S: np.ndarray, T: int) -> bool:
        """
        This function is used to check if the sum of the subset is equal or less to the target value.

        Inputs:
        - state: state to be evaluated (subset)
        - S: set of members
        - T: target value

        Outputs:
        - True (1) if the sum of the state is equal or less to the target value, False (0) otherwise
        """
        sum = np.sum(np.dot(np.array(state), np.array(S)))
        result = True if sum < T else False
        return result

    def fliping(self, state: np.ndarray, n: int) -> np.ndarray:

        copy_of_seq = np.copy(state)
        random_possition = np.random.randint(n)
        copy_of_seq[random_possition] = 1 if copy_of_seq[random_possition] == 0 else 0

        return copy_of_seq

    def neighbor_state_generator(self, state: np.ndarray, S: np.ndarray, T: int, prob: float = 0.5) -> np.ndarray:
        """
        Generate neighbor state: This function is used to generate the neighbor state.

        Inputs:
        - state: current state
        - prob: probability of flipping the second bit (described below)

        The neighbor state is generated in the following way:
        - Copy the current state
        - Get a random number between 0 and the length of the state
        - Flip the bit at the random position
        - Generate a random number between 0 and 1
        - If the random number is less than or equal to prob, 
            get another random number between 0 and the length of the state and flip the bit at the random position
        - If the neighbor state is feasible, return it. Else do the previous stages until the neighbor state is feasible

        Outputs:
        - neighbor state
        """

        n = len(state)
        feasibilty = False
        copy_of_seq = np.copy(state)

        while(feasibilty is False):
            copy_of_seq = self.fliping(copy_of_seq, n)
            random_prob = np.random.rand(1)
            copy_of_seq = self.fliping(
                copy_of_seq, n) if random_prob < prob else copy_of_seq
            feasibilty = True if self.is_feasible(copy_of_seq, S, T) else False

        return copy_of_seq

    def cost_function(self, state: np.ndarray, S: np.ndarray, T: int) -> int:
        """
        Cost function: This function is used to calculate the cost of a certain state.

        Inputs:
        - state: state to be evaluated
        - S: set of members
        - T: target value

        It calculates the absolute difference between the sum of the members included in the subset
            (i.e. sum of S[i]s where state[i] == 1) and the target value.

        Outputs:
        - cost of the state
        """
        cost = np.abs(np.dot(state, S)-T)

        return cost

    def prob_accept(self, state_cost: int, next_state_cost: int, temperature: float) -> float:
        """
        Probability of accepting the next state: This function is used to calculate the probability of accepting the next state.

        Inputs:
        - state_cost: cost of the current state
        - next_state_cost: cost of the next state
        - temperature: current temperature

        It calculates the probability of accepting the next state using the following formula:
            P = exp(- ((next_state_cost - state_cost) ** 1.5) / temperature)

        NOTE: Feel free to change the formula if you want to. (You can also consider other variables in determining the probability of accepting the next state.)

        Outputs:
        - probability of accepting the next state
        """
        P = math.exp(- ((next_state_cost - state_cost) ** 1.5) / temperature)

        return float(P)

    def run_algorithm(self, S: np.ndarray, T: int, neigbor_prob: float = 0.5, stopping_iter: int = 3000, temperature: float = 30):
        """
        Run algorithm: This function is used to run the simulated annealing algorithm.

        Inputs:
        - S: set of members
        - T: target value
        - neigbor_prob: probability of flipping the second bit (described in neighbor_state_generator)
        - stopping_iter: number of iterations to stop the algorithm
        - temperature: initial temperature

        It runs the simulated annealing in the following way:
        - Generate the initial state and set the best state to the initial state and the best cost to the cost of the initial state
        - Calculate the decay rate in the following way: decay_rate = temperature / stopping_iter
        - For each iteration:
        -   Generate the neighbor state
        -   Calculate the cost of the current state and the neighbor state
        -   If the cost of the neighbor state is less than or equal to the cost of the current state:
                set the current state to the neighbor state and the best state to the neighbor state and the best cost to the cost of the neighbor state
        -   Else:
        -       Generate a random number between 0 and 1
        -       If the random number is less than or equal to the probability of accepting the next state, set the current state to the neighbor state
        - Decrease the temperature by its decay rate
        - Append the current best cost and current best solution to the records list
        - Return the best cost, the best solution, and the records


        Outputs:
        - best cost
        - best solution
        - records
        """
        # UPDATE THESE VARIABLES (best_cost, best_solution, records)
        best_cost = np.Inf
        best_solution = None
        records = []

        n = len(S)
        initial_state = self.random_state_generator(n)
        current_state = np.copy(initial_state)
        best_solution = np.copy(initial_state)
        best_cost = np.copy(self.cost_function(best_solution, S, T))

        # compute decay rate
        decay_rate = temperature / stopping_iter

        for i in tqdm(range(stopping_iter)):

            neighbor_state = self.neighbor_state_generator(
                current_state, S, T, prob=0.5)
            # ------------------
            cost_of_current = self.cost_function(current_state, S, T)
            cost_of_next = self.cost_function(neighbor_state, S, T)

            iter_prob = self.prob_accept(
                cost_of_current, cost_of_next, temperature)

            if(cost_of_next < cost_of_current):
                best_solution = np.copy(neighbor_state)
                current_state = np.copy(neighbor_state)
                best_cost = np.copy(cost_of_next)

            else:
                random_number = np.random.rand(1)
                best_solution = neighbor_state if random_number <= iter_prob else best_solution
                current_state = neighbor_state if random_number <= iter_prob else current_state
                best_cost = cost_of_next if random_number <= iter_prob else best_cost

            # YOUR CODE HERE
            temperature -= decay_rate

            records.append({'iteration': i, 'best_cost': best_cost,
                           'best_solution': best_solution})  # DO NOT REMOVE THIS LINE

        records = pd.DataFrame(records)  # DO NOT REMOVE THIS LINE
        return best_cost, best_solution, records
