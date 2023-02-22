import numpy as np
import itertools


class CSP:

    def __init__(self, number_of_marks):
        """
        Here we initialize all the required variables for the CSP computation,
        according to the number of marks.
        """
        # # Your code here
        if(number_of_marks == 1):
            self.current_length = 0
            self.result = True
            self.variables = [0]
        else:
            self.counter = 0
            self.result = False
            self.number_of_marks = number_of_marks
            self.current_length = int((number_of_marks-1) *
                                      (number_of_marks)/2)  # Update this line
            self.variables = [-1 for _ in range(self.number_of_marks)]
            self.variables[0] = 0
            self.differences = [[0]
                                for _ in range(number_of_marks)]  # Update this line
            self.available_val = {key: [k for k in range(
                1, self.current_length+1)] for key in range(self.number_of_marks)}

    def assign_value(self, i, v):
        """
        assign a value v to variable with index i
        """
        # Your code here
        self.variables[i] = v

    def check_constraints(self, index) -> bool:
        """
        Here we loop over the differences array and update values.
        Meanwhile, we check for the validity of the constraints.
        index : the last variable that get value 
        """
        bool_var = True
        self.differences[index:] = [[0]
                                    for _ in range(self.number_of_marks-index)]
        flatten_diff = list(itertools.chain(*self.differences[:index]))
        # Your code here

        for i in range(index):

            differnce1 = self.variables[index] - self.variables[i]

            if differnce1 not in flatten_diff:

                self.differences[index].append(differnce1)
            else:
                bool_var = False
                self.differences[index] = [0]
                break
        return bool_var

    def backtrack(self, i):
        """
         In this function we should loop over all the available values for
         the variable with index i, and recursively check for other variables values.
        """
        # print(self.variables)
        if(self.number_of_marks == 0):
            return self.result

        for v in self.available_val[i]:

            if(i == 1 and v != 1):
                break
            self.assign_value(i, v)
            check_mark = self.check_constraints(i)

            if(check_mark and i < self.number_of_marks-1):
                self.forward_check(i)
                self.backtrack(i+1)

            if(check_mark and i == self.number_of_marks-1):
                self.result = True
            if(self.result):
                break
        return self.result

    def forward_check(self, i):
        """
        After assigning a value to variable i, we can make a forward check - if needed -
        to boost up the computing speed and prune our search tree.
        """
        # print(self.variables)
        domain_list = [var for var in range(self.current_length+1)]
        # Your code here
        filtered1 = np.array(
            [self.variables[k]+np.array(list(itertools.chain(*self.differences[:i]))) for k in range(i+1)]).flatten()
        filtered1 = np.unique(filtered1)
        # -------------------------------
        filtered2 = np.array(
            [self.variables[k]-np.array(list(itertools.chain(*self.differences[:i]))) for k in range(i+1)]).flatten()
        filtered2 = np.unique(filtered2)

        # ------------------------------
        total_not_elgibile = np.union1d(filtered1, filtered2)
        for c in range(i+1, self.current_length):
            self.available_val[c] = [
                element for element in domain_list if (element not in total_not_elgibile and element > self.variables[i])]
        # print(self.available_val)

    def find_minimum_length(self) -> int:
        """
        This is the main function of the class.
        First, we start by assigning an upper bound value to variable current_length.
        Then, using backtrack and forward_check functions, we decrease this value until we find
        the minimum required length.
        """
        counter = 0
        while self.result is False and self.current_length < 2**(self.number_of_marks):

            counter += 1
            self.result = self.backtrack(1)
            # print(self.current_length)
            # print(self.available_val)

            if(self.result):
                break
            else:

                self.__init__(self.number_of_marks)
                self.current_length += counter
                self.available_val = {key: [k for k in range(
                    1, self.current_length+1)] for key in range(self.number_of_marks)}

        return self.current_length

    def get_variables(self) -> list:
        """
        Get variables array.
        """
        # No need to change
        return self.variables
