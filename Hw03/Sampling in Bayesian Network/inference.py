import pandas as pd
import numpy as np
from functools import reduce
import copy


class BN(object):
    """
    Bayesian Network implementation with sampling methods as a class

    Attributes
    ----------
    n: int
        number of variables

    G: dict
        Network representation as a dictionary. 
        {variable:[[children],[parents]]} # You can represent the network in other ways. This is only a suggestion.

    topological order: list
        topological order of the nodes of the graph

    CPT: list
        CPT Table
    """

    def __init__(self, graph, CPT: dict) -> None:
        ############################################################
        # Initialzie Bayesian Network                              #
        # (1 Points)                                               #
        ############################################################
        self.CPT = CPT
        self.graph = graph

    def cpt(self, node) -> dict:
        """
        This is a function that returns cpt of the given node

        Parameters
        ----------
        node:
            a variable in the bayes' net

        Returns
        -------
        result: dict
            {value1:{{parent1:p1_value1, parent2:p2_value1, ...}: prob1, ...}, value2: ...}
        """
        ############################################################
        # (3 Points)                                               #
        ############################################################

        # Your code
        return self.CPT[node]

    def intersection(self, lst1, lst2):
        return list(set(lst1) & set(lst2))

    def pmf(self, query, evidence) -> float:
        """
        This function gets a variable and its value as query and a list of evidences and returns probability mass function P(Q=q|E=e)

        Parameters
        ----------
        query:
            a variable and its value
            e.g. ('a', 1)
        evidence:
            list of variables and their values
            e.g. [('b', 0), ('c', 1)]

        Returns
        -------
        PMF: float
            P(query|evidence)
        """
        all_variables = [key for key in self.graph]
        query_var = [key for key in query]
        evidence_var = [key for key in evidence]
        not_hidden = self.merge_list(query_var, evidence_var)
        hidden_var = [var for var in all_variables if var not in not_hidden]
        remained_cpts = [self.CPT[df] for df in self.CPT]
        # -------------------------------------
        # eliminate additive states
        for i, df in enumerate(remained_cpts):
            for key in evidence_var:
                df = df[df[key] == evidence[key]].reset_index(
                    drop=True) if key in df else df
                remained_cpts[i] = df

        # ------------------------------------
        # -------merge and marginilize -------

        for var in hidden_var:

            cpt_list = []
            index_list = []
            for i, df in enumerate(remained_cpts):
                if(var in df):
                    cpt_list.append(df)
                    index_list.append(i)

            for i in range(len(index_list)-1, -1, -1):
                remained_cpts.pop(index_list[i])

        #     #--------------------------------
        #     #--------------------------------
            # print(cpt_list)
            merged = cpt_list[0]
            for i, cpt in enumerate(cpt_list):
                if(i >= 1):
                    intersection_var = self.intersection(
                        list(merged), list(cpt))
                    intersection_var.remove('p_value')
                    merged = pd.merge(merged, cpt, on=intersection_var)
                    df_float = merged.select_dtypes(include='float64')
                    merged.drop([key for key in df_float],
                                axis=1, inplace=True)
                    merged['p_value'] = df_float.prod(axis=1)
            merged = merged.groupby([key for key in merged if key not in [var, 'p_value']])[
                'p_value'].sum().reset_index()
            remained_cpts.append(merged)

        # -------------------------------------
        final_joint = remained_cpts[0]
        for i, cpt in enumerate(remained_cpts):
            if(i >= 1):
                intersection_var = self.intersection(
                    list(final_joint), list(cpt))
                intersection_var.remove('p_value')
                final_joint = pd.merge(final_joint, cpt, on=intersection_var)
                df_float = final_joint.select_dtypes(include='float64')
                final_joint.drop([key for key in df_float],
                                 axis=1, inplace=True)
                final_joint['p_value'] = df_float.prod(axis=1)
        margin_list = query_var.copy()
        margin_list.append('p_value')
        # ---------------------------------------
        numerator = final_joint.copy()
        for key in query_var:
            numerator = numerator.loc[numerator[key]
                                      == query[key]].reset_index(drop=True)
        normilize_num = final_joint.groupby([key for key in final_joint if key not in margin_list])[
            'p_value'].sum().reset_index().at[0, 'p_value']
        pmf = numerator.at[0, 'p_value']/normilize_num

        return pmf

        ############################################################
        # (3 Points)                                               #
        ############################################################
    def merge_list(self, list1, list2):
        list_new = list1+list2
        list_new = list(np.unique(np.array(list_new)))
        return list_new

    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin=1e2) -> float:
        """
        Parameters
        ----------
        query: list
            list of variables an their values
            e.g. [('a', 0), ('e', 1)]
        evidence: list
            list of observed variables and their values
            e.g. [('b', 0), ('c', 1)]
        sampling_method:
            "Prior", "Rejection", "Likelihood Weighting", "Gibbs"
        num_iter:
            number of the generated samples 
        num_burnin:
            (used only in gibbs sampling) number of samples that we ignore at the start for gibbs method to converge

        Returns
        -------
        probability: float
            approximate P(query|evidence) calculated by sampling
        """
        ############################################################
        # (27 Points)                                              #
        #     Prior sampling (6 points)                            #
        #     Rejection sampling (6 points)                        #
        #     Likelihood weighting (7 points)                      #
        #     Gibbs sampling (8 points)                      #
        ############################################################

        # Your code
        if(sampling_method == 'Prior'):
            prob = self.prior_sample(query, evidence, num_iter)
            return prob
         # ------------------------------
        if(sampling_method == 'Rejection'):
            prob = self.rejection_sample(query, evidence, num_iter)
            return prob
        # --------------------------------
        if(sampling_method == 'Likelihood Weighting'):
            prob = self.likelihood_sample(query, evidence, num_iter)
            return prob
        # --------------------------------
        if(sampling_method == 'Gibbs'):
            prob = self.gibbs_sample(query, evidence, num_iter, num_burnin)
            return prob

    def prior_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            prior samples
        """
        toplogical_order_list = self.topological_sort()
        prior_samples = []

        # --generate samples
        for num in range(num_iter):
            sample = {key: [] for key in toplogical_order_list}
            for node in toplogical_order_list:
                cpt = self.CPT[node]
                parents = self.graph[node][1]
                condition_dic = {}
                if(len(parents) >= 1):
                    condition_dic = {key: sample[key] for key in parents}
                condition_dic[node] = 1  # find pmf when node is 1
                for condition in condition_dic:
                    cpt = cpt.loc[cpt[condition] ==
                                  condition_dic[condition]].reset_index(drop=True)
                threshold = cpt.at[0, 'p_value']
                random_num = np.random.rand()
                sample[node] = 1 if random_num < threshold else 0
            prior_samples.append(sample)

        # ----------------------------------
        # --calculate probabilty

        num = 0
        denom = 0
        for sample in prior_samples:

            if(self.sample_consistent_with_evidence(sample, evidence)):
                denom += 1

            if(self.sample_consistent_with_evidence(sample, evidence) and self.sample_consistent_with_query(sample, query)):
                num += 1
        pmf = num/denom
        return pmf

    def sample_consistent_with_evidence(self, sample, evidence):
        """
            To check if a sample is consistent with evidence or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                evidence set

            Returns
            -------
            True if the sample is consistent with evidence, False otherwise.
        """
        evidence_check = True
        for condition in evidence:
            if(sample[condition] != evidence[condition]):
                evidence_check = False
                break

        return evidence_check

    def sample_consistent_with_query(self, sample, query):
        """
            To check a sample is consistent with query or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                query set

            Returns
            -------
            True if the sample is consistent with query, False otherwise.
        """
        query_check = True
        for condition in query:
            if(sample[condition] != query[condition]):
                query_check = False
                break

        return query_check

    def rejection_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            rejection samples
        """
        toplogical_order_list = self.topological_sort()
        accepted_samples = []
        rejection_flag = False
        accepted_sample_number = 0

        while(accepted_sample_number < num_iter):

            rejection_flag = False
            sample = {key: [] for key in toplogical_order_list}
            for node in toplogical_order_list:
                cpt = self.CPT[node].copy()
                parents = self.graph[node][1]
                condition_dic = {}
                if(len(parents) >= 1):
                    condition_dic = {key: sample[key] for key in parents}
                condition_dic[node] = 1  # find pmf when node is 1
                for condition in condition_dic:
                    cpt = cpt.loc[cpt[condition] ==
                                  condition_dic[condition]].reset_index(drop=True)
                threshold = cpt.at[0, 'p_value']
                random_num = np.random.rand()
                sample[node] = 1 if random_num < threshold else 0

                # ------reject unconsistent samples
                if(node in evidence):
                    if(sample[node] != evidence[node]):
                        rejection_flag = True
                        break
                # ---------------------------
            if(rejection_flag == False):
                accepted_samples.append(sample)
                accepted_sample_number += 1

        # -------calculate pmf--------
        num = 0
        denom = len(accepted_samples)
        for sample in accepted_samples:

            if(self.sample_consistent_with_query(sample, query)):
                num += 1
        # ---------------------------
        pmf = num/denom
        return pmf

    def likelihood_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            likelihood samples
        """
        all_conditions = query | evidence
        toplogical_order_list = self.topological_sort()
        likelihood_samples = []
        # --generate samples
        for num in range(num_iter):
            sample = {key: [] for key in toplogical_order_list}
            sample['weight'] = 1
            # ----set evidence constant
            for key in evidence:
                sample[key] = evidence[key]
            # ------------------------------
            for node in toplogical_order_list:
                cpt = copy.deepcopy(self.CPT[node])
                parents = self.graph[node][1]
                condition_dic = {}
                if(len(parents) >= 1):
                    condition_dic = {key: sample[key] for key in parents}
                # find pmf when node is 1
                condition_dic[node] = 1 if node not in evidence else evidence[node]
                for condition in condition_dic:
                    cpt = cpt.loc[cpt[condition] ==
                                  condition_dic[condition]].reset_index(drop=True)
                if(node not in evidence):
                    threshold = cpt.at[0, 'p_value']
                    random_num = np.random.rand()
                    sample[node] = 1 if random_num < threshold else 0
                else:
                    sample['weight'] *= cpt.at[0, 'p_value']
            likelihood_samples.append(sample)

        # -------calculate pmf--------
        num = 0
        denom = 0
        condition_check = True
        for sample in likelihood_samples:

            condition_check = True
            for condition in all_conditions:
                if(sample[condition] != all_conditions[condition]):
                    condition_check = False
                    break

            if condition_check:
                num += sample['weight']
            denom += sample['weight']

        # ----------------------------------
        pmf = num/denom
        return pmf

    def condition_pmf(self):

        remained_cpts = [self.CPT[df] for df in self.CPT]
        merged = remained_cpts[0]
        for i, cpt in enumerate(remained_cpts):
            if(i >= 1):
                intersection_var = self.intersection(list(merged), list(cpt))
                intersection_var.remove('p_value')
                merged = pd.merge(merged, cpt, on=intersection_var)
                df_float = merged.select_dtypes(include='float64')
                merged.drop([key for key in df_float], axis=1, inplace=True)
                merged['p_value'] = df_float.prod(axis=1)
        normilize_num = merged['p_value'].sum()
        merged['p_value'] = merged['p_value'].div(normilize_num)
        return merged

    def gibbs_sample(self, query, evidence, num_iter, num_burnin):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of genereted samples

            Returns
            -------
            gibbs samples
        """
        all_variables = [key for key in self.graph]
        all_conditions = query | evidence
        toplogical_order_list = self.topological_sort()
        gibbs_samples = []
        joint_dis = self.condition_pmf()
        # -----initilze sample
        sample = {key: 1 if np.random.rand(
        ) < 0.5 else 0 for key in toplogical_order_list}
        for key in evidence:
            sample[key] = evidence[key]

        # --generate samples ---
        for num in range(num_iter):

            # ------------------------------
            for node in toplogical_order_list:
                if(node in evidence):
                    continue
                cpt = copy.deepcopy(joint_dis)
                condition_dic = {}
                condition_dic = {key: sample[key]
                                 for key in all_variables if key != node}
                for condition in condition_dic:
                    cpt = cpt.loc[cpt[condition] ==
                                  condition_dic[condition]].reset_index(drop=True)
                # ----------------------------
                normilize_num = cpt['p_value'].sum()
                cpt['p_value'] = cpt['p_value'].div(normilize_num)
                cpt = cpt.loc[cpt[node] == 1].reset_index(drop=True)
                # --------------------------------------
                threshold = cpt.at[0, 'p_value']
                random_num = np.random.rand()
                sample[node] = 1 if random_num < threshold else 0
            copy_sample = sample.copy()
            gibbs_samples.append(copy_sample)
        # ---------------------------------
        filter_sample = gibbs_samples[num_burnin:] if num_burnin > len(
            gibbs_samples) else gibbs_samples[int(num_burnin/2):]

        # -------calculate probabilty--------
        num = 0
        condition_check = True
        for sample in filter_sample:
            condition_check = True

            for condition in all_conditions:
                if(sample[condition] != all_conditions[condition]):
                    condition_check = False
                    break

            if condition_check:
                num += 1

        pmf = num/len(filter_sample)
        return pmf

    def remove_edge(self, element, graph: dict):

        new_dic = copy.deepcopy(graph)
        for key in graph:
            if element in new_dic[key][1]:
                new_dic[key][1].remove(element)
        return new_dic

    def topological_sort(self):
        """
            This function wants to make a topological sort of the graph and set the topological_order parameter of the class.

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

        """
        node = [key for key in self.graph]
        size = len(node)
        stack = []
        temp_graph = copy.deepcopy(self.graph)
        c = 0
        while(c != size):
            for key in node:
                if(len(temp_graph[key][1]) == 0 and key not in stack):
                    stack.append(key)
                    temp_graph = self.remove_edge(key, temp_graph)
                    c += 1

        return stack

    def set_topological_order(self):
        """
            This function calls topological sort function and set the topological sort.
        """
        pass

    def all_parents_visited(self, node, visited) -> bool:
        """
            This function checks if all parents are visited or not?

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

            Return
            ----------
            return True if all parents of node are visited, False otherwise.
        """
        pass

    # no need to following functions.
    def remove_nonmatching_evidences(self, evidence, factors):
        pass

    def join_and_eliminate(self, var, factors, evidence):
        pass

    def get_joined_factor(self, var_factors, var, evidence):
        pass

    def get_rows_factor(self, factor, var, evidence, values, variables_in_joined_factor):
        pass

    def get_var_factors(self, var, factors):
        pass

    def get_variables_in_joined_factor(self, var_factors, var, evidence):
        pass

    def get_join_all_factors(self, factors, query, evidence):
        pass

    def get_row_factor(self, factor, query_vars, evidence, values):
        pass

    def normalize(self, joint_factor):
        pass
