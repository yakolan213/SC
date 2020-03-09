from data import Models
import random

class Voter(object):
    def __init__(self, id,
                 preference_list,
                 utilities_dict,
                 vote,
                 s):
        """
        :param id: the voter id
        :param preference_list: (list) the voter preference, the candidate he prefer the most will be in index 0 and so on
        :param utilities_dict: (dict) key: candidate, value: the voter utility if the candidate wins
        :param vote: (str) the candidate name the voters votes for
        """
        self.id = id
        self.preference_list = preference_list
        self.utilities_dict = utilities_dict
        self.vote = vote
        self.s = s
        self.models = Models()

    def get_CV_parameter(self, voters_list, division_param, l_bound, u_bound):
        """
        :param voters_list: (list) the voters list
        :param division_param: (int) how many b values the user wants to check
        :param l_bound: (int) the lower bound for the parameter
        :param u_bound: (int) the upper bound for the parameter
        :return: the pivotal p
        """
        model_values_dict = {}
        for i in range(1, division_param + 1):
            psai = l_bound + int(i * (u_bound - l_bound) / division_param)
            pivotal_dict = self.models.pivotal_p(psai, voters_list)
            model_value = self.models.CV(pivotal_dict, self.utilities_dict)
            model_values_dict[psai] = model_value
        return model_values_dict, self.s

    def get_KP_parameter(self, k):
        """
        :param k: (int) how big is the winners group
        :return: predication by KP model for each size
        """
        model_values_dict = {}
        if k not in [1,2,3]:
            raise RuntimeError('k can be only 1, 2 or 3 and not {0}'.format(k))
        candidate_score = self.s
        candidate_list = [(k, v) for k, v in candidate_score.items()]
        candidate_list.sort(key=lambda tup: tup[1])
        for i in range(k):
            group_k = candidate_list[:k]
            group_k_list_of_dict = {}
            for tup in group_k:
                group_k_list_of_dict[tup[0]] =tup [1]
            model_value = self.models.KP(group_k_list_of_dict, self.utilities_dict)
            model_values_dict[i + 1] = model_value
        return model_values_dict, self.s

    def get_AT_parameter(self, division_param, l_bound, u_bound):
        """
        :param division_param: (int) how many b values the user wants
        :param l_bound: (int) the lower bound for the parameter
        :param u_bound: (int) the upper bound for the parameter
        :return: all AT predictions to all b values
        """
        model_values_dict = {}
        candidates = list(self.s.keys())
        for i_1 in range(1, division_param + 1):
            b_1 = i_1 * (u_bound - l_bound)/ division_param
            for i_2 in range(1, division_param + 1):
                b_2 = i_2 * (u_bound - l_bound) / division_param
                for i_3 in range(1, division_param + 1):
                    b_3 = i_3 * (u_bound - l_bound) / division_param
                    b_dict = {candidates[0]: b_1, candidates[1]: b_2, candidates[2] : b_3}
                    model_value = self.models.AT(self.utilities_dict, b_dict, self.s,)
                    b_list = tuple(b_dict.items())
                    model_values_dict[b_list] = model_value
        return model_values_dict, self.s


    def get_AU_parameters(self, division_param_b, l_bound_b, u_bound_b, e, division_param_a, l_bound_a = 0, u_bound_a = 2):
        """
        :param division_param_b: (int) how many b values the user wants
        :param division_param_a: (int) how many a values the user wants
        :param l_bound_b: (int) the lower bound for the parameter b
        :param u_bound_b: (int) the upper bound for the parameter b
        :param l_bound_a: (int) the lower bound for the parameter a, if none l_bound_a = 0
        :param u_bound_a: (int) the upper bound for the parameter a if none u_bound_a = 2
        :param e: (float) the epsilon
        :return: all AU predictions to all b and a values
        """

        model_values_dict = {}
        candidates = list(self.s.keys())
        for i_1 in range(1, division_param_b + 1):
            b_1 = round(i_1 * (u_bound_b - l_bound_b) / division_param_b,3)
            for i_2 in range(1, division_param_b + 1):
                b_2 = round(i_2 * (u_bound_b - l_bound_b) / division_param_b,3)
                for i_3 in range(1, division_param_b + 1):
                    b_3 = round(i_3 * (u_bound_b - l_bound_b) / division_param_b,3)
                    b_dict = {candidates[0]: b_1, candidates[1]: b_2, candidates[2]: b_3}
                    for j in range(1, division_param_a + 1):
                        a = round(j * (u_bound_a - l_bound_a) / division_param_a,3)
                        model_value = self.models.AU(self.utilities_dict, e, a, b_dict, self.s,)
                        params_dict = b_dict.copy()
                        params_dict.update({'a': a})
                        params_list = tuple(params_dict.items())
                        model_values_dict[params_list] = model_value
        return model_values_dict, self.s

    def get_LD_parameter(self, division_param, l_bound, u_bound):
        """
        :param division_param: (int) how many r values the user wants
        :param l_bound: (int) the lower bound for the parameter
        :param u_bound: (int) the upper bound for the parameter
        :return: all LD predictions to all r values
        """
        model_values_dict = {}
        for i in range(1, division_param + 1):
            r = i * (u_bound - l_bound)/ division_param #can change it to np.linspace
            model_value = self.models.LD(r, self.s, self.utilities_dict)
            model_values_dict[r] = model_value
        return model_values_dict, self.s






