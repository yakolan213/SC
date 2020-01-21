import operator
import math
import random

class Models(object):
    def KP(self, group_b):
        """
        :param group:(list) the winners group (names)
        :return: the best candidate by KP model
        """
        return self.find_best_candidate_from_group(group_b)

    def find_best_candidate_from_group(self, group):
        """
        :param group:(list of dictionary) the winners group (names)
        :return: the candidate from the winning group that maximaize the voter utility
        """
        utilities_group = {}
        for c in group:
            utilities_group.update(c)
        max_utility = max(utilities_group.values())
        max_keys = [k for k, v in utilities_group.items() if v == max_utility]
        return max_keys

    def CV(self, P_dict, utilities_dict):
        """
        :param P_dict: (dict) key: (candidate x, candidate y), values: the probability the voter is pivotal for candidate x over candidate y
        :param utilities_dict: (dict) key: candidate, value: the voter utility if the candidate wins
        :return: the best candidate by CV model
        """
        f = lambda c1, c2, p: p * (utilities_dict[c1] - utilities_dict[c2])
        sum = 0
        best_candidate = None

        for candidate_1 in utilities_dict.keys():
            sum_temp = 0
            for candidate_2 in utilities_dict.keys():
                if candidate_1 != candidate_2:
                    p = P_dict[(candidate_1, candidate_2)]
                    sum_temp += f(candidate_1, candidate_2, p)
            if sum_temp > sum:
                sum = sum_temp
                best_candidate = candidate_1
        return best_candidate

    def calculate_n(self, s):
        """
        :param s: (dict) key : candidate name , value: the votes the candidate have
        :return: number of total votes
        """
        votes_count = 0
        for candidate, votes in s.items():
            votes_count += votes
        return votes_count

    def find_U_group(self, r, s):
        """
        :param r:(int) random parameter
        :param s: (dict) key : candidate name , value: the votes the candidate have
        :return: U group (all candidates that there votes are max(s) - 2 * r * n
        """
        n = self.calculate_n(s)
        U_group = []
        max_c = max(s.items(), key=operator.itemgetter(1))[0]
        max_s = s[max_c]
        for candidate, score in s.items():
            if score >= max_s - 2 * r * n:
                U_group.append({candidate: score})
        return U_group

    def LD(self, r, s, utilities_dict):
        """
        :param r:(int) random parameter
        :param s: (dict) key : candidate name , value: the votes the candidate have
        :param utilities_dict: (dict) keys: candidate, value: the voter utility if the candidate wins
        :return: the best candidate by LD model
        """
        U_group = self.find_U_group(r, s)
        return self.find_best_candidate_from_group(utilities_dict, U_group)

    def AT(self, utilities_dict, b_dict, s):
        """
        :param utilities_dict: (dict) key: candidate, value: the voter utility if the candidate wins
        :param b_dict: (dict) key: candidate, value: candidates b parameter
        :param s: (dict) key : candidate name , value: the votes the candidate have
        :return:  the best candidate by AT model
        """
        score_dict = {}
        f = lambda A, c: A * utilities_dict[c]
        for candidate, utility in utilities_dict.items():
            b = b_dict[candidate]
            A = self.calc_A(b, s, candidate)
            score_dict[candidate] = f(A, candidate)
        best_candidate = max(score_dict.items(), key=operator.itemgetter(1))[0]
        return best_candidate

    def calc_A(self, b, s, candidate):
        """
        :param b: (int) uniqe parameter to each voter
        :param s: (dict) key : candidate name , value: the votes the candidate have
        :param candidate: the specific candidate
        :return: calculated A
        """
        f = lambda b, sj: (1/math.pi) * math.atan(b*(sj - 0.5)) + 0.5
        return f(b, s[candidate])

    def AU(self, utilities_dict, e, a, b_dict, s):
        """
        :param utilities_dict: (dict) key: candidate, value: the voter utility if the candidate wins
        :param e: (int) error parameter (for 0 utility)
        :param a_dict:
        :param b_dict:
        :param s: (dict) key : candidate name , value: the votes the candidate have
        :return:  the best candidate by AU model
        """
        score_dict = {}
        f = lambda A, c, e, a: (math.pow(A, 2 - a)) * (math.pow(utilities_dict[c] + e, a))
        for candidate, utility in utilities_dict.items():
            b = b_dict[candidate]
            A = self.calc_A(b, s, candidate)
            score_dict[candidate] = f(A, candidate, e, a)
        best_candidate = max(score_dict.items(), key=operator.itemgetter(1))[0]
        return best_candidate

    def pivotal_p(self, psai, voters_list):
        """
        :param psai:(int) how many voters the user wants to calculate the probabilities from
        :param voters_list: (list) the voters list
        :return: the pivotal probabilities to all candidates
        """
        p_pivotal = {}
        psai_origin = psai
        n = len(voters_list)
        k = 1
        if psai > n:
            k = int(psai/n) + 1
        for j in range(k):
            vouters_count = 0
            voters = voters_list.copy()
            if psai >= n:
                current_n = n
            else:
                current_n = psai - int(psai/n) * n
            while vouters_count < current_n:
                vouters_count += 1
                voter = random.choice(voters)
                voters.remove(voter)
                s_voter = voter.s
                for candidate_1, score_1 in s_voter.items():
                    for candidate_2, score_2 in s_voter.items():
                        if candidate_1 == candidate_2:
                            continue
                        if (candidate_1, candidate_2) not in p_pivotal:
                            p_pivotal[(candidate_1, candidate_2)] = 0
                        if score_1 == score_2:
                            p_pivotal[(candidate_1, candidate_2)] += 1
            psai -= n
        p_pivotal = {k: v / psai_origin for k, v in p_pivotal.items()}
        return p_pivotal


