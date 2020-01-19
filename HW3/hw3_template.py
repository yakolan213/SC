# Imports: You may use these packages freely. Contact me if you intend on using other packages.
import pandas as pd
import csv
from collections import defaultdict
from collections import OrderedDict
import math
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from statistics import median
import copy

# Make sure you are reading the data file either like this:
#data = pd.read_csv('voters.csv', header=None)
# Or like this:
#with open('votes.csv') as file:
#    reader = csv.reader(file, delimiter=',')
#    for line in reader:
#        pass


def DTD_borda(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    votes.index = range(num_of_unique_voters)
    scores = {}
    for key in nb:
        scores[key] = 0
    weights = [1] * num_of_unique_voters
    for row in range(0, num_of_unique_voters, 1):
        buildings = num_of_buildings - 1
        array = votes.loc[[row]]
        for element in array:
            scores[int(array[element])] += buildings
            buildings += -1
    pref = reorder_by_rank(list(scores.values()),nb)
    pref = list(map(int, pref))
    return_f = []
    weights = []
    errors = list(range(0, num_of_unique_voters, 1))
    votes_list = votes.values.tolist()
    for f in errors:
        fi = kendall_tau(pref, votes_list[f])
        fi = fi / (num_of_buildings * (num_of_buildings - 1) / 2)
        return_f.append(fi)
        if(fi<0.01):
            fi = 0.01
        if(fi>0.99):
            fi = 0.99
        wi = math.log10((1 - fi) / fi)
        weights.append(wi)
        for key in nb:
            scores[key] = 0
    for row in range(0, num_of_unique_voters, 1):
        buildings = num_of_buildings - 1
        array = votes.loc[[row]]
        for element in array:
            scores[int(array[element])] += buildings*weights[row]
            buildings += -1
    pref = reorder_by_rank(list(scores.values()),nb)
    pref = list(map(int, pref))
    return pref,return_f

def DTD_Copeland(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    votes.index = range(num_of_unique_voters)
    weights = [1]*num_of_unique_voters
    mat = calculate_pairwise_matrix(votes,weights,nb)
    pref = reorder_by_rank(list(mat.sum(axis=1)),nb)
    pref = list(map(int, pref))
    return_f = []
    weights = []
    errors = list(range(0,num_of_unique_voters,1))
    votes_list = votes.values.tolist()
    for f in errors:
        fi = kendall_tau(pref,votes_list[f])
        fi = fi / (num_of_buildings * (num_of_buildings-1)/2)
        return_f.append(fi)
        if(fi<0.01):
            fi = 0.01
        if(fi>0.99):
            fi = 0.99
        wi = math.log10((1 - fi) / fi)
        weights.append(wi)
    mat = calculate_pairwise_matrix(votes,weights,nb)
    pref = reorder_by_rank(list(mat.sum(axis=1)),nb)
    pref = list(map(int, pref))
    return pref,return_f

def PTD_borda(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    votes.index = range(num_of_unique_voters)
    weights = []
    return_f = []
    votes_list = votes.values.tolist()
    for row in votes_list:
        pi = Estimate_Error(row, votes_list)
        fi = pi
        fi = fi / (num_of_buildings * (num_of_buildings - 1) / 2)
        return_f.append(fi)
        if(fi<0.01):
            fi = 0.01
        if(fi>0.99):
            fi = 0.99
        wi = math.log10((1 - fi) / fi)
        weights.append(wi)
    scores = {}
    for key in nb:
        scores[key] = 0
    for row in range(0, num_of_unique_voters, 1):
        buildings = num_of_buildings - 1
        array = votes.loc[[row]]
        for element in array:
            scores[int(array[element])] += buildings*weights[row]
            buildings += -1
    pref = reorder_by_rank(list(scores.values()),nb)
    pref = list(map(int, pref))
    return pref, return_f

def PTD_Copeland(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    weights = []
    return_f = []
    votes_list = votes.values.tolist()
    for row in votes_list:
        pi = Estimate_Error(row, votes_list)
        fi = pi
        fi = fi / (num_of_buildings * (num_of_buildings - 1) / 2)
        return_f.append(fi)
        if(fi<0.01):
            fi = 0.01
        if(fi>0.99):
            fi = 0.99
        wi = math.log10((1 - fi) / fi)
        weights.append(wi)
    mat = calculate_pairwise_matrix(votes, weights,nb)
    pref = reorder_by_rank(list(mat.sum(axis=1)),nb)
    pref = list(map(int, pref))
    return pref, return_f

def UW_borda(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    votes.index = range(num_of_unique_voters)
    scores = {}
    for key in nb:
        scores[key] = 0
    weights = [1] * num_of_unique_voters
    for row in range(0, num_of_unique_voters, 1):
        buildings = num_of_buildings - 1
        array = votes.loc[[row]]
        for element in array:
            scores[int(array[element])] += buildings
            buildings += -1
    pref = reorder_by_rank(list(scores.values()),nb)
    pref = list(map(int, pref))
    return_f = []
    errors = list(range(0, num_of_unique_voters, 1))
    votes_list = votes.values.tolist()
    for f in errors:
        fi = kendall_tau(pref, votes_list[f])
        fi = fi / (num_of_buildings * (num_of_buildings - 1) / 2)
        return_f.append(fi)
    return pref, return_f

def UW_Copeland(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    weights = [1] * num_of_unique_voters
    mat = calculate_pairwise_matrix(votes, weights,nb)
    pref = reorder_by_rank(list(mat.sum(axis=1)),nb)
    pref = list(map(int, pref))
    return_f = []
    errors = list(range(0, num_of_unique_voters, 1))
    votes_list = votes.values.tolist()
    for f in errors:
        fi = kendall_tau(pref, votes_list[f])
        fi = fi / (num_of_buildings * (num_of_buildings - 1) / 2)
        return_f.append(fi)
    return pref, return_f

def your_algorithm(votes):
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    num_of_buildings = len(pd.DataFrame.count(votes, axis=0))
    pref_c, fault_c = UW_Copeland(votes)
    pref_b, fault_b = UW_borda(votes)
    fault_combined = []
    for i in range(len(fault_c)):
        fault_combined.append(float(fault_b[i]+fault_c[i])/2)
    return_f = fault_combined.copy()
    med = median(fault_combined)
    for i in range(len(fault_combined)):
        if (fault_combined[i]>med):
            fault_combined[i] = 0
        else:
            fault_combined[i] *= 2
    nb_values = votes.values.tolist()
    nb = nb_values[0].copy()
    nb.sort()
    scores = {}
    for key in nb:
        scores[key] = 0
    for row in range(0, num_of_unique_voters, 1):
        buildings = num_of_buildings - 1
        array = votes.loc[[row]]
        for element in array:
            scores[int(array[element])] += buildings * (1-fault_combined[row])
            buildings += -1
    pref = reorder_by_rank(list(scores.values()), nb)
    pref = list(map(int, pref))
    return pref, return_f

def kendall_tau(rank_a, rank_b):
    pairs = combinations(rank_a,2)
    dist = 0
    for x, y in pairs:
        a = rank_a.index(x) - rank_a.index(y)
        b = rank_b.index(x) - rank_b.index(y)
        if a * b < 0:
            dist += 1
    return dist

def calculate_pairwise_matrix(df,weights,competitors):
    num_of_buildings = len(pd.DataFrame.count(df, axis=0))
    num_of_unique_voters = len(pd.DataFrame.count(df, axis=1))
    df.index = range(num_of_unique_voters)
    pairwise_matrix = np.zeros((num_of_buildings, num_of_buildings))
    # competitors = list(df.keys())
    building1, building2 = 0, 0
    for x in range(0,num_of_buildings,1):
        for y in range(x+1,num_of_buildings,1):
            building1 = competitors[x]
            building2 = competitors[y]
            if (building1 >= building2):
                continue
            scores = {building1: 0, building2: 0}
            for row in range(0, num_of_unique_voters, 1):
                array = df.loc[[row]]
                for element in array:
                    if (int(array[element]) == building1):
                        scores[building1] += weights[row]
                        break
                    if (int(array[element]) == building2):
                        scores[building2] += weights[row]
                        break
            pairwise_winner, pairwise_loser = 0, 0
            if (scores[building1] == scores[building2]):
                frac = random.random()
                if(frac>0.5):
                    pairwise_winner = building1
                    pairwise_loser = building2
                else:
                    pairwise_winner = building2
                    pairwise_loser = building1
            else:
                pairwise_winner = max(scores, key=scores.get)
                pairwise_loser = min(scores, key=scores.get)
            if (pairwise_loser != pairwise_winner):
                pairwise_matrix[int(competitors.index(pairwise_winner)), int(competitors.index(pairwise_loser))] = 1
    return pairwise_matrix

def reorder_by_rank(order,buildings):
    dict = {}
    building = 0
    for rank_score in order:
        dict[building] = rank_score
        building += 1
    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}
    pref, tie, exist = ([] for i in range(3))
    for b, s in sorted_dict.items():
        if b in exist:
            continue
        exist.append(b)
        tie.append(b)
        for b2, s2 in sorted_dict.items():
            if b2 in exist:
                continue
            if s != s2:
                break
            else:
                tie.append(b2)
                exist.append(b2)
        order = random_tie_break(tie)
        pref += order
    return_pref = [int(i) for i in pref]
    return_pref.reverse()
    pref_order = []
    for i in range(len(buildings)):
        pref_order.append(buildings[return_pref[i]])
    return pref_order

def random_tie_break(list_tie):
    copy_list_tie = copy.deepcopy(list_tie)
    n = len(copy_list_tie)
    order = []
    for i in range (0,n):
        building = random.choice(list_tie)
        order.append(building)
        list_tie.remove(building)
    return order

def Estimate_Error(pref , votes):
     n = len(votes)
     sum_dist = 0
     for row in votes:
         if row != pref:
            sum_dist += kendall_tau(pref , row)
     pi = (1/(n-1)) * sum_dist
     return pi

def scatter_plot(votes):
    truth_df = pd.read_csv("votes - Truth.csv")
    buildings = truth_df.keys().tolist()
    rankings = truth_df.values.tolist()
    true_buildings = []
    rank = rankings[0]
    for i in range(len(buildings)):
        true_buildings.append(int((buildings[rank[i]])))
    num_of_buildings = len(pd.DataFrame.count(truth_df, axis=0))
    num_of_unique_voters = len(pd.DataFrame.count(votes, axis=1))
    new_votes = copy.deepcopy(votes)
    new_votes[~new_votes.isin(true_buildings)] = -1
    arch_votes_list = new_votes.values.tolist()
    d_list = []
    pi_votes_list = []
    for vote in arch_votes_list:
        new_rank = []
        for element in vote:
            if (element>=0):
                new_rank.append(element)
        pi_votes_list.append(new_rank)
        d = kendall_tau(new_rank,true_buildings)
        d = d / ( num_of_buildings* (num_of_buildings - 1) / 2)
        d_list.append(d)
    return_f = []
    for row in pi_votes_list:
        pi = Estimate_Error(row, pi_votes_list)
        fi = pi
        fi = fi / (num_of_buildings * (num_of_buildings - 1) / 2)
        return_f.append(fi)
    x = list(range(0,num_of_unique_voters))
    plt.scatter(d_list, return_f, label = 'KT Distance VS Proxy pi', color='red')
    plt.legend(loc='best')
    plt.title('The distance from the truth VS The Proxy distance pai')
    plt.xlabel('Distance')
    plt.ylabel('pi')
    plt.show()

def true_error(votes):
    truth_df = pd.read_csv("votes - Truth.csv")
    num_of_buildings = len(pd.DataFrame.count(truth_df, axis=0))
    buildings = truth_df.keys().tolist()
    rankings = truth_df.values.tolist()
    true_buildings = []
    rank = rankings[0]
    for i in range(len(buildings)):
        true_buildings.append(int((buildings[rank[i]])))
    new_votes2 = copy.deepcopy(votes)
    new_votes2[~new_votes2.isin(true_buildings)] = -1
    arch_votes_list = new_votes2.values.tolist()
    data = []
    for vote in arch_votes_list:
        new_rank = []
        for element in vote:
            if (element >= 0):
                new_rank.append(element)
        data.append(new_rank)
    new_df = pd.DataFrame(data)
    sample_size = list(range(5,90,5))
    c_dtd, c_ptd, c_uw, b_dtd, b_ptd, b_uw, y_alg  = ([] for i in range(7))
    for sample in sample_size:
        num_of_samples = 0
        c_dtd_avg, c_ptd_avg, c_uw_avg, b_dtd_avg, b_ptd_avg, b_uw_avg, y_alg_avg = (0 for i in range(7))
        num_of_iterations = 150
        print("now running in sample:",sample)
        while(num_of_samples<=num_of_iterations):
            num_of_samples += 1
            sample_df = new_df.sample(n=sample)
            x_hat,err = DTD_Copeland(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            c_dtd_avg+=d
            x_hat,err = PTD_Copeland(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            c_ptd_avg+=d
            x_hat,err = UW_Copeland(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            c_uw_avg+=d
            x_hat,err = DTD_borda(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            b_dtd_avg+=d
            x_hat,err = PTD_borda(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            b_ptd_avg+=d
            x_hat,err = UW_borda(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            b_uw_avg+=d
            x_hat,err = your_algorithm(sample_df)
            d = kendall_tau(x_hat, true_buildings)
            d = d / (num_of_buildings * (num_of_buildings - 1) / 2)
            y_alg_avg+=d
        c_dtd.append(c_dtd_avg/num_of_samples)
        c_ptd.append(c_ptd_avg/num_of_samples)
        c_uw.append(c_uw_avg/num_of_samples)
        b_dtd.append(b_dtd_avg/num_of_samples)
        b_ptd.append(b_ptd_avg/num_of_samples)
        b_uw.append(b_uw_avg/num_of_samples)
        y_alg.append(y_alg_avg/num_of_samples)

    plt.plot(sample_size, c_dtd, label='DTD copeland')
    plt.plot(sample_size, c_ptd, label='PTD copeland')
    plt.plot(sample_size, c_uw, label='UW copeland')
    plt.plot(sample_size, y_alg, label='Our Algorithm')
    plt.legend(loc='best')
    plt.title('AVG Error as a function of Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('AVG Error')
    plt.legend()
    plt.show()

    plt.plot(sample_size, b_dtd, label='DTD borda')
    plt.plot(sample_size, b_ptd, label='PTD borda')
    plt.plot(sample_size, b_uw, label='UW borda')
    plt.plot(sample_size, y_alg, label='Our Algorithm')
    plt.legend(loc='best')
    plt.title('AVG Error as a function of Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('AVG Error')
    plt.legend()
    plt.show()

    # plt.plot(sample_size, y_alg, label='Our Algorithm')
    # plt.legend(loc='best')
    # plt.title('AVG Error as a function of Sample Size')
    # plt.xlabel('Sample Size')
    # plt.ylabel('AVG Error')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':

    votes = pd.read_csv("voters.csv", header=None)
    # with open('voters.csv') as file:
    #     votes = []
    #     reader = csv.reader(file, delimiter=',')
    #     for line in reader:
    #         votes.append(line)

    # print(DTD_borda(votes))
    # print(DTD_Copeland(votes))
    # print(PTD_borda(votes))
    # print(PTD_Copeland(votes))
    # print(UW_borda(votes))
    # print(UW_Copeland(votes))
    # print(your_algorithm(votes))


    with open("estimations.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(DTD_borda(votes))
        wr.writerow(DTD_Copeland(votes))
        wr.writerow(PTD_borda(votes))
        wr.writerow(PTD_Copeland(votes))
        wr.writerow(UW_borda(votes))
        wr.writerow(UW_Copeland(votes))
        wr.writerow(your_algorithm(votes))
        scatter_plot(votes)
        true_error(votes)