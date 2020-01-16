import numpy as np
import pandas as pd
import networkx as nx
import copy
from itertools import combinations

### Read file to get data ###
parse_shirts = pd.read_csv("preferences.csv",nrows=1,skip_blank_lines=1)
num_of_shirts = int(parse_shirts.keys()[0])
parse_voters = pd.read_csv("preferences.csv",nrows=1,skip_blank_lines=1,skiprows=(num_of_shirts+1))
num_of_unique_voters = float(parse_voters.keys()[2])
num_of_unique_voters = round(num_of_unique_voters)
# shirts = {1:'Australia',2:'Braille',3:'Brush Strokes',4:'Exponential',5:'College',6:'Graph Coloring',
#           7:'Red',8:'Simple',9:'Star Trek',10:'TSP',11:'VRP'}
with open("preferences.csv", 'r') as preferences:
    shirts_raw = preferences.readlines()[1:num_of_shirts + 1]
shirts = {}
for line in shirts_raw:
    line_objests = line.split(',')
    shirts[int(line_objests[0])] = line_objests[1]
col_names = list(range(0, (num_of_shirts+1)))
df = pd.read_csv("preferences.csv",skiprows=(num_of_shirts+2),names=col_names)
num_of_same_pref = df[0]
our_rule_keys = df.drop([0], axis=1)
our_rule = []

### Pairwise Winners - Compute pairwise winners to build a matrix from which we can determine Condorcent Winners. ###
condorcent = 1
copeland = shirts.copy()
#Set Data for Copeland as derive from Condorcent check.
for key in copeland.keys():
    copeland[key] = 0
shirts_left = num_of_shirts
c_winners = list()
votes = shirts.copy()
#Look for all Condorcent winners
while(condorcent == 1):
    pairwise_matrix = np.zeros( (num_of_shirts, num_of_shirts) )
    tmp_df = copy.deepcopy(df)
    tmp_df = tmp_df.drop([0], axis=1)
    len_competitors = len(votes.keys())
    competitors = list(votes.keys())
    shirt1,shirt2 = 0,0
    for x in range(len_competitors):
        for y in range(len_competitors):
            shirt1 = competitors[x]
            shirt2 = competitors[y]
            if(shirt1==shirt2):
                continue
            scores = {shirt1: 0, shirt2: 0}
            for row in range(0, num_of_unique_voters, 1):
                array = tmp_df.loc[[row]]
                for element in array:
                    if (int(array[element]) == shirt1):
                        same_pref = num_of_same_pref[row]
                        scores[shirt1] += int(same_pref)
                        break
                    if (int(array[element]) == shirt2):
                        same_pref = num_of_same_pref[row]
                        scores[shirt2] += int(same_pref)
                        break
            pairwise_winner,pairwise_loser= 0,0
            if(scores[shirt1]==scores[shirt2]):
                if(shirts[shirt1]<shirts[shirt2]):
                    pairwise_winner = shirt1
                    pairwise_loser = shirt2
                else:
                    pairwise_winner = shirt2
                    pairwise_loser = shirt1
            else:
                pairwise_winner = max(scores, key=scores.get)
                pairwise_loser = min(scores, key=scores.get)
            if(pairwise_loser!=pairwise_winner):
                pairwise_matrix[pairwise_winner-1,pairwise_loser-1] = 1
    our_rule = pairwise_matrix
    len_c = len(c_winners)
    if(len_c>0):
        for i in range(len_c):
            for s in range(len_competitors):
                pairwise_matrix[s,c_winners[i]-1] = 0
    find_condorcent = pairwise_matrix.sum(axis=1)
    len_condorcent = len(find_condorcent)
    for i in range(len_condorcent):
        copeland[i+1] = int(find_condorcent[i])
        if (find_condorcent[i] == shirts_left -1):
            print("Alternative",shirts[i+1],"is a Condorcet winner")
            condorcent =1
            df[df[:] == i+1] = 0
            c_winners.append(i+1)
            votes[i+1] = "Condorcent Winner"
            shirts_left = shirts_left -1
            votes.pop(i+1, None)
            break
        else:
            condorcent = 0

### Plurality ###
for key in votes.keys():
    votes[key] = 0
tmp_df = copy.deepcopy(df)
tmp_df = tmp_df.drop([0], axis=1)
for row in range(0, num_of_unique_voters, 1):
    array = tmp_df.loc[[row]]
    for element in array:
        if (int(array[element]) != 0):
            same_pref = num_of_same_pref[row]
            votes[int(array[element])] += same_pref
            break
winner = max(votes, key=votes.get)
print("Plurality winner:",shirts[winner],",scores:",votes)

### Borda ###
for key in votes.keys():
    votes[key] = 0
temp_df = copy.deepcopy(df)
temp_df = temp_df.drop([0], axis=1)
for row in range(0, num_of_unique_voters, 1):
    shirts_counting = shirts_left-1
    array = temp_df.loc[[row]]
    for element in array:
        if(int(array[element]) != 0):
            same_pref = int(num_of_same_pref[row])
            votes[int(array[element])] += shirts_counting*same_pref
            shirts_counting += -1
winner = max(votes, key=votes.get)
print("Borda winner:",shirts[winner],",scores:",votes)

### Nanson ###
temp_df = copy.deepcopy(df)
temp_df = temp_df.drop([0], axis=1)
winners = list(votes.keys())
winner,same_mean=0,0
remove_list = list()
round = 1
scores = copy.deepcopy(votes)
for key in scores.keys():
    scores[key] = 1
while(len(winners)>1 and winner ==0):
    round += 1
    for key in votes.keys():
        votes[key] = 0
    sum = 0
    for row in range(0, num_of_unique_voters, 1):
        shirts_left = len(winners) - 1
        array = temp_df.loc[[row]]
        for element in array:
            if(int(array[element]) != 0):
                same_pref = num_of_same_pref[row]
                votes[int(array[element])] += shirts_left*same_pref
                shirts_left -= 1
    for score in votes.values():
        sum += score
    mean = sum/len(winners)
    for canidate in winners:
        if int(votes[canidate]) < mean:
            remove_list.append(canidate)
            temp_df[temp_df[:] == canidate] = 0
        else:
            scores[canidate] = round
    for i in remove_list:
        winners.remove(i)
    remove_list = list()
    if(same_mean==mean):
        winner = shirts[winners[0]]
        for i in range(1,len(winners),1):
            if(shirts[winners[i]]<winner):
                winner = shirts[winners[i]]
    same_mean = mean
    if(len(winners)==1):
        winner = shirts[winners[0]]
print("Nanson winner:", winner,",scores:",scores)

### Single transferable vote ###
winner,round = 0,1
scores = copy.deepcopy(votes)
for key in scores.keys():
    scores[key] = 1
tmp_df = copy.deepcopy(df)
tmp_df = tmp_df.drop([0], axis=1)
while(len(votes)>1):
    for key in votes.keys():
        votes[key] = 0
        if(round>1):
            scores[key] = round
    for row in range(0, num_of_unique_voters, 1):
        array = tmp_df.loc[[row]]
        for element in array:
            if (int(array[element]) != 0):
                same_pref = num_of_same_pref[row]
                votes[int(array[element])] += int(same_pref)
                break
    eliminate = min(votes, key=votes.get)
    for key in votes.keys():
        if(votes[key]==votes[eliminate] and shirts[key]>shirts[eliminate]):
            eliminate = key
    votes.pop(eliminate)
    tmp_df[tmp_df[:] == eliminate] = 0
    round += 1
print("STV winner:", shirts[list(votes.keys())[0]],"scores:",scores)

### Copeland ###
winner = max(copeland, key=copeland.get)
for i in c_winners:
    copeland.pop(i, None)
print("Copeland winner:",shirts[winner],",scores:",copeland)

### PageRank ###
tran_matrix = np.transpose(pairwise_matrix)
G = nx.DiGraph(tran_matrix)
SG=nx.stochastic_graph(G)
iterations = 100
alpha = 0.85
x=dict.fromkeys(SG, 1.0 / SG.number_of_nodes())
num_of_nodes=SG.number_of_nodes()
out_degree=SG.out_degree(weight=None)
for i in range(iterations):
    x_last=x
    x=dict.fromkeys(x_last.keys(), 0)
    sum_values = 0
    for value in x_last.values():
        sum_values += value
    teleport_sum= (1.0 - alpha) / num_of_nodes * sum_values
    for n in x:
        for num in SG[n]:
            x[num]+= alpha * x_last[n]*SG[n][num]['weight']
        x[n] += teleport_sum
    inv=1.0/sum_values
    for n in x:
        x[n]*=inv
        x[n] = float("{0:.5f}".format(x[n]))
pagerank = votes
for key in x.keys():
    pagerank[int(key+1)] = x[key]
print("PageRank:",pagerank)

### Kendal-Tau Distance ###
profile_num1 = num_of_unique_voters - 1
profile_num2 =1
kt_df = copy.deepcopy(df)
kt_df = kt_df.drop([0], axis=1)
profile1 = kt_df.loc[profile_num1]
profile2 = kt_df.loc[profile_num2]
rank_a = list()
rank_b = list()
for x in profile1:
    if(x!=0):
        rank_a.append(x)
for y in profile2:
    if(y!=0):
        rank_b.append(y)
pairs = combinations(range(1, len(rank_a)+1), 2)
dist = 0
for x, y in pairs:
    a = rank_a.index(x) - rank_a.index(y)
    b = rank_b.index(x) - rank_b.index(y)
    if a * b < 0:
        dist += 1
print("Profile 1:",rank_a)
print("Profile 2:",rank_b)
print("Kendal-Tau Distance:",dist)

### Our Rule - Tournamnet/Playoffs (more info in PDF file) ###
A = list(our_rule_keys.keys())
scores = dict([(key, 0) for key in our_rule_keys.keys()])
B = A[:len(A)//2]
C = A[len(A)//2:]
round=1
while(len(B)+len(C)>1):
    remove_listB = list()
    remove_listC = list()
    if(len(B)==0 or len(C)==0):
        temp = max(B,C)
        B = temp[:len(temp) // 2]
        C = temp[len(temp) // 2:]
    for i in range(min(len(B),len(C))):
        if(our_rule[B[i]-1][C[i]-1] == 1):
            remove_listC.append(C[i])
        else:
            remove_listB.append(B[i])
    for i in remove_listB:
        scores[i] = round
        B.remove(i)
    for i in remove_listC:
        scores[i] = round
        C.remove(i)
    round+=1
winner = max(B+C)
scores[winner]= "Winner"
print("Our Rule winner:", shirts[winner],"scores:",scores)
