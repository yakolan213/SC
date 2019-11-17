import numpy as np
import pandas as pd
import networkx as nx
import copy
from itertools import combinations

#num_of_shirts = pd.read_csv("preferences.csv",sep=',', header = None, nrows = 1 )
parse_shirts = pd.read_csv("preferences.csv",nrows=1,skip_blank_lines=1)
num_of_shirts = int(parse_shirts.keys()[0])
parse_voters = pd.read_csv("preferences.csv",nrows=1,skip_blank_lines=1,skiprows=(num_of_shirts+1))
num_of_unique_voters = float(parse_voters.keys()[2])
num_of_unique_voters = round(num_of_unique_voters)
#shirts_kind = pd.read_csv("preferences_yakov.csv",nrows=num_of_shirts-1 ,skip_blank_lines=1, skiprows=1)
shirts = {1:'Australia',2:'Braille',3:'Brush Strokes',4:'Exponential',5:'College',6:'Graph Coloring',
          7:'Red',8:'Simple',9:'Star Trek',10:'TSP',11:'VRP'}
col_names = list(range(0, (num_of_shirts+1)))
#print(col_Names)
df = pd.read_csv("preferences.csv",skiprows=(num_of_shirts+2),names=col_names)
num_of_same_pref = df[0]
our_rule_keys = df.drop([0], axis=1)
our_rule = []
### Pairwise Winners ###
condorcent = 1
copeland = shirts.copy()
for key in copeland.keys():
    copeland[key] = 0
shirts_left = num_of_shirts
c_winners = list()
votes = shirts.copy()
while(condorcent == 1):
    pairwise_matrix = np.zeros( (num_of_shirts, num_of_shirts) )
    tmp_df = copy.deepcopy(df)
    tmp_df = tmp_df.drop([0], axis=1)
    len_competitors = len(votes.keys())
    competitors = list(votes.keys())
    #len_competitors = len(tmp_df.keys())
    #competitors = list(tmp_df.keys())
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
#votes = dict([(key, 0) for key in temp_df.keys()])
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
#votes = dict([(key, 0) for key in temp_df.keys()])
winners = list(votes.keys())
winner,same_mean=0,0
remove_list = list()
#for i in range(len_c):
    #winners.remove(c_winners[i])
round = 1
scores = copy.deepcopy(votes)
for key in scores.keys():
    scores[key] = 1
while(len(winners)>1 and winner ==0):
    round += 1
    for key in votes.keys():
        votes[key] = 0
    #votes = dict([(key, 0) for key in temp_df.keys()])
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
while(winner==0):
    for key in votes.keys():
        votes[key] = 0
        if(round>1):
            scores[key] = round
    for row in range(0, num_of_unique_voters, 1):
        array = tmp_df.loc[[row]]
        for element in array:
            if (int(array[element]) != 0):
                same_pref = num_of_same_pref[row]
                votes[int(array[element])] += same_pref
                break
    leader = max(votes, key=votes.get)
    remove_list = list()
    for key in votes.keys():
        if(votes[key]==votes[leader] and key!=leader):
            remove_list.append(min(votes, key=votes.get))
    if(len(remove_list)==0):
            winner = leader
    for i in remove_list:
        votes.pop(i)
        tmp_df[tmp_df[:] == i + 1] = 0
    remove_list = list()
    round += 1
print("STV winner:", shirts[winner],"scores:",scores)

### Copeland ###
winner = max(copeland, key=copeland.get)
for i in c_winners:
    copeland.pop(i, None)
print("Copeland winner:",shirts[winner],",scores:",copeland)

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
dist = 0
n_candidates = len(rank_a)
for i, j in combinations(range(n_candidates), 2):
    dist += (np.sign(rank_a[i] - rank_a[j]) ==
            -np.sign(rank_b[i] - rank_b[j]))
print("Profile 1:",rank_a)
print("Profile 2:",rank_b)
print("Kendal-Tau Distance:",dist)

# print(pairwise_matrix)
# print(np.transpose(pairwise_matrix))
# G = nx.DiGraph(np.transpose(pairwise_matrix))
# print(nx.pagerank(G))

### PageRank ###
def pagerank(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

        # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

            # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x

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
