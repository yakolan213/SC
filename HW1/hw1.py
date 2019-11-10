import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#num_of_shirts = pd.read_csv("preferences.csv",sep=',', header = None, nrows = 1 )
num_of_shirts = 11
shirts = {1:'Australia',2:'Braille',3:'Brush Strokes',4:'Exponential',5:'College',6:'Graph Coloring',
          7:'Red',8:'Simple',9:'Star Trek',10:'TSP',11:'VRP'}
col_Names = list(range(0,(num_of_shirts+1)))
#print(col_Names)
df = pd.read_csv("preferences.csv",skiprows=13,names=col_Names)

### Plurality ###
votes = shirts.copy()
for key in votes.keys():
    votes[key] = 0
for vote in df[1]:
    votes[vote]+=1
winner = max(votes, key=votes.get)
print("Plurality winner: ",shirts[winner],", scores: ",votes)

### Borda ###
votes = shirts.copy()
for key in votes.keys():
    votes[key] = 0
keys = df.keys()[df.keys()>0]
for key in keys:
    for num in df[key]:
        votes[num] += (num_of_shirts - key)
winner = max(votes, key=votes.get)
print("Borda winner: ",shirts[winner],", scores: ",votes)

### Nanson ###



