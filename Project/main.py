import pandas
from voters import Voter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from UI import UI


def create_voters(path):
    """
    :param path: (str) path to the excel file
    :return: voters objects list
    """
    excel_file = pandas.read_excel(path, skiprows=1)
    voters = []
    candidates = {}
    for index, row in excel_file.iterrows():
        candidates = {1: row['CanName1'],
                      2: row['CanName2'],
                      3: row['CanName3']}
        vote = candidates[row['NewVote']]
        preference_list = [candidates[row['Pref1']],
                           candidates[row['Pref2']],
                           candidates[row['Pref3']]]
        utilities_dict = {preference_list[0]: row['Util1'],
                          preference_list[1]: row['Util2'],
                          preference_list[2]: row['Util3']}
        id = row.VoterID
        s = {candidates[1]: row['VotesCand1'],
             candidates[2]: row['VotesCand2'],
             candidates[3]: row['VotesCand3']}
        voter = Voter(id=id,
                      preference_list=preference_list,
                      utilities_dict=utilities_dict,
                      vote=vote,
                      s=s)
        voters.append(voter)
    return voters, candidates

def create_plot(param_values_dict, v):
    """
    create the plot object
    :param param_values_dict: the parameter we want to create the plot for
    :param v: the voter object, in order to give the plot an id
    :return: plot figure
    """
    bar_v = []
    height = []
    colors_list = []
    tie_candidates = {}
    param_list = []
    legend = ''
    for param, candidate in param_values_dict.items():
        bar_v.append(param)
        height.append(1)
        param_list.append(param)
        if len(candidate) == 1:
            if candidate[0] == 'Grey':
                colors_list.append('grey')
            elif candidate[0] == 'Blue':
                colors_list.append('blue')
            elif candidate[0] == 'Red':
                colors_list.append('red')
        else:
            colors_list.append('green')
            tie_candidates[param] = candidate

    red = mpatches.Patch(color='red', label='Red candidate')
    blue = mpatches.Patch(color='blue', label='Blue candidate')
    grey = mpatches.Patch(color='grey', label='Grey candidate')

    for param, candidate in tie_candidates.items():
        legend += 'Param {0}, tie between {1} \n'.format(param, candidate)

    if not legend:
        legend = 'Tie'
    green = mpatches.Patch(color='green', label=legend)

    plt.legend(handles=[red, blue, grey, green], loc='upper left')

    y_pos = np.arange(len(bar_v))

    plt.ylabel('Candidate', {'fontsize': 15})
    plt.xlabel('Parameter Value', {'fontsize': 15})
    plt.yticks((), color='k', size=10)
    plt.bar(y_pos, height, color=colors_list)
    plt.xticks(y_pos, bar_v)
    plt.title('{0} Voter'.format(v.id))
    #fig = plt.figure(num=v.id, figsize=(10,6), bbox_inches='tight')
    plt.show()
    return



if __name__ == '__main__':
    voters, candidates = create_voters('OneShot-FullData_2204.xlsx')
    plot_list = []
    #find_pivotal_probabilities(3 , voters[:3])
    for v in voters[:4]:
        x = v.get_KP_parameter(3)
        v_plot = create_plot(x, v)
        plot_list.append(v_plot)
    UI_obj = UI(voters)



