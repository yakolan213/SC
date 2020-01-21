import pandas
import random
from voters import Voter

def create_voters(path):
    """
    :param path: (str) path to the excel file
    :return: voters objects list
    """
    excel_file = pandas.read_excel(path, skiprows=1)
    voters = []
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
    return voters


if __name__ == '__main__':
    voters = create_voters('OneShot-FullData_2204.xlsx')
    #find_pivotal_probabilities(3 , voters[:3])
    for v in voters:
        #v.get_KP_parameter(3)
        v.get_CV_parameter(voters,50, 6000, 7000)