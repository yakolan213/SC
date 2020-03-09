import pandas
from voters import Voter
import argparse
import plots

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



def parser_object():
    """
    :return: Parse the parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_name", help="Experiment file name", type=str)
    parser.add_argument("-model_name", help="Prediction model, if you chose KP model you must fill k parameter too", choices=['KP', 'CV', 'AU', 'LD', 'AT'])
    parser.add_argument("-k", help="The size of the winners group, can be 1/2/3 for KP model", type=int, default=None)
    parser.add_argument("-division_param", help="Division parameter", type=int, default=3)
    parser.add_argument("-l_bound", help="Lower bound for the parameter", type=int, default=None)
    parser.add_argument("-u_bound", help="Higher bound for the parameter", type=int, default=None)
    parser.add_argument("-division_param_b", help="Division parameter for a parameter in AU", type=int, default=None)
    parser.add_argument("-division_param_a", help="Division parameter b parameter in AU", type=int, default=None)
    parser.add_argument("-l_bound_a", help="Lower bound for the a parameter for AU model", type=int, default=None)
    parser.add_argument("-u_bound_a", help="Higher bound for the a parameter for AU model", type=int, default=None)
    parser.add_argument("-l_bound_b", help="Lower bound for the b parameter for AU model", type=int, default=None)
    parser.add_argument("-u_bound_b", help="Higher bound for the b parameter for AU model", type=int, default=None)
    parser.add_argument("-e", help="The error allowed for AU model", type=float, default=None)
    parser.add_argument("-voter_id", help="The voter ID we want to check the models parameter for", type=str, default=None)
    parser.add_argument("-voters_start_index", help="The voters start index in the voters list", type=int, default=0)
    parser.add_argument("-voters_end_index", help="The voter end index in the voters list", type=int, default=9999999999)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_object()
    voters, candidates = create_voters(args.file_name)
    model = args.model_name
    result = []
    checked_voters = []
    if args.voter_id:
        for voter in voters:
            if voter.id == args.voter_id:
                checked_voters.append(voter)
    else:
        checked_voters = voters[args.voters_start_index: args.voters_end_index]

    if model == 'KP':
        if not args.k:
            raise RuntimeError('For KP calculation you must enter k parameter')
        for voter in checked_voters:
            result,s = voter.get_KP_parameter(args.k)
            plots.create_plot(result, voter, 'KP',s)

    elif model == 'CV':
        if not args.division_param or not args.l_bound or not args.u_bound:
            raise RuntimeError('For CV calculation you must enter division_param, l_bound and u_bound parameters')
        if args.l_bound > args.u_bound:
            raise RuntimeError('Lower bound is bigger then the upper bound!')
        for voter in checked_voters:
            result,s = voter.get_CV_parameter(checked_voters, args.division_param, args.l_bound, args.u_bound)
            plots.create_plot(result, voter, 'CV',s)

    elif model == 'LD':
        if not args.division_param or not args.l_bound or not args.u_bound:
            raise RuntimeError('For LD calculation you must enter division_param, l_bound and u_bound parameters')
        if args.l_bound > args.u_bound:
            raise RuntimeError('Lower bound is bigger then the upper bound!')
        for voter in checked_voters:
            result,s = voter.get_LD_parameter(args.division_param, args.l_bound, args.u_bound)
            plots.create_plot(result, voter, 'LD',s)

    elif model == 'AT':
        if not args.division_param or not args.l_bound or not args.u_bound:
            raise RuntimeError('For AT calculation you must enter division_param, l_bound and u_bound parameters')
        if args.l_bound > args.u_bound:
            raise RuntimeError('Lower bound is bigger then the upper bound!')
        for voter in checked_voters:
            result,s = voter.get_AT_parameter(args.division_param, args.l_bound, args.u_bound)
            plots.create_plot(result, voter, 'AT',s)

    elif model == 'AU':
        if not args.division_param_a\
                or not args.division_param_b\
                or not args.e:
            raise RuntimeError('For AU calculation you must enter division_param_a, division_param_b and '
                               'e parameters, the params are {0}'.format(args))
        if args.l_bound_a > args.u_bound_a:
            raise RuntimeError('Lower bound is bigger then the upper bound!')
        for voter in checked_voters:
            result,s = voter.get_AU_parameters(args.division_param_b, args.l_bound_b,
                                            args.u_bound_b, args.e,
                                            args.division_param_a, args.l_bound_a, args.u_bound_a)
            plots.create_plot(result, voter, 'AU',s)




