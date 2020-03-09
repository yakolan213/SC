import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def print_plt(tie_candidates, bar_v, height, colors_list, v,model_name):
    plt.figure(figsize=(11, 6))
    plt.gcf().subplots_adjust(bottom=0.20)
    legend = ''
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
    plt.xlabel('Parameter Value', {'fontsize': 15},)
    plt.yticks((), color='k', size=10)
    plt.bar(y_pos, height, color=colors_list)
    new_bar_l = []
    for x in bar_v:
        temp = ''
        if (model_name == 'AU'):
            # for y in x:
            #     temp += y + '\n'
            old = "), ("
            new = ")\n("
            temp = x.replace(old,new)
        else:
             temp = str(x)
        new_bar_l.append(temp)
    plt.xticks(y_pos, new_bar_l,fontsize=10)
    title = "Model: " + model_name + '\n' + 'Voter ID:' + '{0}'.format(v.id)
    plt.title(title)
    plt.show()
    x=1

def create_color_list(param_values_dict):
    bar_v = []
    height = []
    colors_list = []
    tie_candidates = {}
    for param, candidate in param_values_dict.items():
        bar_v.append(str(param))
        height.append(1)
        if len(candidate) == 1:
            if candidate[0] == 'Grey':
                colors_list.append('grey')
            elif candidate[0] == 'Blue':
                colors_list.append('blue')
            elif candidate[0] == 'Red':
                colors_list.append('red')
        else:
            colors_list.append('green')
            tie_candidates[str(param)] = candidate
    return tie_candidates, bar_v, height, colors_list

def create_plot(param_values_dict, v, model_name):
    """
    create the plot object
    :param param_values_dict: the parameter we want to create the plot for
    :param v: the voter object, in order to give the plot an id
    :return: plot figure
    """
    temp_dict = {}
    for param, candidate in param_values_dict.items():
        if len(temp_dict) < 3:
            temp_dict[param] = candidate
        else:
            tie_candidates, bar_v, height, colors_list = create_color_list(temp_dict)
            temp_dict = {}
            print_plt(tie_candidates, bar_v, height, colors_list, v,model_name)
    tie_candidates, bar_v, height, colors_list = create_color_list(temp_dict)
    print_plt(tie_candidates, bar_v, height, colors_list, v, model_name)
