from __future__ import print_function
from __future__ import absolute_import
import json
import pickle
import pandas as pd
import numpy as np

def load_json(file_path):
    with open(file_path) as f: 
        return json.load(f)


def load_pickle(file_path):
    with open(file_path, 'rb') as pkl:
        return pickle.load(pkl)   


def load_lookup_data(lookup_name, path='./lookup'):
    if not path.endswith('/'):
        path = path + '/'
    lookup_path = "{}{}.csv".format(path, lookup_name)
    try:
        lookup = pd.read_csv(lookup_path)        
        if lookup_name == 'PTB-LSTM':            
            lookup['best_perplexity'] = get_best_of_train(lookup, metric='perplexity')
            max_perplexity = 1000.0
            lookup['best_acc'] = (max_perplexity - lookup['best_perplexity']) / max_perplexity  
        elif lookup_name == 'CIFAR10-VGG':
            lookup['best_acc'] = get_best_of_train(lookup, start_col=11, end_col=61)
        elif lookup_name == 'CIFAR100-VGG':
            lookup['best_acc'] = get_best_of_train(lookup, start_col=11, end_col=61)
        elif lookup_name == 'CIFAR10-ResNet':
            lookup['best_acc'] = get_best_of_train(lookup, start_col=9, end_col=109)            
        else:
            lookup['best_acc'] = get_best_of_train(lookup)
        return lookup
    except:
        raise ValueError("{} is not found.".format(lookup_path))


def get_best_of_train(lookup_table, start_col=10, end_col=25, metric='acc'):
    if metric == 'acc':
        return np.max(get_train_curve(lookup_table, start_col, end_col), axis=1)
    elif metric == 'perplexity':
        best_perp = np.min(get_train_curve(lookup_table, start_col, end_col), axis=1)
        return best_perp  


def get_train_curve(lookup_table, start_col, end_col):
    return lookup_table.iloc[:, start_col:end_col].values


def get_difficulty_stats(lookup, difficulties=[]):
    num_lookup = len(lookup)
    if len(difficulties) == 0:
        difficulties = np.array([float(100/num_lookup), float(10/num_lookup)])    
    stats = []
    best_accs = lookup['best_acc']
    sorted_accs = np.sort(best_accs)[::-1]
    top_acc = max(sorted_accs)
    t_i = 0
    for df in difficulties:
        stat = {}
        th = int(df * num_lookup)        
        stat['difficulty'] = float(df)
        stat['rank'] = int(num_lookup * float(df))
        stat['error'] = 1.0 - sorted_accs[th]
        stat['accuracy'] = sorted_accs[th]
        stat['regret'] = top_acc - sorted_accs[th]
        stats.append(stat)
    return stats


def name_map(name):
    if name == 'SP-GP-EI(6)':
        return 'Synch. GP-EI-MCMC(10)'
    elif name == 'P-GP-EI(6)':
        return 'GP-EI-MCMC(10)'
    elif name == 'P-GP-EI-MCMC1(6)':
        return 'GP-EI-MCMC(1)'
    elif name == 'P-RF-EI(6)':
        return 'RF-EI'
    elif name == 'xN-Div-I':
        return 'Theoretical'
    elif name == 'P-Div(6)':
        return 'P-Div'
    elif name == 'P-Div-P(6)':
        return 'P-Div (in-progress)'
    else:
        return name


def get_label(arm):
    label = arm.replace('_', '-')
    postfix = ""

    if '-NR' in label:
        label = label.replace("-NR", "-R")
    elif '-NC' in label:
        label = label.replace("-NC", "-N")
    elif '-NP' in label:
        label = label.replace("-NP", "-P")  

    if '-LOG-ERR' in label:
        label = label.replace("-LOG-ERR", " ")
        postfix = " (log err)"
    elif '-ADALOG3TIME' in label:
        label = label.replace("-ADALOG3TIME", "-Div")
        #postfix = " (partial log + early stop)"
    elif '-ADALOG3' in label:
        label = label.replace("-ADALOG3", "-Div")
        #postfix = " (partial log)"
    elif '-TIME' in label:
        label = label.replace("-TIME", " ")
        postfix = " (early stop)"
    elif '-LOGMIX' in label:
        label = label.replace("-LOGMIX", " ")
        postfix = " (pure & adalog)"

    if 'SMAC-' in label:
        label = label.replace('SMAC-', 'RF-',)
    elif '-NM' in label:
        label = label.replace('-NM', '-MCMC1')

    if '-HLE' in label:
        label = label.replace('-HLE', '')

    elif 'DIVERSIFIED' in label:
        if 'RANDOM' in label:
            return 'R-Div' + postfix
        elif 'SEQ' in label:
            return 'S-Div' + postfix
        elif 'SKO' in label:
            return 'S-Knockout' + postfix
        elif 'HEDGE' in label:
            return 'Hedge' + postfix
        elif 'GT-' in label:
            return u"\u03B5" + "-greedy"
        elif 'EG-' in label:
            return 'e-greedy' + postfix

    if 'BATCH' in label:
        label = label.replace('ASYNC-BATCH', 'P')
        label = label.replace('SYNC-BATCH', 'SP')

        if 'P-GP+SMAC' in label:
            label = label.replace('P-GP+SMAC', 'P-Div')              

        return label + postfix
    elif 'RANDOM' in label:
        return 'Random' + postfix

    return label + postfix


def get_style(arm, all_items):
    markers = ['o', 'p', '*', '^', 's', 'D', 'x', '<', '.', 'v',
               '>', '+', '1', '2', '3', 'P', '4', 'H', '8', 'd']
    marker_colors = ['xkcd:brown', 'xkcd:purple', 'xkcd:violet', 
                     'xkcd:green', 'xkcd:lime green', 'xkcd:teal', 
                     'xkcd:magenta', 'xkcd:mustard', 'xkcd:orange', 
                     'xkcd:red', 'xkcd:pink', 'xkcd:yellow',                      
                     'xkcd:peach', 'xkcd:lavender', 'xkcd:fuchsia',
                     'xkcd:goldenrod', 'xkcd:light green', 'xkcd:leaf green', 
                     'xkcd:deep purple', 'xkcd:sage']
        
    if 'DIV' in arm:
        line_style = '-'
    else:
        line_style = '--'
        arm = arm.replace('+', '_')
    try:
        index = list(all_items).index(arm)
    except:
        index = 0

    return markers[index], marker_colors[index], line_style


def get_predefined_style(name):
    marker = ''
    color = 'black'
    palette = ['gray', 'xkcd:red', 'xkcd:deep blue', 'xkcd:periwinkle']
    line_style = '-'
    markers = ['', 'p', '^', '*', 's', 'v', 'D', '<', '>',
               '1', '3', '2', '4', '8', "|", "_", '', ",", 'H', '+', 'P', ',', 'h', 'x']

    marker_index = 0


    if 'DEEP-BO' in name:
        line_style = '-'
        color = 'xkcd:red'
        if 'S-Div' in name:            
            marker = 'x'
        elif 'R-Div' in name:
            marker = 'o'
        elif 'P-Div' in name:
            #line_style = '--'
            if '-R' in name:
                marker = '*'  
                color = 'gray'
            elif '-N' in name:
                marker = 'd'
                color = 'orange'
            elif '-P' in name:
                line_style = '-'
                marker = 'o'
                color = 'red'
            else:
                marker = 'x'
        elif 'x6-Div' in name:
            marker_index += 5            
        else:
            #color = 'xkcd:violet'
            #line_style = ':'
            if 'xN-Div' in name:
                marker = 'D'
            elif 'xN-Div-I' in name:
                marker = '*'
    elif 'Diversif' in name:
        line_style = '-'
        color = 'xkcd:red'        
    elif 'Hedge' in name:
        line_style = '-'
        color = palette[2]
        marker = ''
        if "(k=3)" in name:
            marker = '^'
        elif "(k=9)" in name:
            marker = 'v'
    elif '-greedy' in name:
        color = palette[2]
        marker = 's'
    elif 'Random' in name:
        color = 'gray'
        line_style = '-'
    elif 'Ind-Avg' == name:
        line_style = ':'
    elif 'Knockout' in name:
        line_style = '--'
    elif 'BOHB' in name:
        color = palette[3]
        #marker = 's'

    if 'GP-' in name and not 'GP-Hedge' in name:
        # thin blues
        palette = ['xkcd:royal blue', 'xkcd:bright blue',
                'xkcd:baby blue', 'xkcd:sky blue']
        color = palette[0]
        line_style = '-.'
        
        if '-MCMC10' in name:
            marker_index += 1
        elif '-MCMC1' in name:
            marker_index += 2
        marker = markers[marker_index]

    elif 'RF-' in name:
        # thick greens
        palette = ['xkcd:forest green',
                'xkcd:green', 'xkcd:olive', 'xkcd:teal']
        color = palette[0]
        line_style = '--'
        marker = markers[marker_index]
    elif 'TPE' in name:
        line_style = ':'

    if '-EI' in name:
        color = palette[1]
    elif '-PI' in name:
        color = palette[2]
    elif '-UCB' in name:
        color = palette[3]

    if 'P-GP-' in name or 'P-RF-' in name or 'P-Div-' in name:
        #marker_index += 3
        if 'SP-' in name:
            marker_index += 3        
            marker = markers[marker_index]

    if '(baseline' in name:
        line_style = ':'

    if '(surrogate' in name:
        marker = '8'
        line_style = '--'

    if '-LCE' in name:
        #marker = 'd'
        line_style = '-'

    if '-CR' in name:
        line_style = '-'

    if '-MSR' in name:
        #marker = 'o'
        line_style = '-'

    if '(naive' in name:
        marker = '^'
        #color = 'xkcd:royal blue'
    elif '(log' in name:
        marker = '*'
        #color = 'xkcd:royal blue'
    elif '(hybrid' in name:
        marker = 'o'
        #color = 'black'
    elif '(baseline' in name:
        marker = ''
        #color = 'black'                 
    if 'β=0.1' in name:
        color = 'xkcd:orange'
        marker = '*'  
    elif 'β=0.25' in name:
        color = 'xkcd:orange'
        marker = 'd' 
    elif 'β=0.2' in name:
        color = 'xkcd:orange'
        marker = 'v'
    elif 'β=0.05' in name:
        color = 'xkcd:orange'
        marker = '|'   

    if 'fantasy' in name:
        marker = '*'
        line_style = ':'

    return marker, color, line_style


def test_style():
    name = get_label('P-Div-P(6)')
    marker, color, line_style = get_predefined_style(name)
    print("{}, {}, {}, {}".format(name, marker, color, line_style))


if __name__ == '__main__':
    test_style()
