from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

import math

import copy
import analysis as anal
from viz_util import *

from pylab import rcParams
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import scipy.stats as sps


def debug(args):
    #print(args)
    pass


def draw_catastrophic_failures(results, target_goal,
                               num_runs=50, criteria='Hours', x_max=25, step_size=1,
                               title=None, indi=None, div=None, ada=None,
                               save_name=None, target_folder='../../../figs/',
                               indi_max=None, div_max=None, ada_max=None,
                               indi_scale=1, div_scale=1, ada_scale=1, style_format=None,
                               avgs=None, subgroups=None,
                               width=10, height=6, legend=None):
    opt_iterations = {}
    opt_failures = {}

    x = []
    x = range(0, x_max, step_size)

    opts = list(sorted(results.keys()))

    if style_format is None:
        style_format = get_predefined_style


    for opt in list(sorted(results.keys())):
        x_values = None
        if criteria is 'Iterations':
            x_values = np.array(anal.get_num_iters_over_threshold(
                results[opt], num_runs, target_goal))
        else:
            x_values = np.array(anal.get_exec_times_over_threshold(
                results[opt], num_runs, target_goal, unit=criteria))

        opt_iterations[opt] = x_values
        failures = []
        for i in x:
            failure = (x_values[x_values > i].shape[0] / float(num_runs))
            failures.append(failure)

        opt_failures[opt] = failures

    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if avgs is not None:
        # calculate average of each arm's performance
        n_avgs = len(avgs)
        avg_failures = []
        for avg in sorted(avgs):
            fail_list_size = len(opt_failures[avg])
            if len(avg_failures) == 0:
                avg_failures = [0.0 for i in range(fail_list_size)]
            for i in range(fail_list_size):
                f = opt_failures[avg][i]
                if f >= 0:
                    avg_failures[i] += (f / n_avgs)

        # Average plot of Bayesian optimizations
        name = 'Ind-Avg'
        marker, color, line_style = style_format(name)
        fr = opt_failures[opt]
        for t in range(x_max):
            x_ = x
        ax.plot(x_, fr,
                marker=marker, color=color, linestyle=linestyle, label=get_label(opt))

    if subgroups is not None:
        for g in subgroups:
            if 'opt' in g:
                opt = g['opt']
                name = opt
                marker, color, linestyle = style_format(get_label(opt))
                fr = opt_failures[opt]
                x_ = x
                _max = x_max

                _scale = 1
                if 'scale' in g:
                    _scale = g['scale']
                    # linestyle=':'
                    name += ' [scaling]'

                if _scale * _max > x_max:
                    _max = int(x_max / _scale) + 1

                x_ = [x * _scale for x in range(0, _max, step_size)]
                fr = np.asarray(opt_failures[opt])

                max_index = min(len(x_), len(fr))
                if criteria != 'Iterations' and 'max_hour' in g:
                    max_index = g['max_hour'] + 1
                    if criteria == '10 mins':
                        max_index = (max_index - 1) * 6 + 1

                if fr.ndim == 1:
                    if x_max < max_index:
                        max_index = x_max
                    if len(x_) < max_index:
                        max_index = len(x_)
                    fr = fr.tolist()
                    debug("x: {}, y: {}, x_max: {}, max_index: {}".format(
                        len(x_), len(fr), x_max, max_index))

                    ax.plot(x_[:max_index], fr[:max_index],
                            marker=marker, color=color, linestyle=linestyle, label=get_label(name))
                else:
                    for j in range(fr.ndim):
                        ax.plot(x_[:max_index], fr[:max_index, j],
                                marker=marker, color=color, linestyle=linestyle, label=get_label(name))

    if indi is not None:
        for opt in indi:
            marker, color, linestyle = style_format(get_label(opt))
            fr = opt_failures[opt]
            x_ = x
            if indi_max is not None:
                x_ = range(0, indi_max, step_size)
                if indi_scale > 1:
                    if indi_scale * indi_max > x_max:
                        indi_max = int(x_max / indi_scale) + 1
                    x_ = [x * indi_scale for x in range(0, indi_max, step_size)]
                fr = opt_failures[opt][:indi_max]
            ax.plot(x_, fr,
                    marker=marker, color=color, linestyle=linestyle, label=get_label(opt))

    if div is not None:
        for opt in div:
            marker, color, linestyle = style_format(get_label(opt))
            fr = opt_failures[opt]
            x_ = x
            if div_max is not None:
                x_ = range(0, div_max, step_size)
                if div_scale > 1:
                    if div_scale * div_max > x_max:
                        div_max = int(x_max / div_scale) + 1
                    x_ = [x * div_scale for x in range(0, div_max, step_size)]
                    #linestyle = ':'

                fr = opt_failures[opt][:div_max]
            ax.plot(x_, fr,
                    marker=marker, color=color, linestyle=linestyle, label=get_label(opt))

    if ada is not None:
        for opt in ada:
            marker, color, linestyle = style_format(get_label(opt))
            fr = opt_failures[opt]
            x_ = x
            if ada_max is not None:
                x_ = range(0, div_max, step_size)
                fr = opt_failures[opt][:div_max]

            ax.plot(x_, fr,
                    marker=marker, color=color, linestyle=linestyle, label=get_label(opt))
    ax.grid(alpha=0.5)

    if criteria == "10 mins":
        subset_x = []
        for i in range(len(x)):
            if i % 6 == 0:
                subset_x.append(x[i])

        ax.set_xticks(subset_x)
    else:
        ax.set_xticks(x)

    minor_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.3)

    if title is not None:
        ax.set_title(title)

    plt.xlabel("x {}".format(criteria), size=20)
    plt.ylabel("Failure Probability", fontsize=20)
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    if legend is not None:

        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']

    plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)

    if save_name is not None:
        plt.tight_layout()

        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)

    else:
        return plt


def draw_success_rate_fig(results, target_goal, x_max, 
                        num_runs=None, x_unit='Hour', step_size=1,
                        title=None, indi=None, div=None, ada=None,
                        save_name=None, target_folder='.',
                        indi_max=None, div_max=None, ada_max=None,
                        indi_scale=1, div_scale=1, ada_scale=1, name_map=None,
                        avgs=None, parallel=None, l_order=None,
                        width=10, height=6, legend=None, style_format=None,
                        show_marginal_best=False):
    opt_iterations = {}
    opt_successes = {}
    opts = list(sorted(results.keys()))

    x = range(0, x_max, step_size)

    if name_map is None:
        def map_names(name):
            return name
        name_map = map_names
    
    if num_runs == None:
        num_runs = len(results[opts[0]].keys())

    if style_format is None:
        style_format = get_predefined_style

    for opt in list(sorted(results.keys())):
        x_values = None
        if x_unit is 'Iteration':
            x_values = np.array(anal.get_num_iters_over_threshold(
                results[opt], num_runs, target_goal))
        else:
            x_values = np.array(anal.get_exec_times_over_threshold(
                results[opt], num_runs, target_goal, unit=x_unit))
        debug("x_values: {}".format(x_values[0]))
        opt_iterations[opt] = x_values
        successes = []
        for i in x:
            failure = (x_values[x_values > i].shape[0] / float(num_runs))
            success = 1.0 - failure
            successes.append(success)

        opt_successes[opt] = successes

    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(alpha=0.5)

    if parallel is not None:
        for g in parallel:
            if 'opt' in g:
                opt = g['opt']
                name = opt
                marker, color, linestyle = style_format(get_label(opt))
                sr = opt_successes[opt]
                x_ = x
                _max = x_max

                _scale = 1
                if 'scale' in g:
                    _scale = g['scale']
                    # linestyle=':'
                    name += ' [scaling]'

                if _scale * _max > x_max:
                    _max = int(x_max / _scale) + 1

                x_ = [x * _scale for x in range(0, _max, step_size)]
                sr = np.asarray(opt_successes[opt])

                max_index = min(len(x_), len(sr))
                if x_unit == 'Hour' and 'max_hour' in g:
                    max_index = g['max_hour'] + 1

                if sr.ndim == 1:
                    if x_max < max_index:
                        max_index = x_max
                    if len(x_) < max_index:
                        max_index = len(x_)
                    sr = sr.tolist()
                    debug("x: {}, y: {}, x_max: {}, max_index: {}".format(
                        len(x_), len(sr), x_max, max_index))

                    ax.plot(x_[:max_index], sr[:max_index],
                            marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))
                else:
                    for j in range(sr.ndim):
                        ax.plot(x_[:max_index], sr[:max_index, j],
                                marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))


    if indi is not None:
        for opt in indi:
            marker, color, linestyle = style_format(get_label(opt))
            sr = opt_successes[opt]
            x_ = x
            if indi_max is not None:
                x_ = range(0, indi_max, step_size)
                if indi_scale > 1:
                    if indi_scale * indi_max > x_max:
                        indi_max = int(x_max / indi_scale) + 1
                    x_ = [x * indi_scale for x in range(0, indi_max, step_size)]
                sr = opt_successes[opt][:indi_max]
            ax.plot(x_, sr,
                    marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))

    if ada is not None:
        for opt in ada:
            marker, color, linestyle = style_format(get_label(opt))
            sr = opt_successes[opt]
            x_ = x
            if ada_max is not None:
                x_ = range(0, div_max, step_size)
                sr = opt_successes[opt][:div_max]

            ax.plot(x_, sr,
                    marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))
                
    if avgs is not None and show_marginal_best:
        best_failures = []
        opt = 'xN-Div-I'
        for avg in sorted(avgs):
            list_size = len(opt_successes[avg])
            if len(best_failures) == 0:                
                best_failures = [ 1.0 for i in range(list_size)]
            #print("{}:{}".format(avg, opt_successes[avg]))
            for i in range(list_size):
                s = opt_successes[avg][i]
                f = 1.0 - s
                best_failures[i] *= f
        best_successes = []
        for bf in best_failures:
            best_successes.append(1.0 - bf)

        marker, color, linestyle = style_format(get_label(opt))
        ax.plot(x, best_successes, 
                marker='*', color=color, linestyle=linestyle, label=name_map(get_label(opt)))                   

    if div is not None:
        for opt in div:
            marker, color, linestyle = style_format(get_label(opt))
            sr = opt_successes[opt]
            x_ = x
            if div_max is not None:
                x_ = range(0, div_max, step_size)
                if div_scale > 1:
                    if div_scale * div_max > x_max:
                        div_max = int(x_max / div_scale) + 1
                    x_ = [x * div_scale for x in range(0, div_max, step_size)]
                    #linestyle = ':'

                sr = opt_successes[opt][:div_max]
            ax.plot(x_, sr,
                    marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))

    if x_unit == "10min":
        subset_x = []
        for i in range(len(x)):
            if i % 6 == 0:
                subset_x.append(x[i])
        ax.set_xticks(subset_x)
        xlabels = [ x * 10 for x in subset_x ]
        ax.set_xticklabels(xlabels)
        x_unit = 'Minute'
    else:
        ax.set_xticks(x)

    minor_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.3)

    if title is not None:
        ax.set_title(title)

    plt.xlabel("{}".format(x_unit), size=20)
    plt.ylabel("Success rate", fontsize=20)
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    if legend is not None:

        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']
    
    if l_order is not None:
        handles, labels = ax.get_legend_handles_labels()        
        plt.legend([handles[idx] for idx in l_order], [labels[idx] for idx in l_order],
        prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)        
    else:
        plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        return plt


def draw_bar_catastrophic_failures(results, target_goal,
                                   percentiles=[0.7, 0.85, 0.9], criteria='Hours', num_iters=50,
                                   title=None, opts=None, save_name=None, style_format=None,
                                   width=14, height=8):

    means = {}
    std_devs = {}
    if opts is None:
        opts = list(sorted(results.keys()))
    for opt in opts:
        mean_list, std_dev_list = anal.get_percentile(
            results[opt], target_goal, num_iters, percentiles, criteria=criteria)
        means[opt] = mean_list
        std_devs[opt] = std_dev_list

    n_groups = 3
    rcParams['figure.figsize'] = width, height

    if style_format is None:
        style_format = get_predefined_style

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.85 / float(len(opts))
    opacity = 0.8
    for i in range((len(opts))):
        marker, color, linestyle = style_format(get_label(opt))
        rects = plt.bar(index + i * bar_width,
                        means[opts[i]], bar_width,
                        color=color,
                        yerr=std_devs[opts[i]],
                        alpha=opacity, label=get_label(opts[i]))

    #plt.xlabel('goal accuracy: {}'.format(target_goal))
    plt.ylabel(criteria, fontsize=20)
    xticks = []
    for percentile in percentiles:
        xticks.append('{} percentile'.format(int(percentile * 100)))
    plt.xticks(index + bar_width * (len(opts) / 2),
               set(xticks))
    plt.legend()

    if title is not None:
        plt.title(title)

    if save_name is not None:
        plt.tight_layout()
        target_folder = '../../../figs/'
        plt.savefig(target_folder + save_name, format='png')

    return plt


def draw_boxplot_strategies(results, threshold,
                            y_max=24, num_runs=50, criteria='Hours',
                            title=None, opts=None, save_name=None,
                            width=10, height=6, target_folder='../../../figs/'):

    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')

    data = []
    indexes = []
    index = 0
    if opts is None:
        opts = list(sorted(results.keys()))
    for opt in opts:
        value = 0
        if criteria == 'Trials':
            value = anal.get_num_iters_over_threshold(
                results[opt], num_runs, threshold)
        else:
            value = anal.get_exec_times_over_threshold(
                results[opt], num_runs, threshold, unit=criteria)
        data.append(value)
        index += 1
        indexes.append(index)
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    plot = fig.add_subplot(111)
    plot.tick_params(axis='x', which='major', labelsize=20)
    plot.tick_params(axis='y', which='major', labelsize=12)
    plot.boxplot(data, 0, 'gD', meanprops=meanlineprops,
                 meanline=True, showmeans=True)
    names = []
    for opt in opts:
        names.append(get_label(opt))
    plot.set_xticks(indexes)
    plot.set_xticklabels(names, rotation=45)
    plot.set_ylim(0, y_max + 1)

    if title is not None:
        plot.set_title(title)

    #plot.xlabel('model acquisition function')
    plot.set_ylabel('Hours', fontsize=20)

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)


def draw_trials_curve(results, arm, run_index,
                      x_unit='Hour', guidelines=[], g_best_acc=None,
                      xlim=None, ylim=None, title=None, save_name=None, 
                      loc=3, width=10, height=6):
    selected = anal.get_result(results, arm, run_index)
    x_time = anal.get_total_times(selected, x_unit)
    y_errors = selected['error']
    
    if g_best_acc != None:
        g_best_err = 1.0 - g_best_acc
        y_errors = y_errors - g_best_err 

    max_err = 1.0
    line_best_errors = np.array(anal.get_best_errors(selected))
    if g_best_acc != None:
        line_best_errors = line_best_errors - g_best_err
        max_err = g_best_acc
    preamble = np.array([max_err])
    line_best_errors = np.concatenate((preamble, line_best_errors))
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    plot_func = subplot.semilogy
    if g_best_acc != None:
        plot_func = subplot.plot 

    available_arms = set([arm])
    #marker, color, linestyle = get_style(arm, results.keys())
    if 'select_trace' in selected:
        available_arms = set(selected['select_trace'])
    unlabeled_arms = set([arm])
    if len(available_arms) > 0:
        unlabeled_arms = copy.copy(available_arms)
    available_arms = list(sorted(available_arms))
    color = 'black'
    marker = '.'
    opacity = 1.0
    default_marker_size = 6.0
    marker_size = default_marker_size * opacity
    
    for i in range(len(x_time)):
        if len(available_arms) > 0:
            if 'select_trace' in selected:            
                arm = selected['select_trace'][i]

            marker, color, _ = get_style(arm, available_arms)
            if 'train_epoch' in selected:
                opacity = float(selected['train_epoch'][i] / max(selected['train_epoch']))
                marker_size = float(default_marker_size * opacity * 1.2)
                marker = 'o' # XXX: set same marker

            if 'model_index' in selected:
                if selected['model_index'][i] < 0:
                    color = 'red'

            if "epoch" in selected:
                if selected['epoch'][i] <= 3:
                    marker = '.'
                elif selected['epoch'][i] < 10:
                    marker = '*'
                else:
                    marker = 'o'

        if arm in unlabeled_arms:
            plot_func(
                x_time[i], y_errors[i], 
                color=color, linestyle='', alpha=opacity,
                marker=marker, markersize=marker_size,
                label=get_label(arm))
            unlabeled_arms.remove(arm)
        else:
            plot_func(x_time[i], y_errors[i],
                             color=color, linestyle='', alpha=opacity,
                             marker=marker, markersize=marker_size)

    # line plot for best error
    plot_func([0] + x_time, line_best_errors, color='blue',
                     linestyle='--', label='best error')

    if title is not None:
        subplot.set_title(title)
    if ylim is None:
        plt.ylim(ymax=1.0)
    else:
        plt.ylim(ylim)
    x_range = [0, 0]
    if xlim is not None:
        plt.xlim(xlim)
        x_range = list(xlim)
        x_ticks = [x for x in range(x_range[0], x_range[-1] + 1)]
        plt.xticks(x_ticks)

    for s in guidelines:
        label = ""
        pos = s['error']
        if g_best_acc != None and "regret" in s:
            pos = s['regret']

        if "rank" in s:
            label = "Top-{}".format(s['rank'])
        elif 'difficulty' in s:
            label = "Top {:.2f}%".format(s['difficulty']*100)

        plt.text(x_range[-1] + 0.1, pos, label, size=12)
        plt.axhline(y=pos, color='gray', linestyle=':')

    if g_best_acc != None:
        plt.ylabel("Intermidiate regret", fontsize=15)
    else:
        plt.ylabel("Test error", fontsize=15)
    plt.xlabel(x_unit, size=15)
    plt.legend(loc=loc, prop={'size': 15})

    if save_name is not None:
        # plt.tight_layout()
        target_folder = '../../../figs/'
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        plt.show()


def add_error_fill_line(x, y, yerr, color=None, linestyle='-',
                        alpha_fill=0.3, ax=None, label=None, marker=None, y_scale=1.0):
    #ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr

    ymin = np.maximum(y_scale * 0.001, ymin)
    ymax = np.minimum(y_scale * 1.0, ymax)
    debug("ymin: {}".format(ymin[0]))
    debug("ymax: {}".format(ymax[-1]))
    ax.semilogy(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def draw_best_error_curve(results, arms, repeats,
                          guidelines=[], summary=False, title=None, x_unit="Hour",
                          xlim=None, ylim=(.001, 1), alpha_fill=0.1, std_div=4,
                          width=14, height=8, x_steps=1, plot_func='semilogy',
                          save_name=None, target_folder='.', y_scale=1,
                          x_ticks=None, y_ticks=None, sub_y_metric="test error",
                          legend=None, l_order=None, style_format=None):

    if type(arms) is not list:
        arms = [arms]

    unlabeled_arms = set(arms)
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    subplot.grid(alpha=0.5)
    t_max = None

    

    for arm in arms:
        if not arm in results.keys():
            raise ValueError(
                "results log do not have record for {}".format(arm))
        if style_format == None:
            style_format = get_predefined_style

        marker, color, linestyle = style_format(arm)
        if summary is False:
            best_errors = []
            for i in range(repeats):
                selected = anal.get_result(results, arm, i)
                x_time = anal.get_total_times(selected, x_unit)
                y_best_errors = np.array(anal.get_best_errors(selected)) * y_scale
                best_errors.append({'x': x_time, 'y': y_best_errors.tolist() })

            pfunc = subplot.plot
            if plot_func == 'semilogy':
                pfunc = subplot.semilogy
            elif plot_func == 'loglog':
                pfunc = subplot.loglog
                subplot.set_xscale('log',  nonposx='clip')
            for best_error in best_errors:
                if arm in unlabeled_arms:
                    pfunc([0] + best_error['x'], 
                        ([1.0] + best_error['y']), 
                        color=color, linestyle=linestyle, label=arm, marker=marker)
                    unlabeled_arms.remove(arm)
                else:
                    pfunc([0] + best_error['x'], ([1.0] + best_error['y']), color=color, linestyle=linestyle, marker=marker)
        else:
            errors_by_interval = { }            
            subplot.set_yscale('log')
            if plot_func == 'semilogy':
                subplot.set_yscale('log')
            elif plot_func == 'loglog':
                subplot.set_yscale('log')
                subplot.set_xscale('log')
            
            for i in range(repeats):

                selected = anal.get_result(results, arm, i)
                y_best_errors = np.array(anal.get_best_errors(selected)) * y_scale
                t_times = anal.get_total_times(selected, x_unit)
                t_max = int(max(t_times)) + 1
                
                if i == 0:
                    debug("t_times: {}".format(t_times[0]))
                    debug("y_best_errors: {}".format(y_best_errors[0]))

                t = 0
                cur_best_err = 1.0 * y_scale
                for j in range(len(t_times)):
                    
                    cur_time = t_times[j]
                    
                    while t < cur_time:
                        if not t in errors_by_interval.keys():
                            errors_by_interval[t] = []                          
                        errors_by_interval[t].append(cur_best_err)
                        t += 1

                    if y_best_errors[j] < cur_best_err:
                        cur_best_err = y_best_errors[j] 

                if i == 0:
                    debug("errors_by_interval: {}".format(errors_by_interval[0]))
                        
                        
            x = np.array([])
            y = np.array([]) 
            yerr = np.array([])
            
            for i in range(0, t_max):
                errors = errors_by_interval[i] 

                y = np.append(y, np.mean(errors))
                yerr = np.append(yerr, np.std(errors)/std_div)
                x = np.append(x, i)
                
            
            if arm in unlabeled_arms:
                add_error_fill_line(x, y, yerr, color, ax=subplot, marker=marker, y_scale=y_scale,
                                    label=arm, linestyle=linestyle, alpha_fill=alpha_fill)
                unlabeled_arms.remove(arm)
            else:
                add_error_fill_line(x, y, yerr, color, marker=marker, y_scale=y_scale,
                    ax=subplot, linestyle=linestyle, alpha_fill=alpha_fill)
    if title != None:
        subplot.set_title('{}'.format(title))
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    
    if legend is not None:

        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']
    
    if l_order is not None:
        handles, labels = subplot.get_legend_handles_labels()        
        plt.legend([handles[idx] for idx in l_order], [labels[idx] for idx in l_order],
        prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)        
    else:
        plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)
    x_range = [0, 0]
    
    if x_ticks == None and xlim is not None:
        plt.xlim(xlim)
        x_range = list(xlim)
        x_ticks = [x for x in range(x_range[0], x_range[-1] + 1, x_steps)]
    
    if x_ticks != None:
        plt.xticks(x_ticks)

    if y_ticks != None:
        plt.yticks(y_ticks)

    if ylim is not None:
        plt.ylim(ylim)

    for s in guidelines:
        label = ""
        if "label" in s:
            label = s['label']
        elif "rank" in s:
            label = "Top-{}".format(s['rank'])
        elif 'difficulty' in s:
            label = "Top {:.2f}%".format(s['difficulty']*100)
        
        if 'perplexity' in s:
            plt.text(x_range[0] + .1, s['perplexity'], label)
            plt.axhline(y=s['perplexity'], color='gray', linestyle=':')
        else:
            plt.text(x_range[0] + .1, s['error'], label)
            plt.axhline(y=s['error'], color='gray', linestyle=':')

    plt.ylabel("Min Function Value ({})".format(sub_y_metric), fontsize=15)
    plt.xlabel(x_unit, fontsize=15)
    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        plt.show()
        return plt


def draw_mean_sd_corr(opt, estimates, results, num_trial,
                      lookup=None, top_p=0.1, loc='best', m_size=2, dpi=72,
                      start_index=0, use_rank=True, save_name=None,
                      n_row=None, n_col=4, span=7, it_list=None, xlim=None, ylim=None):
    est_opt = estimates[opt]
    result = results[opt]

    t = str(num_trial)

    if t in est_opt.keys():
        num_exploits = 0
        num_explores = 0

        if n_row is None:
            if type(it_list) is list and len(it_list) > n_col:
                n_row = int(len(it_list) / n_col)
            else:
                n_row = 1

        fig = plt.figure(num=None, figsize=(n_col * 6, n_row * 6),
                         dpi=dpi, facecolor='w', edgecolor='k')
        #fig.suptitle('{}, trial {}'.format(opt, it))

        fig_index = 1
        trials = est_opt[t]
        iter = {'i': int(t) + start_index, 'trials': []}
        cur_best_acc = 0.0
        if lookup is not None:
            top_ranks = lookup.nlargest(
                int(len(lookup) * 0.01 * top_p), 'best_acc')  # .rank()#.sort_values()
            top_min = top_ranks['best_acc'].min()
            n_rankers = len(top_ranks)

        for i in range(len(trials)):
            trial = {}
            trial['step'] = i + 1
            goal_achieved = False

            est = trials[i]['estimated_values']
            arm = opt
            if opt.find('DIVERSIFIED') >= 0:
                arm = result[t]['select_trace'][i]
            trial['arm'] = arm
            trial['cur_acc'] = result[t]['accuracy'][i]
            best_errors = anal.get_best_errors(result[t])
            trial['cum_op_time'] = sum(result[t]['exec_time'][:i+1]) + \
                sum(result[t]['opt_time'][:i+1])
            op_hours = trial['cum_op_time'] / (60 * 60)

            if trial['cur_acc'] > cur_best_acc:
                cur_best_acc = trial['cur_acc']
                debug('current best {:.4f} updated at iteration {}.'.format(
                    cur_best_acc, i))
                if cur_best_acc >= top_min:
                    goal_achieved = True
                    debug('target to achieve top {} percent completed after {:.2f} hours'.format(
                        top_p, op_hours))

            if est is None:
                num_explores += 1
            else:
                best_cand = np.argmax(est['acq_funcs'])

                next_index = int(est['candidates'][best_cand])
                total_cand = len(est['candidates'])

                if type(it_list) is list and len(it_list) > 0:
                    plot_cond = (i > 1 and i in it_list)
                else:
                    plot_cond = (i > 1 and i <= n_col *
                                 n_row * span and i % span == 0)

                if plot_cond:

                    subplot = fig.add_subplot(n_row, n_col, fig_index)
                    scaled_acq = np.asarray(
                        est['acq_funcs']) / np.asarray(est['acq_funcs'][best_cand])
                    if use_rank:
                        scaled_acq = (
                            rankdata(est['acq_funcs']) - 1) / len(est['acq_funcs'])
                    means = np.asarray(est['means'])
                    vars = np.asarray(est['vars'])
                    sds = np.sqrt(vars)
                    color = 'gray'
                    alpha = 0.1
                    marker = 'x'
                    plt.scatter(sds, means, alpha=alpha,
                                color=color, marker=marker)
                    labeled = False
                    for j in range(len(scaled_acq)):
                        top_k_rate = 1.0 - (0.01 * top_p)
                        if scaled_acq[j] > top_k_rate:
                            color = 'orange'
                            alpha = (scaled_acq[j] - top_k_rate) * \
                                (1.0 / (0.01 * top_p))
                            marker = 'o'
                            if labeled:
                                plt.scatter(sds[j], means[j],
                                            alpha=alpha, color=color, marker=marker, s=m_size * 10)
                            elif alpha >= 0.5:
                                labeled = True
                                plt.scatter(sds[j], means[j],
                                            alpha=alpha, color=color, marker=marker,
                                            label='Est. top-{}'.format(n_rankers), s=m_size * 10)

                    plt.plot(sds[best_cand], means[best_cand], 'ro',
                             label='Est. top 1', markersize=m_size)
                    cur_best_err = best_errors[i]
                    #debug('{}, current best error: {}'.format(i, cur_best_err))
                    #plt.plot(cur_best_err, 0.0,  'gD', label='current best error')
                    #plt.axhline(np.amax(sds), linestyle='--', color='yellow', label='max sd')

                    if lookup is not None:
                        labeled = False
                        # pin points of top rankers
                        debug(n_rankers)
                        ik = n_rankers
                        for k in top_ranks.index.values:
                            debug("{} exists? : {}".format(
                                k, (k in est['candidates'])))
                            if k in est['candidates']:
                                l = est['candidates'][k]
                                alpha = ik * 1.0 / float(n_rankers)
                                ik -= 1
                                color = 'blue'
                                marker = '*'
                                if labeled:
                                    plt.scatter(sds[l], means[l],  alpha=alpha, color=color, marker=marker,
                                                s=m_size * 10)
                                elif alpha >= 0.9:
                                    labeled = True
                                    plt.scatter(sds[l], means[l],  alpha=alpha, color=color, marker=marker,
                                                label='True top-{}'.format(n_rankers), s=m_size * 10)

                    if (n_col * n_row) == 1 or fig_index % n_col == 1:
                        plt.legend(loc=loc, prop={'size': 18})
                    else:
                        #debug('no legend: {} {} {}'.format(n_col, n_row, fig_index))
                        pass

                    #debug("{} - acq best: {}, selected: {}".format(i, np.amax(est['acq_funcs']), est['acq_funcs'][best_cand]))
                    if xlim is not None:
                        plt.xlim(xlim)
                    if ylim is not None:
                        plt.ylim(ylim)

                    if fig_index % n_col == 1:
                        plt.ylabel("Est. mean", fontsize=18)

                    if fig_index - (n_row - 1) * n_col > 0:
                        plt.xlabel(" Est. Std", fontsize=18)
                    achieve = ""
                    if goal_achieved:
                        achieve = "achieves goal"
                    # debug(get_label(arm))
                    plt.title("{} ({:.1f} hr)".format(
                        get_label(trial['arm']), op_hours), fontsize=25)
                    fig_index += 1
    else:
        raise ValueError('no such iteration existed: {}'.format(trial))

    if save_name is not None:
        plt.tight_layout()
        target_folder = '../../../figs/'
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=dpi)
    else:
        return plt


def draw_acq_values(opt, estimates, results, it,
                    start_index=0, use_rank=True, n_row=5, n_col=2, span=7, dpi=72):
    est_opt = estimates[opt]
    result = results[opt]

    it = str(it)

    if it in est_opt.keys():
        num_exploits = 0
        num_explores = 0

        fig = plt.figure(num=None, figsize=(30, n_row * 6),
                         dpi=dpi, facecolor='w', edgecolor='k')
        fig.suptitle('{}, trial {}'.format(opt, it))
        fig_index = 1
        trials = est_opt[it]
        iter = {'i': int(it) + start_index, 'trials': []}
        cur_best_acc = 0.0

        for i in range(len(trials)):
            trial = {}
            trial['step'] = i + 1

            est = trials[i]['estimated_values']
            arm = opt
            if opt.find('DIVERSIFIED') >= 0:
                arm = result[it]['select_trace'][i]
            trial['arm'] = arm
            trial['cur_acc'] = result[it]['accuracy'][i]
            if trial['cur_acc'] > cur_best_acc:
                cur_best_acc = trial['cur_acc']
                debug('current best {} updated at iteration {}.'.format(
                    cur_best_acc, i))
            best_errors = anal.get_best_errors(result[it])
            trial['cum_op_time'] = sum(result[it]['exec_time'][:i+1]) + \
                sum(result[it]['opt_time'][:i+1])
            if est is None:
                num_explores += 1
            else:
                best_cand = np.argmax(est['acq_funcs'])

                next_index = int(est['candidates'][best_cand])
                cand = est['candidates']
                total_cand = len(cand)

                if i > 1 and i <= n_col * n_row * span and i % span == 0:
                    subplot = fig.add_subplot(n_row, n_col, fig_index)
                    scaled_acq = est['acq_funcs'] / est['acq_funcs'][best_cand]
                    if use_rank:
                        scaled_acq = (
                            rankdata(est['acq_funcs']) - 1) / len(est['acq_funcs'])
                    means = est['means']
                    sds = np.sqrt(est['vars'])
                    color = 'blue'
                    alpha = 0.1
                    marker = 'x'
                    #plt.plot(cand, means, alpha=alpha, color=color, marker=marker, label='mean')
                    color = 'red'
                    #plt.plot(cand, sds, alpha=alpha, color=color, marker=marker, label='sd')
                    for j in range(len(scaled_acq)):
                        if scaled_acq[j] > 0.999:
                            color = 'blue'
                            alpha = (scaled_acq[j] - 0.999) * 1000
                            marker = '.'
                            plt.plot(cand[j], means[j],  alpha=alpha,
                                     color=color, marker=marker)
                            color = 'red'
                            plt.plot(cand[j], sds[j],  alpha=alpha,
                                     color=color, marker=marker)

                            color = 'green'
                            func_s = np.sqrt(est['vars'][j]) + 0.0001
                            u = (best_errors[-1] - means[j]) / func_s
                            ncdf = sps.norm.cdf(u)
                            npdf = sps.norm.pdf(u)
                            ei = func_s * (u * ncdf + npdf)
                            plt.plot(cand[j], ei,  alpha=alpha,
                                     color=color, marker='x')

                            color = 'xkcd:brown'
                            func_s = np.sqrt(est['vars'][j]) + 0.0001
                            u = (best_errors[-1] - means[j]) / func_s
                            pi = sps.norm.cdf(u)

                            plt.plot(cand[j], pi,  alpha=alpha,
                                     color=color, marker='x')

                            color = 'xkcd:goldenrod'
                            func_s = np.sqrt(est['vars'][j])
                            ucb = func_s - means[j]

                            plt.plot(cand[j], ucb,  alpha=alpha,
                                     color=color, marker='x')

                            color = 'black'
                            plt.plot(cand[j], est['acq_funcs'][j],
                                     alpha=alpha, color=color, marker='o')

                    plt.plot(cand[best_cand], means[best_cand],
                             'bo', label='mean by acq')
                    plt.plot(cand[best_cand], sds[best_cand],
                             'ro', label='sd by acq')
                    cur_best_err = best_errors[i]
                    #debug('{}, current best error: {}'.format(i, cur_best_err))
                    # plt.plot(, 0.0,  'go', label='current best error')
                    plt.axhline(cur_best_err, linestyle='--',
                                color='yellow', label='current best error')

                    if fig_index % n_col == 1:
                        plt.legend()

                    #debug("{} - acq best: {}, selected: {}".format(i, np.amax(est['acq_funcs']), est['acq_funcs'][best_cand]))
                    plt.ylabel("value", fontsize=10)
                    plt.xlabel("candidate index", fontsize=10)
                    plt.title("{}: {}  iterations".format(arm, i), fontsize=15)
                    fig_index += 1

        fig.show()

