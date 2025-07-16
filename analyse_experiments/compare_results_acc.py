import json
import re

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from definitions import ROOT_DIR
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

from src.utils.argument_parser import Namespace


def run():
    id = f"sr_bench"
    result_dict = {}
    if id == "sr_bench":
        folder_path = 'results/datasets_srbench/dataset_size_300'
        cvgp_path = {
            0: ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalCVGP_27_Feb_2025_21:10:33.json',
            1: ROOT_DIR / folder_path / 'seed_1' / 'num_res10/pysr_i10' / 'normalCVGP_28_Feb_2025_09:24:06.json',
            2: ROOT_DIR / folder_path / 'seed_2' / 'num_res10/pysr_i10' / 'normalCVGP_28_Feb_2025_10:49:41.json',
        }
        path_to_results = {
            0: {
                'NeSymReS': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalnesymres_27_Feb_2025_21:14:35.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_27_Feb_2025_21:15:35.json',
                #'E2E': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalkaminey_01_Mar_2025_23:31:56.json',
                'PySR': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalPySR_27_Feb_2025_21:14:52.json',
                'GPLearn': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalGP_27_Feb_2025_21:14:19.json'
            },
            1: {
                'NeSymReS': ROOT_DIR / folder_path / 'seed_1' / 'num_res10/pysr_i10' / 'normalnesymres_28_Feb_2025_09:24:34.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_1' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_28_Feb_2025_09:25:18.json',
                'PySR': ROOT_DIR / folder_path / 'seed_1' / 'num_res10/pysr_i10' / 'normalPySR_28_Feb_2025_09:24:57.json',
                'GPLearn': ROOT_DIR / folder_path / 'seed_1' / 'num_res10/pysr_i10' / 'normalGP_28_Feb_2025_09:24:19.json'
            },
            2: {
                'NeSymReS': ROOT_DIR / folder_path / 'seed_2' / 'num_res10/pysr_i10' / 'normalnesymres_28_Feb_2025_10:50:28.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_2' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_28_Feb_2025_13:26:35.json',
                'PySR': ROOT_DIR / folder_path / 'seed_2' / 'num_res10/pysr_i10' / 'normalPySR_28_Feb_2025_10:50:58.json',
                'GPLearn': ROOT_DIR / folder_path / 'seed_2' / 'num_res10/pysr_i10' / 'normalGP_28_Feb_2025_10:50:01.json'
            },
        }

    if id == "dso_original":
        folder_path = 'datasets_dso'
        cvgp_path = {
            0: ROOT_DIR / 'results' / folder_path / 'seed_0/CVGP_25_Feb_2025_16:49:11.json',
        }
        path_to_results = {0: {

            # 'E2E': ROOT_DIR / 'results' /folder_path / 'seed_0/kaminey_19_Feb_2025_22:07:35.json',
            'NeSymReS': ROOT_DIR / 'results' / folder_path / 'seed_0/nesymres_25_Feb_2025_15:33:45.json',
            'SymbolicGPT': ROOT_DIR / 'results' / folder_path / 'seed_0/symbolicgpt_25_Feb_2025_15:33:51.json',
            'PySR': ROOT_DIR / 'results' / folder_path / 'seed_0/PySR_25_Feb_2025_22:11:10.json',
            'GPLearn': ROOT_DIR / 'results' / folder_path / 'seed_0/GP_26_Feb_2025_11:30:48.json'
        },
        }
    if id == "dso_1000":
        folder_path = 'results/datasets_dso_1000/dataset_size_300'
        cvgp_path = {
            4: ROOT_DIR / folder_path / 'seed_4' / 'num_res10/pysr_i10' / 'normalCVGP_28_Feb_2025_15:01:39.json',
            5: ROOT_DIR / folder_path / 'seed_5' / 'num_res10/pysr_i10' / 'normalCVGP_01_Mar_2025_08:31:35.json',
            6: ROOT_DIR / folder_path / 'seed_6' / 'num_res10/pysr_i10' / 'normalCVGP_01_Mar_2025_08:32:19.json',
        }
        path_to_results = {
            4: {
                'NeSymReS': ROOT_DIR / folder_path / 'seed_4' / 'num_res10/pysr_i10' / 'normalnesymres_01_Mar_2025_20:29:49.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_4' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_28_Feb_2025_13:27:02.json',
                'PySR': ROOT_DIR / folder_path / 'seed_4' / 'num_res10/pysr_i10' / 'normalPySR_01_Mär_2025_20:31:25.json',
                'GPLearn': ROOT_DIR / folder_path / 'seed_4' / 'num_res10/pysr_i10' / 'normalGP_28_Feb_2025_15:01:19.json'
            },
            5: {
                'NeSymReS': ROOT_DIR / folder_path / 'seed_5' / 'num_res10/pysr_i10' / 'normalNeSymRes_01_Mär_2025_23:55:00.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_5' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_28_Feb_2025_23:55:15.json',
                'PySR': ROOT_DIR / folder_path / 'seed_5' / 'num_res10/pysr_i10' / 'normalPySR_01_Mär_2025_23:33:21.json',
                'GPLearn': ROOT_DIR / folder_path / 'seed_5' / 'num_res10/pysr_i10' / 'normalGP_01_Mar_2025_22:07:24.json'
            },
            6: {
                'NeSymReS': ROOT_DIR / folder_path / 'seed_6' / 'num_res10/pysr_i10' / 'normalnesymres_28_Feb_2025_23:55:36.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_6' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_28_Feb_2025_23:55:52.json',
                'PySR': ROOT_DIR / folder_path / 'seed_6' / 'num_res10/pysr_i10' / 'normalPySR_01_Mär_2025_22:09:06.json',
                'GPLearn': ROOT_DIR / folder_path / 'seed_6' / 'num_res10/pysr_i10' / 'normalGP_01_Mar_2025_22:10:02.json'
            },
        }
    args = Namespace()
    args.mode = 'test'
    args.fill_methods_with_classic = True


    for seed in path_to_results.keys():
        cvgp_results = read_results_of_CVGP(cvgp_path[seed], args)
        result_dict[seed] = {}
        for i, (name, path) in enumerate(path_to_results[seed].items()):
            result_dict[seed][name] = read_results_of_approach(args, path, cvgp_results)

    result_dict, runtime_dict = resort_result_dict(result_dict)

    calculate_statistics(args, result_dict)
    significance_table = calculate_significance(args, result_dict)
    results_df = fill_result_table(result_dict, runtime_dict, significance_table)
    splited_path = path.stem.split('_')
    splited_path[0] = '_'.join(list(result_dict.keys()))
    save_path = path.parent.parent / f"{'_'.join(splited_path)}.tex"
    latex_table = formate_latex_table(results_df, result_dict)
    with open(save_path, "w") as text_file:
        text_file.write(latex_table)
    print(f"table_saved @ {save_path}")

    fill_ax_of_figure(result_dict)
    save_path = path.parent.parent / f"{'_'.join(splited_path)}.pdf"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"figure saved @ {save_path}")


def get_best_run_from_multiple(d, experiment, approach, Mode=None):
    minimum_error = np.inf
    num_op = np.inf
    best_node_id = -1
    runtime = 0
    try:
        approach_dict = d[experiment][approach][Mode]
        for node_id in approach_dict.keys():
            try:
                runtime += d[experiment][approach]['train'][node_id]['time']
            except:
                pass
            if 'error' in approach_dict[node_id]:
                error = approach_dict[node_id]['error']
                if error < minimum_error:
                    minimum_error = error
                    num_op = approach_dict[node_id]['num_operations']
                    best_node_id = node_id
    except:
        return np.nan, np.nan, np.nan, np.nan
    if minimum_error == np.inf:
        return np.nan, np.nan, np.nan, np.nan
    return minimum_error, num_op, best_node_id, runtime


def mean_and_round(number, decimal):
    return np.round(np.nanmean(number), decimal)


def std_and_round(number, decimal):
    return np.round(np.nanstd(number), decimal)


def sort_key(item):
    parts = item.split('_')
    if parts[0] == 'I':
        prefix = 1
    elif parts[0] == 'II':
        prefix = 2
    elif parts[0] == 'III':
        prefix = 3
    else:
        prefix = 4  # For 'test' or other prefixes not covered by the rules
    if len(parts) > 1:
        part_1 = int(parts[1]) if parts[1].isdigit() else 0
    else:
        part_1 = 0
    if len(parts) > 2:
        part_2 = int(parts[2]) if parts[2].isdigit() else 0
    else:
        part_2 = 0

    # Convert the rest of the parts to integers where possible, handle non-integer parts appropriately
    return (prefix, part_1, part_2)


def forward(value):
    result = np.where(value <= 0.1, value * 5, (value - 0.1) * 5 / 9 + 0.5)
    return result


def inverse(value):
    return np.where(value <= 0.5, value / 5, (value - 0.5) * 9 / 5 - 0.1)

def calculate_significance(args, result_dict ):
    # calculate_Wilcoxon_Signed_Rank_Test()
    error_dict = prepare_error_dict(args, result_dict)

    significance_table = {}
    for eds, eds_dict in error_dict.items():
        significance_table[eds] = {}
        for approach, err_vector in eds_dict.items():
            if approach != 'residuals':
                difference = (error_dict[eds]['residuals'] - err_vector)
                cleaned_array = difference[~np.isnan(difference)]
                statistic, pvalue = wilcoxon(cleaned_array, alternative='two-sided')
                #statistic, pvalue = mannwhitneyu(error_dict[eds]['residuals'],  err_vector)

                significance_table[eds][approach] = pvalue
    return significance_table


def prepare_error_dict(args, result_dict):
    error_dict = {}
    for eds, eds_dict in result_dict.items():
        error_dict[eds] = {}
        for approach, approach_dict in eds_dict.items():
            err_matrix = pd.DataFrame(approach_dict['err_filtered']).to_numpy()
            err_vector = err_matrix.reshape(-1)
            if args.fill_methods_with_classic and approach != 'classic':
                classic_error = error_dict[eds]['classic']
                err_vector = fill_approach_with_classic(classic_error, err_vector)
            error_dict[eds][approach] = err_vector
    return error_dict


def fill_approach_with_classic(classic_error, err_vector):
    e = err_vector
    c = classic_error
    e[np.isnan(e) & ~np.isnan(c)] = c[np.isnan(e) & ~np.isnan(c)]
    err_vector = err_vector
    return err_vector


def resort_result_dict(result_dict):
    new_dict = {}
    runtime_dict = {}
    # Iterate through the original dictionary and restructure it
    for k_seed, value1 in result_dict.items():
        for k_finder, value2 in value1.items():
            for k_approach, value3 in value2.items():
                if isinstance(value3, dict):
                    try:
                        for k_stat, value4 in value3.items():
                            if k_finder not in new_dict:
                                new_dict[k_finder] = {}
                            if k_approach not in new_dict[k_finder]:
                                new_dict[k_finder][k_approach] = {}
                            if k_stat not in new_dict[k_finder][k_approach]:
                                new_dict[k_finder][k_approach][k_stat] = {}
                            new_dict[k_finder][k_approach][k_stat][k_seed] = value4
                    except:
                        pass
                else:
                    if k_finder not in runtime_dict:
                        runtime_dict[k_finder] = {}
                    runtime_dict[k_finder][k_seed] = value2['runtime']
    return new_dict, runtime_dict


def read_results_of_CVGP(path, args):
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    if not "finished" in d:
        print(f"{path} not finished yet, number of runs {len(d)}")
    approach_dict = {'CVGP': {'classic': {}}}
    for experiment in d.keys():
        feynman_id = experiment.replace('.tsv.gz', '').replace('feynman_', '')
        error, num_op, best_node_id, runtime = get_best_run_from_multiple(
            d, experiment, 'classic', Mode=args.mode)

        approach_dict['CVGP']['classic'][feynman_id] = {
            'err': error,
            'num_op': num_op,
            'runtime': runtime
        }
    return approach_dict


def read_results_of_approach(args, path, cvgp_results):
    #
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    if not "finished" in d:
        print(f"{path} not finished yet, number of runs {len(d)}")
    approach_dict = create_approach_dict_with_empty_lists()
    for experiment in d.keys():
        if isinstance(d[experiment], dict):
            for approach, sub_d in approach_dict.items():
                if approach == 'CVGP':
                    pass
                error, num_op, best_node_id, runtime = get_best_run_from_multiple(
                    d, experiment, approach, Mode=args.mode)

                sub_d['err'].append(error)
                sub_d['num_op'].append(num_op)
                sub_d['runtime'].append(runtime)
                sub_d['experiment_list'].append(
                    experiment.replace('.tsv.gz', '').
                    replace('feynman_', '')
                )
    if 'CVGP' in cvgp_results:
        d_cvgp = cvgp_results['CVGP']['classic']
        approach_dict['CVGP'] = {
            'err': [d_cvgp[name]['err'] if name in d_cvgp else np.nan for name in approach_dict['residuals']['experiment_list']],
            'num_op': [d_cvgp[name]['num_op'] if name in d_cvgp else np.nan for name in approach_dict['residuals']['experiment_list']],
            'runtime': [d_cvgp[name]['runtime'] if name in d_cvgp else np.nan for name in approach_dict['residuals']['experiment_list']],
            'experiment_list': approach_dict['residuals']['experiment_list']
        }
    approach_dict['runtime'] = d['runtime']
    return approach_dict


def create_approach_dict_with_empty_lists():
    approach_dict = {
        'classic': {'err': [],
                    'num_op': [],
                    'runtime': [],
                    'experiment_list': []
                    },
        'post_gp': {'err': [],
                    'num_op': [],
                    'runtime': [],
                    'experiment_list': []
                    },
        'hyperparameter': {'err': [],
                           'num_op': [],
                           'runtime': [],
                           'experiment_list': []
                           },
        'refit_constants': {'err': [],
                            'num_op': [],
                            'runtime': [],
                            'experiment_list': []
                            },
        'permute': {'err': [],
                    'num_op': [],
                    'runtime': [],
                    'experiment_list': []
                    },
        'CVGP': {'err': [],
                 'num_op': [],
                 'runtime': [],
                 'experiment_list': []
                 },
        'residuals': {'err': [],
                      'num_op': [],
                      'runtime': [],
                      'experiment_list': []
                      },
    }
    return approach_dict


def calculate_statistics(args, result_dic):
    for equation_finder, d_equation_finder in result_dic.items():
        for approach, d_approach in d_equation_finder.items():
            if isinstance(d_approach, dict):
                # only needed for classic approach
                unfiltered_statistics(d_approach)

                # statistics for table
                err_matrix_where_classic_is_grater = filtered_statistics(
                    approach,
                    d_approach,
                    d_equation_finder,
                    args
                )

                statistics_for_plot(d_approach, err_matrix_where_classic_is_grater)


def statistics_for_plot(d_approach, err_matrix_where_classic_is_grater):
    experiment_once_succ = np.any(~np.isnan(err_matrix_where_classic_is_grater), axis=1)
    d_approach['err_filter_mean'] = np.nanmean(err_matrix_where_classic_is_grater, axis=1)[experiment_once_succ]
    d_approach['err_filter_var'] = np.nanvar(err_matrix_where_classic_is_grater, axis=1)[experiment_once_succ]
    k = list(d_approach['experiment_list'].keys())[0]
    d_approach['experiment_list_filtered'] = np.array(d_approach['experiment_list'][k])[experiment_once_succ]


def filtered_statistics(approach, d_approach, d_equation_finder, args):
    greater_threshold = pd.DataFrame(d_equation_finder['classic']['err']).to_numpy() > 0.001
    err_matrix_where_classic_is_grater = pd.DataFrame(d_approach['err']).to_numpy()
    err_matrix_where_classic_is_grater[~greater_threshold] = np.nan
    if args.fill_methods_with_classic and approach != 'classic':
        c = d_equation_finder['classic']['err_filtered']
        err_matrix_where_classic_is_grater = fill_approach_with_classic(
            classic_error=c,
            err_vector=err_matrix_where_classic_is_grater
        )
    d_approach['err_filtered'] = err_matrix_where_classic_is_grater
    err_filter_quantile = np.round(
        np.nanquantile(err_matrix_where_classic_is_grater.flatten(),
                       (0.25, 0.5, 0.75),
                       axis=0),
        3)
    d_approach['err_filter_quantile'] = err_filter_quantile
    d_approach['successful_runs_filter'] = np.mean(
        np.sum(~np.isnan(err_matrix_where_classic_is_grater), axis=0)
    )
    d_approach['runtime_median'] = np.nanquantile(
        np.reshape(pd.DataFrame(d_approach['runtime']).to_numpy()[greater_threshold], -1), (0.5))
    num_opp_filter_list = pd.DataFrame(d_approach['num_op']).to_numpy()[greater_threshold]
    num_opp_filter_quantile = np.round(
        np.nanquantile(num_opp_filter_list.flatten(), (0.25, 0.5, 0.75), axis=0),
        3)
    d_approach['num_op_filter_quantile'] = num_opp_filter_quantile
    return err_matrix_where_classic_is_grater


def unfiltered_statistics(d_approach):
    err_matrix = pd.DataFrame(d_approach['err']).to_numpy()
    err_quantile = np.round(np.nanquantile(err_matrix.flatten(), (0.25, 0.5, 0.75), axis=0), 3)
    d_approach['err_quantile'] = err_quantile
    d_approach['successful_runs'] = np.mean(np.sum(~np.isnan(err_matrix), axis=0))
    num_opp_matrix = pd.DataFrame(d_approach['num_op']).to_numpy()
    num_opp_quantile = np.round(np.nanquantile(num_opp_matrix.flatten(), (0.25, 0.5, 0.75), axis=0), 3)
    d_approach['num_op_quantile'] = num_opp_quantile


def fill_result_table(result_dict, runtime_dict, significance_table):
    table_dict = {}
    for equation_finder, equation_finder_d in result_dict.items():
        # name = f"\\rot[30][3em]{{{equation_finder}}}"
        name = equation_finder
        table_dict[name] = {}
        mean_runtime = np.mean(pd.Series(runtime_dict[equation_finder]).to_numpy())
        table_dict[name]["runtime"] = f"{int(round(mean_runtime))}"
        for approach, d_approach in equation_finder_d.items():  # ['classic', 'residuals', 'post_gp', 'hyperparameter', 'refit_constants', 'permute', 'CVGP']:
            if isinstance(d_approach, dict):
                table_dict[name][f"{approach}"] = " "
                try:
                    if approach == 'classic':
                        table_dict[name][f"{approach} # Success"] = int(d_approach['successful_runs'])
                        table_dict[name][f"{approach} # MSE > 0.001"] = int(d_approach['successful_runs_filter'])
                    else:
                        table_dict[name][f"{approach} # Success"] = int(d_approach['successful_runs_filter'])
                except:
                    table_dict[name][f"{approach} # Success"] = '-'
                try:
                    table_dict[name][f"{approach} MSE Q2"] = round(d_approach['err_filter_quantile'][1], 2)
                    table_dict[name][f"{approach} MSE Q3"] = round(d_approach['err_filter_quantile'][2], 2)

                    # table_dict[name][f"{approach} MSE Q1, Q3"] = [round(d_approach['err_filter_quantile'][0], 2),
                    #                                         round(d_approach['err_filter_quantile'][2], 2)]

                except:
                    table_dict[name][f"{approach} MSE Q2"] = '-'
                    # table_dict[name][f"{approach} MSE Q1, Q3"] = '-'
                try:
                    table_dict[name][f"{approach} Q2 Symbols"] = round(d_approach['num_op_filter_quantile'][1], 0)
                    table_dict[name][f"{approach} Runtime Q2"] = round(d_approach['runtime_median'], 2)
                except:
                    table_dict[name][f"{approach} Q2 Symbols"] = '-'

    cast_all_cells_to_strings(table_dict)

    highlight_extreme_values(highlight='min',
                             highlight_metric='MSE Q2',
                             table_dict=table_dict,
                             pos_in_metric=None)

    highlight_metric = 'MSE Q2'
    highlight_significance(highlight_metric, significance_table, table_dict)
    for eqs in table_dict.keys():
        for approach in ['residuals','CVGP', 'permute', 'refit_constants', 'hyperparameter', 'post_gp', 'classic']:
            table_dict[eqs][f'{approach} MSE Q2'] = f"{table_dict[eqs][f'{approach} MSE Q2']} , {table_dict[eqs][f'{approach} MSE Q3']}"
            #table_dict[eqs][f'{approach} Q2 Symbols'] = f"{table_dict[eqs][f'{approach} Q2 Symbols']} , {table_dict[eqs][f'{approach} Runtime Q2']}"
            del table_dict[eqs][f'{approach} MSE Q3']
            #del table_dict[eqs][f'{approach} Runtime Q2']

    table = pd.DataFrame(table_dict)
    return table


def formate_latex_table(results_df, result_dic):
    latex_table = results_df.to_latex()
    eq_finder = list(result_dic.keys())[0]
    for approach in result_dic[eq_finder]:
        latex_table = latex_table.replace(f"{approach} # Success", "\qquad  # Completed")
        latex_table = latex_table.replace(f"{approach} # MSE > 0.001", "\qquad  # MSE >0.001")
        latex_table = latex_table.replace(f"{approach} MSE Q2", "\qquad MSE Q2, Q3")
        #latex_table = latex_table.replace(f"{approach} Q2 Symbols", "\qquad # Op. , t[s]")
        #latex_table = latex_table.replace(f"{approach} MSE Q1, Q3", "\qquad MSE Q1, Q3 ")
        latex_table = latex_table.replace(f"{approach} Q2 Symbols", "\qquad # Operators Q2")
        #latex_table = latex_table.replace(f"{approach} MSE Q3", "\qquad MSE Q3")
        latex_table = latex_table.replace(f"{approach} Runtime Q2", "\qquad Runtime Q2 [sec]")

    latex_table = latex_table.replace("tabular}", "tabularx}{\\textwidth}")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("&", " & ")
    latex_table = latex_table.replace(",", " , ")
    latex_table = latex_table.replace("\\\\", " \\\\ ")
    latex_table = latex_table.replace("#", "\#")
    latex_table = latex_table.replace("llllll", "lRRRRR")
    latex_table = latex_table.replace("lllll", "lRRRR")
    latex_table = latex_table.replace("varnothing", "$\\varnothing$")
    latex_table = latex_table.replace("phantom", "\\phantom")
    latex_table = latex_table.replace(".00 ", "\phantom{.00} ")
    latex_table = latex_table.replace(".0 ", "\phantom{.0} ")

    latex_table = latex_table.replace("textbf", "\\textbf")
    latex_table = latex_table.replace("underline", "\\underline")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("runtime", "Running Time [sec]")
    latex_table = latex_table.replace("residuals", "\\ours{}")
    latex_table = latex_table.replace("permute", "Permute")
    latex_table = latex_table.replace("classic", "Classic")
    latex_table = latex_table.replace("hyperparameter", "Hyper")
    latex_table = latex_table.replace("refit_constants", "Refit")
    latex_table = latex_table.replace("post_gp", "Seeded GPLearn")

    return latex_table


def cast_all_cells_to_strings(table_dict):
    for equation_finder, metric_dict in table_dict.items():
        for metric, value in metric_dict.items():
            if "MSE Q1, Q3" in metric:
                metric_dict[metric] = f"{value[0]:.2f}, {value[1]:.2f}"
    for eqs, eqs_dict in table_dict.items():
        for metric, metric_value in eqs_dict.items():
            if not isinstance(table_dict[eqs][metric], str):
                table_dict[eqs][metric] = f"{table_dict[eqs][metric]:.2f}"


def highlight_significance(highlight_metric, significance_table, table_dict):
    for eqs, eqs_dict in significance_table.items():
        for approach, value in eqs_dict.items():
            for metric in table_dict[eqs].keys():
                if approach in metric and highlight_metric in metric:
                    if significance_table[eqs][approach] > 0.01:
                        table_dict[eqs][metric] = f"underline{{{table_dict[eqs][metric]}}}"


def highlight_extreme_values(highlight, highlight_metric, table_dict, pos_in_metric):
    for equation_finder, metric_dict in table_dict.items():
        extreme = - np.inf if highlight == 'max' else np.inf
        extreme_metric = []
        for metric, value in metric_dict.items():
            if highlight_metric in metric:
                if pos_in_metric:
                    v = value[pos_in_metric]
                else:
                    v = value
                v = float(v)
                if highlight == 'max' and v > extreme:
                    extreme_metric = [metric]
                    extreme = v
                elif highlight == 'min' and v < extreme:
                    extreme_metric = [metric]
                    extreme = v
                elif v == extreme:
                    extreme_metric.append(metric)
        for m in extreme_metric:
            if pos_in_metric:
                metric_dict[m][pos_in_metric] = f"textbf{{{metric_dict[m][pos_in_metric]}}}"
            else:
                metric_dict[m] = f"textbf{{{metric_dict[m]}}}"


def print_statistics(classic_err_list, perm_err_list, quantile_perm, quantile_residual,
                     residual_err_list):
    print(f'classic err: {mean_and_round(classic_err_list, 0)} pm {std_and_round(classic_err_list, 0)}, '
          f'perm err: {mean_and_round(perm_err_list, 0)} pm {std_and_round(perm_err_list, 0)},'
          f' residual retry: {mean_and_round(residual_err_list, 0)} pm {std_and_round(residual_err_list, 0)}'
          )
    print(
        f'median perm err: {quantile_perm[1]} pm {quantile_perm[[0, 2]]},'
        f'median residual_err_list: {quantile_residual[1]} pm {quantile_residual[[0, 2]]},'
    )


def fill_ax_of_figure(result_dict):
    approach_list = ['classic', 'residuals']
    fig, axs = plt.subplots(len(result_dict), figsize=(10, 10))  # constrained_layout=True
    for i, (equation_finder, equation_finder_d) in enumerate(result_dict.items()):
        ax = axs[i]
        experiment_list = equation_finder_d['residuals']['experiment_list_filtered']
        if len(experiment_list) == 0:
            continue
        sorted_index = np.lexsort(
            np.flip(
                [sort_key(item) for item in
                 experiment_list
                 ], axis=1).transpose()
        )
        experiment_list = np.array(experiment_list)[sorted_index]
        for approach in approach_list:
            d_approach = equation_finder_d[approach]
            if isinstance(d_approach, dict):
                error = d_approach['err_filter_mean'][sorted_index]
                std = d_approach['err_filter_var'][sorted_index]
                if np.all(np.isnan(error)):
                    continue
                error[error > 100] = 100
                error[error < 1e-8] = 1e-8
                if isinstance(d_approach, dict):
                    axs[i].errorbar(
                        experiment_list,
                        error,
                        0,  # std,
                        label=approach,
                        # s=marker_size[:, 1]
                    )
        ax.set_title(f'{equation_finder}')
        ax.set_ylabel('Test MSE')
        # ax.set_yscale('function', functions=(forward, inverse))
        # ax.set_yscale('log')
        x_labels = ax.get_xticklabels()
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        # plt.xticks(rotation=45, ha='right')
        # plt.yscale("log")
        ax.grid(visible=True)
        # ax.legend(loc='upper right')  # 'lower left')
    handles, labels = ax.get_legend_handles_labels()
    # Add a line break to long labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)

    fig.tight_layout(rect=(0, 0, 1, 1))  # (left, bottom, right, top)

    return


if __name__ == '__main__':
    run()
