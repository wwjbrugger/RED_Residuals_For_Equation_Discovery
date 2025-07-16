import json
import re

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from analyse_experiments.compare_results_acc import read_results_of_CVGP, create_approach_dict_with_empty_lists, resort_result_dict
from definitions import ROOT_DIR
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns

from src.utils.argument_parser import Namespace


def run():
    id = f"sr_bench"
    fill_methods_with_classic = True
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
                'E2E': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalkaminey_01_Mar_2025_23:31:56.json',
                'SymbolicGPT': ROOT_DIR / folder_path / 'seed_0' / 'num_res10/pysr_i10' / 'normalsymbolicgpt_27_Feb_2025_21:15:35.json',
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
    args = Namespace()
    args.mode = 'test'
    args.replace_with_classic = True

    result_dict = {}
    for seed in path_to_results.keys():
        fill_eds_with_CVGP(
            args,
            cvgp_path,
            path_to_results,
            result_dict,
            seed
        )
    result_dict, _ = resort_result_dict(result_dict)
    datasets = list(result_dict['NeSymReS'].keys())

    eqs = 'NeSymReS'
    approach_list=  ['residuals', 'classic', 'post_gp', 'hyperparameter','refit_constants', 'permute',
                     'CVGP']
    labels = {'residuals': 'RED',
              'classic': 'Classic',
              'post_gp': 'Seeded GPLearn',
              'hyperparameter': 'Hyper',
              'refit_constants': 'Refit',
              'permute': 'Permute',
              'CVGP': 'CVGP'}
    ratio_df = get_ratio_df(approach_list, args, datasets, eqs, result_dict)

    generate_figure(eqs, folder_path, labels, ratio_df)


def generate_figure(eqs, folder_path, labels, ratio_df):
    ratio_df.index = [labels[ind] for ind in ratio_df.index]
    ratio_df.columns = [labels[ind] for ind in ratio_df.columns]
    fig, axs = plt.subplots(1, 1, figsize=(7, 3))
    #axs.set_title('$\\frac{wins}{wins+draws+loss}$')
    sns.heatmap(ratio_df, annot=True, cmap='RdYlGn', linewidths=1.5, ax=axs,
                vmin=0, vmax=1)
    x_labels = axs.get_xticklabels()
    axs.set_xticklabels(x_labels, rotation=30, ha='right')
    # y_labels = axs.get_yticklabels()
    # axs.set_yticklabels(y_labels, rotation=45, va='center')
    fig.tight_layout(rect=(0, 0, 1, 1))
    save_path = ROOT_DIR / folder_path / f'{eqs}_win_loss.pdf'
    print(f"Figure is saved @ {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.show()


def get_ratio_df(approach_list, args, datasets, eqs, result_dict):
    ratio_df = pd.DataFrame()
    for row in range(len(approach_list)):
        for column in range(0, len(approach_list), 1):
            approach_0 = approach_list[row]
            approach_1 = approach_list[column]
            if row == column :
                ratio_df.loc[approach_0, approach_1] = np.nan
                continue
            winsloss = calculate_wins_loss(approach_0, approach_1, args, datasets, eqs, result_dict)
            print(f"{approach_0}, {approach_1} num comparison: {winsloss.num_comparison}")
            ratio_df.loc[approach_0, approach_1] = winsloss.wins / (winsloss.wins+ winsloss.loss + winsloss.draw)

    return ratio_df


def calculate_wins_loss(approach_0, approach_1, args, datasets, eqs, result_dict):
    winsloss = WinsLoss()
    for dataset in datasets:
        try:
            err_vector_classic = get_results_one_experiment(
                result_dict,
                eqs=eqs,
                dataset=dataset,
                approach='classic',
                seed=0,
                metric='error'
            )
            if err_vector_classic[0] < 0.001 or np.isnan(err_vector_classic[0]):
                continue

            vector_0 = get_results_one_experiment(
                result_dict,
                eqs=eqs,
                dataset=dataset,
                approach=approach_0,
                seed=0,
                metric='error'
            )
            if np.isnan(vector_0[0]):
                if args.replace_with_classic:
                    vector_0 = err_vector_classic
                else:
                    continue

            vector_1 = get_results_one_experiment(
                result_dict,
                eqs=eqs,
                dataset=dataset,
                approach=approach_1,
                seed=0,
                metric='error'
            )

            if np.isnan(vector_1[0]):
                if args.replace_with_classic:
                    vector_1 = err_vector_classic
                else:
                    continue
            winsloss.compare_all(
                vector_0,
                vector_1
            )
        except:
            pass
    return winsloss


def get_results_one_experiment(result_dict, eqs, dataset, approach, seed, metric):
    try:
        single_run_dict = result_dict[eqs][dataset][approach][seed]['test']
        metric_list = []
        if len(single_run_dict.keys()) == 0:
            return [np.nan]
        for i in single_run_dict.keys():
            metric_list.append(single_run_dict[i][metric])
        return metric_list
    except:
        return [np.nan]

def fill_eds_with_CVGP(args, cvgp_path, path_to_results, result_dict, seed):
    with open(cvgp_path[seed]) as f:
        cvgp_dict = json.load(f)
    result_dict[seed] = {}
    for i, (eds, path) in enumerate(path_to_results[seed].items()):
        path = Path(path)
        with open(path) as f:
            eds_dict = json.load(f)
        if not "finished" in eds_dict:
            print(f"{path} not finished yet, number of runs {len(eds_dict)}")
        keys_to_delete = []
        for experiment, exp_dic in eds_dict.items():
            if isinstance(eds_dict[experiment], dict):
                try:
                    exp_dic['CVGP'] = cvgp_dict[experiment]['classic']
                except:
                    exp_dic['CVGP'] = cvgp_dict[experiment]

            else:
                keys_to_delete.append(experiment)
        for exp in keys_to_delete:
            del eds_dict[exp]
        result_dict[seed][eds] = eds_dict
    return result_dict




def mk_array_dim(element):
    a = np.array(element)
    if len(a.shape()<1):
        a = np.expand_dims(a, axis=0)
    return a

class WinsLoss():
    def __init__(self):
        self.wins= 0
        self.loss = 0
        self.draw = 0
        self.num_comparison = 0

    def compare_all(self, vector_0, vector_1):
        self.num_comparison +=1
        for i in vector_0:
            for j in vector_1:
                if i<j:
                    self.wins +=1
                elif i==j:
                    self.draw +=1
                elif i>j:
                    self.loss += 1

if __name__ == '__main__':
    run()