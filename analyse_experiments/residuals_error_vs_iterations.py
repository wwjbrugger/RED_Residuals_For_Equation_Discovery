import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from analyse_experiments.compare_results_acc import read_results_of_approach
from definitions import ROOT_DIR
import pandas as pd


def run():
    id = f"sr_bench"
    result_dict = {}
    if id == "sr_bench":
        folder_path = 'results/datasets_srbench/dataset_size_300/seed_0/num_res100/pysr_i10'
        cvgp_path = {
            0: ROOT_DIR / folder_path/  'normalCVGP_28_Feb_2025_12:30:03.json',
        }

        path_to_results = {
            0: {
                'SymbolicGPT': ROOT_DIR / folder_path/ 'normalsymbolicgpt_27_Feb_2025_22:38:12.json',
                'NeSymReS': ROOT_DIR / folder_path/ 'normalnesymres_28_Feb_2025_09:26:00.json',
                #'PySR': ROOT_DIR / folder_path/ 'normalPySR_28_Feb_2025_13:26:14.json',
                #'GP': ROOT_DIR / folder_path/  'normalGP_28_Feb_2025_12:29:45.json'
            },
        }
    global Mode
    Mode = 'test'
    result_dict={}
    for seed in path_to_results.keys():
        result_dict[seed] = {}
        for i, (name, path) in enumerate(path_to_results[seed].items()):
            with open(path) as f:
                d = json.load(f)
                if not "finished" in d:
                    print(f"{path} not finished yet, number of runs {len(d)}")
            result_dict[seed][name] = get_runs(d)
    result_dict = resort_result_dict(result_dict)

    fig, axs = plt.subplots(len(result_dict), 3, figsize=(6, 3.5))
    for row, approach in enumerate(result_dict):
        axs[row][1].set_title(approach)

        mean_rel_error = np.mean(flatten_seeds(result_dict,approach,'error_train_rel'),axis=0)
        mean_rel_std = np.var(flatten_seeds(result_dict, approach, 'error_train_rel'), axis=0)
        axs[row][0].errorbar(range(len(mean_rel_error)), mean_rel_error, yerr=0, fmt='o-', ms=2)#mean_rel_std
        #axs[row][0].plot(range(len(mean_rel_error)), mean_rel_error)
        axs[row][0].set_ylim((0,1))
        axs[row][0].set_xlabel('Iteration')
        axs[row][0].set_ylabel('Rel. Train MSE')

        mean_rel_error = np.mean(flatten_seeds(result_dict,approach,'error_test_rel'), axis=0)
        mean_rel_std = np.var(flatten_seeds(result_dict,approach,'error_test_rel'), axis=0)
        axs[row][1].errorbar(range(len(mean_rel_error)), mean_rel_error, yerr=0, fmt='o-', ms=2)
        axs[row][1].set_ylim((0, 1))
        axs[row][1].set_xlabel('Iteration')
        axs[row][1].set_ylabel('Rel. Test MSE')

        mean_num_operation =np.mean(flatten_seeds(result_dict,approach,'num_operations'), axis=0)# np.mean([result_dict[approach][file]['num_operations'] for file in result_dict[approach]], axis=0)
        mean_num_operation_std = np.var(flatten_seeds(result_dict,approach,'num_operations'), axis=0)
        axs[row][2].errorbar(range(len(mean_num_operation)), mean_num_operation, yerr=0,fmt='o-', ms=2)

        axs[row][2].set_xlabel('Iteration')
        axs[row][2].set_ylabel('# Operator')

    fig.tight_layout(rect=(0, 0, 1, 1)) #(left, bottom, right, top)

    save_path = path.parent.parent / f"residual_error_vs_iterations_{id}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot is saved to : {save_path}")

    plt.show()

def flatten_seeds(result_dict, approach, metric):
    # returns np.array of shape (files, iterations)
    aa= [pd.DataFrame(result_dict[approach][file][metric]).to_numpy() for file in result_dict[approach]]
    c = []
    for a in aa:
        for i in range(a.shape[1]):
            c.append(a[:, i])
    return np.array(c)


def resort_result_dict(result_dict):
    # Iterate through the original dictionary and restructure it
    # [seed][finder][dataset][stat] -> [finder][dataset][stat][seed]
    new_dict = {}
    for k_seed, value1 in result_dict.items():
        for k_finder, value2 in value1.items():
            for k_dataset, value3 in value2.items():
                for k_stat, value4 in value3.items():
                    if k_finder not in new_dict:
                        new_dict[k_finder] = {}
                    if k_dataset not in new_dict[k_finder]:
                        new_dict[k_finder][k_dataset] = {}
                    if k_stat not in new_dict[k_finder][k_dataset]:
                        new_dict[k_finder][k_dataset][k_stat] = {}
                    new_dict[k_finder][k_dataset][k_stat][k_seed] = value4
    return new_dict
def get_runs(d):
    return_dict = {}
    for experiment, ex_dict in d.items():
        #if experiment in filter:
        if isinstance(ex_dict, dict) and 'residuals' in ex_dict:
            iteration = [0]
            try:
                error_test = [ex_dict['classic']['test']['0']['error']]
            except:
                print(f"For {experiment} no test value recorded")
                continue
            num_operations = [ex_dict['classic']['test']['0']['num_operations']]
            error_train = [ex_dict['classic']['train']['0']['error']]
            res_dict = ex_dict['residuals']
            for i in range(1,100,1):
                iteration.append(int(i))
                i = str(i)
                if i in res_dict['test']:
                    try:
                        error_test.append(res_dict['test'][i]['error'])
                        error_train.append(res_dict['train'][i]['error'])
                        num_operations.append(res_dict['test'][i]['num_operations'])
                    except:
                        error_test.append(error_test[-1])
                        error_train.append(error_train[-1])
                        num_operations.append(num_operations[-1])
                else:
                    error_test.append(error_test[-1])
                    error_train.append(error_train[-1])
                    num_operations.append(num_operations[-1])
            return_dict[experiment] = \
                {
                    'iteration': iteration,
                    'error_test': error_test,
                    'error_test_rel': error_test/np.max(error_test),
                    'error_train': error_train,
                    'error_train_rel': error_train / np.max(error_train),
                    'num_operations': num_operations,
                }
    return return_dict


if __name__ == '__main__':
    run()
