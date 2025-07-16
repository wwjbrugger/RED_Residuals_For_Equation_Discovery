import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from analyse_experiments.compare_results_acc import get_best_run_from_multiple
from definitions import ROOT_DIR
import pandas as pd


def run():
    id ="PySR Iteration"
    markers = {
        'permute': 'o',
        'post_gp': 'v',
        'residuals': 's',
        'classic': 'p'
    }
    colors = {
        'permute': 'forestgreen',
        'post_gp': 'mediumblue',
        'residuals': 'tomato',
        'classic': 'cornflowerblue'
    }
    result_dict = {}
    dataset = 'datasets_srbench'
    path_to_results = {
        # all
        '10': ROOT_DIR / f'results/{dataset}/dataset_size_300/seed_4/num_res10/pysr_i10' / 'pysr_onlyPySR_27_Feb_2025_20:57:08.json',
        '50': ROOT_DIR / f'results/{dataset}/dataset_size_300/seed_4/num_res10/pysr_i50' / 'pysr_onlyPySR_27_Feb_2025_15:19:21.json',
        '100': ROOT_DIR / f'results/{dataset}/dataset_size_300/seed_4/num_res10/pysr_i100' / 'pysr_onlyPySR_27_Feb_2025_15:24:47.json',
        '150': ROOT_DIR / f'results/{dataset}/dataset_size_300/seed_4/num_res10/pysr_i150' / 'pysr_onlyPySR_28_Feb_2025_12:28:59.json',
        '200': ROOT_DIR / f'results/{dataset}/dataset_size_300/seed_4/num_res10/pysr_i200' / 'pysr_onlyPySR_28_Feb_2025_12:29:20.json',
    }

    for i, (name, path) in enumerate(path_to_results.items()):
        result_dict[name] = read_results_of_approach(name, path)

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.1))
    approaches = ['classic']
    for approach in approaches:
        if approach == 'residuals':
            label = 'RED'
        elif approach == 'post_gp':
            label = 'Seeded GPLearn'
        else:
            label = approach.title()

        for i, metric in enumerate(['err_train', 'err_test', 'num_op']):
            x = []
            y = []
            y_low = []
            y_high = []
            for level in result_dict.keys():
                y_values = np.array(result_dict[level][approach][metric])
                if approach == 'classic' and not id == 'PySR Iteration':
                    y_values[y_values < 0.001] = np.nan
                quatiles = np.nanquantile(y_values, (0.25, 0.5, 0.75), axis=0)
                y_low.append(quatiles[0])
                y.append(quatiles[1])
                y_high.append(quatiles[2])
                x.append(float(level))
                nammin = np.nanmin(y_values)
                nammax = np.nanmax(y_values)
                pass
            y_low = np.array(y_low)
            y = np.array(y)
            y_high = np.array(y_high)
            if 'err' in metric:
                axs[i].errorbar(x, y, ms=6, label=label,
                            fmt=f'{markers[approach]}--',
                            c=colors[approach], )
            else:
                axs[i].errorbar(x, y, yerr=[y - y_low, y_high - y],
                            fmt=f'{markers[approach]}--',
                            c=colors[approach],
                            ms=6, label=label
                            )
            axs[i].set_xlabel('Iteration')
            if metric == 'err_train':
                axs[i].set_ylabel(f"Train MSE")
            if metric == 'err_test':
                axs[i].set_ylabel(f"Test MSE")
            if metric == 'num_op':
                axs[i].set_ylabel("# Operator")


    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1),
               ncol=len(approaches))

    fig.tight_layout(rect=(0, 0, 1, 0.88))  # (left, bottom, right, top)

    save_path = ROOT_DIR / f"results/{dataset}/{id}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(F"Figure is saved to: {save_path}")
    plt.show()


def read_results_of_approach(name, path):
    #
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    if not "finished" in d:
        print(f"{path} not finished yet, number of runs {len(d)}")
    approach_dict = {
        'classic': {'err_train': [],
                    'err_test': [],
                    'num_op': [],
                    'runtime': [],
                    'experiment_list': []
                    },
        'post_gp': {'err_train': [],
                    'err_test': [],
                    'num_op': [],
                    'runtime': [],
                    'experiment_list': []
                    },
        'hyperparameter': {'err_train': [],
                           'err_test': [],
                           'num_op': [],
                           'runtime': [],
                           'experiment_list': []
                           },
        'refit_constants': {'err_train': [],
                            'err_test': [],
                            'num_op': [],
                            'runtime': [],
                            'experiment_list': []
                            },
        'residuals': {'err_train': [],
                      'err_test': [],
                      'num_op': [],
                      'runtime': [],
                      'experiment_list': []
                      },
        'permute': {'err_train': [],
                    'err_test': [],
                    'num_op': [],
                    'runtime': [],
                    'experiment_list': []
                    },
    }
    for experiment in d.keys():
        if isinstance(d[experiment], dict):
            for approach, sub_d in approach_dict.items():
                error_test, num_op_test, best_node_id_test, runtime = get_best_run_from_multiple(
                    d, experiment, approach, Mode='test')
                error_train, num_op_train, best_node_id_train, runtime = get_best_run_from_multiple(
                    d, experiment, approach, Mode='train')

                sub_d['err_test'].append(error_test)
                sub_d['err_train'].append(error_train)
                sub_d['num_op'].append(num_op_test)
                sub_d['experiment_list'].append(
                    experiment.replace('.tsv.gz', '').
                    replace('feynman_', '')
                )
    return approach_dict


if __name__ == '__main__':
    run()
