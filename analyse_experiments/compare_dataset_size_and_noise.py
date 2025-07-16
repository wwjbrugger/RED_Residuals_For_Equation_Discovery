import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from analyse_experiments.compare_results_acc import get_best_run_from_multiple
from definitions import ROOT_DIR
import pandas as pd


def run():
    ids = ["Data Set Size", "Rel. Noise"]
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

    fig, axs = plt.subplots(3, 2,
                            figsize=(6, 3.4),
                            dpi=300,
                            )
    approaches = ['residuals', 'permute',]  # 'post_gp',
    for columns, id in enumerate(ids): # "Rel. Noise" , "Data Set Size", "PySR Iteration"
        result_dict = {}
        if id == "Rel. Noise":
            model = 'NeSymRes'
            dataset = 'datasets_srbench'
            folder_path = f'{dataset}/dataset_size_300/seed_0/num_res10/pysr_i10'
            path_to_results = {
                # all
                '0': ROOT_DIR / 'results' / folder_path / 'normalnesymres_27_Feb_2025_21:14:35.json',
                '0.1': ROOT_DIR / 'results' / folder_path / 'noise01nesymres_27_Feb_2025_21:20:52.json',
                '0.3': ROOT_DIR / 'results' / folder_path / 'noise03nesymres_28_Feb_2025_11:10:11.json',
                '0.5': ROOT_DIR / 'results' / folder_path / 'noise05nesymres_27_Feb_2025_22:37:08.json',
                '1.0': ROOT_DIR / 'results' / folder_path / 'noise10nesymres_28_Feb_2025_11:09:43.json',
            }
        if id == "Data Set Size":
            model = 'NeSymRes'
            dataset = 'datasets_srbench'
            path_to_results = {
                # all
                '500': ROOT_DIR / f'results/{dataset}/dataset_size_500/seed_0/num_res10/pysr_i10' / 'number_samplesnesymres_28_Feb_2025_14:19:13.json',
                '300': ROOT_DIR / f'results/{dataset}/dataset_size_300/seed_0/num_res10/pysr_i10' / 'normalnesymres_27_Feb_2025_21:14:35.json',
                '200': ROOT_DIR / f'results/{dataset}/dataset_size_200/seed_0/num_res10/pysr_i10' / 'number_samplesnesymres_28_Feb_2025_13:30:12.json',
                '100': ROOT_DIR / f'results/{dataset}/dataset_size_100/seed_0/num_res10/pysr_i10' / 'number_samplesnesymres_28_Feb_2025_13:29:52.json',
                '50': ROOT_DIR / f'results/{dataset}/dataset_size_50/seed_0/num_res10/pysr_i10' / 'number_samplesnesymres_28_Feb_2025_09:27:44.json',
                '20': ROOT_DIR / f'results/{dataset}/dataset_size_20/seed_0/num_res10/pysr_i10' / 'number_samplesnesymres_28_Feb_2025_09:27:59.json',
                '10': ROOT_DIR / f'results/{dataset}/dataset_size_10/seed_0/num_res10/pysr_i10' / 'number_samplesnesymres_05_Mar_2025_12:59:02.json',

            }


        for i, (name, path) in enumerate(path_to_results.items()):
            result_dict[name] = read_results_of_approach(name, path)
        for approach in approaches:
            if approach == 'residuals':
                label = 'RED'
            elif approach == 'post_gp':
                label = 'Seeded GPLearn'
            else:
                label = approach.title()

            for row, metric in enumerate(['err_train', 'err_test', 'num_op']):
                add_error_and_num_operators(approach, axs[row, columns], colors, id, label, markers, metric, result_dict)
                # add_succesful_fits(approach, axs, colors, id, markers, result_dict)

    handles, labels = axs[row, columns].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1),
                ncol=len(approaches))

    fig.tight_layout(rect=(0, 0, 1, 0.94))  # (left, bottom, right, top)

    save_path = ROOT_DIR / f"results/{dataset}/{id}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(F"Figure is saved to: {save_path}")
    plt.show()


def add_succesful_fits(approach, axs, colors, id, markers, result_dict):
    x = []
    y = []
    for level in result_dict.keys():
        y_values = np.array(result_dict[level][approach]['err_train'])
        if approach == 'classic' and not id == 'PySR Iteration':
            y_values[y_values < 0.001] = np.nan
        y.append(np.sum(~np.isnan(y_values)))
        x.append(float(level))
    axs[3].errorbar(x, y, yerr=0, ms=6, label=approach,
                    fmt=f'{markers[approach]}--',
                    c=colors[approach],
                    )
    axs[3].set_xlabel(id)
    axs[3].set_ylabel('Successful Fits')


def add_error_and_num_operators(approach, ax, colors, id, label, markers, metric, result_dict):
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
        ax.errorbar(x, y, ms=6, label=label,
                        fmt=f'{markers[approach]}--',
                        c=colors[approach], )
    else:
        ax.errorbar(x, y, yerr=[y - y_low, y_high - y],
                        fmt=f'{markers[approach]}--',
                        c=colors[approach],
                        ms=6, label=label
                        )

    ax.set_ylim(-0.2,2) if max(y) <1.8 else ax.set_ylim(-0.2,15)
    if metric == 'num_op':
        ax.set_xlabel(id)
    else:
        ax.set_xticklabels([])
    if  id == "Data Set Size":
        if metric == 'err_train':
            ax.set_ylabel(f"Train MSE")
        if metric == 'err_test':
            ax.set_ylabel(f"Test MSE")
        if metric == 'num_op':
            ax.set_ylabel("# Operator")


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
