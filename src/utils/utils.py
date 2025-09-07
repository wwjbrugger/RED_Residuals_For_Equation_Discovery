import json
import pathlib

import numpy as np
import os, sys

from definitions import ROOT_DIR


def tie_breaking_argmax(a: np.ndarray, eps: float = 1e-8) -> int:
    return np.argmax(a + np.random.random(a.shape) * eps)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def save_result_dict(args, time_stamp, approach, results):
    file = ROOT_DIR / (f'results/{args.dataset_folder}/dataset_size_{args.max_dataset_size}/seed_{args.seed}/num_res{args.max_num_residuals}/'
                       f'pysr_i{args.pysr_niterations}/{args.exp_name}{approach}_{time_stamp}.json')
    file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Results are saved to {file}")
    with open(file, "w") as f:
        json.dump(results, f, indent=4)

def load_result_dict(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results