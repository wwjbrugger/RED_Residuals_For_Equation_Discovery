import json
import traceback
from functools import partial
import omegaconf
import torch
from sympy import sympify
from definitions import ROOT_DIR
from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, BFGSParams
from src.configs.config_NeSymRes import ConfigNeSymRes
from src.configs.config_PySR import ConfigPySR
from src.configs.config_general import ConfigGeneral
from src.configs.config_syntax_tree import ConfigSyntaxTree
from src.evaluate_equation import evaluate_equation, map_equation_to_syntax_tree
from src.experiment_schedule import run_experiments
from src.preprocess import get_datasets_files
from src.utils.utils import HiddenPrints


def load_model(args, hyperparameter_set=0):
    hyperparameter = {
        0: {'beam_size': 2},  # This parameter is a tradeoff between accuracy and fitting time
        1: {'beam_size': 1},
        2: {'beam_size': 4}
    }
    with open(f'{ROOT_DIR}/src/nesymres/jupyter/100M/eq_setting.json', 'r') as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load(f"{ROOT_DIR}/src/nesymres/jupyter/100M/config.yaml")
    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    params_fit = FitParams(word2id=eq_setting["word2id"],
                           id2word={int(k): v for k, v in eq_setting["id2word"].items()},
                           una_ops=eq_setting["una_ops"],
                           bin_ops=eq_setting["bin_ops"],
                           total_variables=list(eq_setting["total_variables"]),
                           total_coefficients=list(eq_setting["total_coefficients"]),
                           rewrite_functions=list(eq_setting["rewrite_functions"]),
                           bfgs=bfgs,
                           **hyperparameter[hyperparameter_set]
                           )
    weights_path = f'{ROOT_DIR}/weights/100M.ckpt'
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    fitfunc = partial(call_nesymres, model=model, params_fit=params_fit)
    return fitfunc


def call_nesymres(model, X_df, Y_df, params_fit, info=None):
    try:
        with HiddenPrints():
            output = model.fitfunc(X_df.to_numpy(), Y_df.to_numpy().squeeze(), params_fit)
        equation = output['best_bfgs_preds'][0]
        for i in range(10):
            equation = equation.replace(f'x_{i + 1}', f' x_{i} ')
        equation = str(sympify(equation))
        tree = map_equation_to_syntax_tree(args, equation, infix=True)
        output = evaluate_equation(args, tree, X_df, Y_df)
    except:
        print(traceback.format_exc())
        return {}
    
    return output


def run(args):
    load_model_func = load_model
    run_experiments(args, load_model_func, 'nesymres')


if __name__ == '__main__':
    print("Start")
    parser = ConfigSyntaxTree.arguments_parser()
    parser = ConfigNeSymRes.arguments_parser(parser)
    parser = ConfigGeneral.arguments_parser(parser)
    parser = ConfigPySR.arguments_parser(parser)
    args = parser.parse_args()

    run(args)
