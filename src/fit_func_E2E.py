import os
import sys
from functools import partial
import requests
import torch
from sympy import sympify
from src.E2E.symbolicregression.model import SymbolicTransformerRegressor
from definitions import ROOT_DIR
from nesymres.architectures.model import Model
from src.configs.config_E2E import ConfigE2E
from src.configs.config_PySR import ConfigPySR
from src.configs.config_general import ConfigGeneral
from src.configs.config_syntax_tree import ConfigSyntaxTree
from src.evaluate_equation import evaluate_equation, map_equation_to_syntax_tree
from src.experiment_schedule import run_experiments
from src.preprocess import get_datasets_files
from src.utils.utils import HiddenPrints


def load_model(args, hyperparameter_set = 0):
    hyperparameter = {
        0: {'max_input_points': 200, 'n_trees_to_refine':100},  # the difference to 1 is no mutation
        1: {'max_input_points': 400, 'n_trees_to_refine':80},
        2: {'max_input_points': 800, 'n_trees_to_refine':60}
    }
    sys.path.append(f'{ROOT_DIR}/src/kaminey_model')
    model_path = "model.pt"
    # try:
    if not os.path.isfile(model_path):
        url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    cfg={
        'max_input_points': args.max_input_points,
        'n_trees_to_refine': args.n_trees_to_refine,
        'rescale': args.rescale
    }
    if not torch.cuda.is_available():
        model = Model(cfg)
        model.load_state_dict(torch.load(model_path), weights_only=True)
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
        model = model.cuda()
    print(model.device)
    print("Model successfully loaded!")
    est = SymbolicTransformerRegressor(
        model=model,
        rescale=True,
        **hyperparameter[hyperparameter_set]
    )
    fitfunc = partial(call_kaminey,
                      model=est,
                      args=args
                      )
    return fitfunc


def call_kaminey(model, args, X_df, Y_df, info=None):
    with HiddenPrints():
        try:
            output = model.fit(X_df.to_numpy(), Y_df.to_numpy())
            model_str = model.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
            replace_dict = {
                'mul': '*',
                'add': '+',
                'sub': '-',
                'inv': '1 / '
            }
            for old, new in replace_dict.items():
                model_str = model_str.replace(old, new)
            model_str = str(sympify(model_str))
            tree = map_equation_to_syntax_tree(args, model_str, infix=True)
        except:
            print(f"Generating an equation failed")
            return {}
        output = evaluate_equation(args, tree, X_df, Y_df)
        return output


def run(args):
    load_model_func = load_model
    run_experiments(args, load_model_func, approach='E2E')


if __name__ == '__main__':
    print("Start")
    parser = ConfigSyntaxTree.arguments_parser()
    parser = ConfigE2E.arguments_parser(parser)
    parser = ConfigGeneral.arguments_parser(parser)
    parser = ConfigPySR.arguments_parser(parser)
    args = parser.parse_args()

    run(args)
