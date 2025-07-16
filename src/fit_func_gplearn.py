import traceback
import warnings

from sympy import sympify

from src.configs.config_PySR import ConfigPySR
from src.configs.config_general import ConfigGeneral
from src.configs.config_gplearn import ConfigGPlearn
from src.configs.config_syntax_tree import ConfigSyntaxTree
from src.evaluate_equation import evaluate_equation, map_equation_to_syntax_tree
from src.gplearn.gplearn.functions import _Function
from src.gplearn.gplearn.genetic import SymbolicRegressor
from src.preprocess import get_datasets_files

warnings.filterwarnings("ignore", category=UserWarning)
import random
from functools import partial

import numpy as np
def load_model(args, hyperparameter_set = 0):
    hyperparameter = {
        0: {'p_crossover':0.9, 'p_subtree_mutation': 0.01,
            'p_hoist_mutation': 0.01, 'p_point_mutation': 0.01,
            'p_point_replace': 0.05 }, # the difference to 1 is no mutation
        1: {'p_crossover':0.8, 'p_subtree_mutation': 0.05,
            'p_hoist_mutation': 0.025, 'p_point_mutation': 0.05,
            'p_point_replace': 0.05 },
        2: {'p_crossover':0.7, 'p_subtree_mutation': 0.1,
            'p_hoist_mutation': 0.05, 'p_point_mutation': 0.1,
            'p_point_replace': 0.05 }
    }
    esp_gp = SymbolicRegressor(population_size=5000, generations=10,
                      max_samples=0.9,
                      parsimony_coefficient=0.01, random_state=0,
                      function_set=(
                               'add','sub','mul','div','sqrt',
                               'log','abs','neg','inv',
                               #'max','min',
                          'sin','cos','tan','pow'),
                      # init_method='grow', init_depth=(2, 3),
                      **hyperparameter[hyperparameter_set]
                      )
    fitfunc = partial(call_gplearn,
                      model=esp_gp,
                      args=args
                      )
    return fitfunc


def call_gplearn(model, X_df, Y_df, args, start_population=None, info=None):
    try:
        if start_population:
            model.fit(X_df, Y_df, start_population=start_population)
        else:
            model.fit(X_df, Y_df)
        gp_prefix = gp_program_to_prefix(model._program.program)
        tree = map_equation_to_syntax_tree(args, gp_prefix, infix=False)
        infix = tree.rearrange_equation_infix_notation()[1]
        sympy_infix = str(sympify(infix))
        sympy_tree = map_equation_to_syntax_tree(args, sympy_infix, infix=True)
    except (SyntaxError, RuntimeError, FloatingPointError) as E:
        print(traceback.format_exc())
        return {}
    output = evaluate_equation(args, sympy_tree, X_df, Y_df)
    return output


def gp_program_to_prefix(program):
    prefix = ''
    for f in program:
        prefix += ' '
        if isinstance(f,_Function):
            f_str = f.__str__()
            if f_str =='add':
                prefix += '+'
            elif f_str =='sub':
                prefix += '-'
            elif f_str =='pow':
                prefix += '**'
            elif f_str =='mul':
                prefix += '*'
            elif f_str == 'div':
                prefix += '/'
            elif f_str == 'neg':
                prefix += ' - 0 '
            elif f_str == 'inv':
                prefix += ' / 1  '
            elif f_str == 'Abs':
                prefix += ' abs  '
            else:
                prefix += f_str
        elif isinstance(f, int):
            prefix += f'x_{f}'
        elif isinstance(f, float):
            prefix += str(f)
        else:
            raise NotImplementedError('This we should never reach '
                                      f'f is a instance of {f}')
    return prefix

def run(args):
    from src.experiment_schedule import run_experiments
    print("Load Model")
    load_model_func = load_model
    run_experiments(args, load_model_func, 'GP')


if __name__ == '__main__':
    print("Start")
    parser = ConfigSyntaxTree.arguments_parser()
    parser = ConfigGPlearn.arguments_parser(parser)
    parser = ConfigGeneral.arguments_parser(parser)
    parser = ConfigPySR.arguments_parser(parser)
    args = parser.parse_args()

    run(args)
