import traceback
import warnings
from sympy import sympify

from definitions import ROOT_DIR
from src.configs.config_PySR import ConfigPySR
from src.configs.config_general import ConfigGeneral
from src.configs.config_syntax_tree import ConfigSyntaxTree
from src.evaluate_equation import evaluate_equation, map_equation_to_syntax_tree
from src.experiment_schedule import run_experiments
from src.preprocess import get_datasets_files
from src.utils.utils import HiddenPrints

warnings.filterwarnings("ignore", category=UserWarning)
import random
from functools import partial
import numpy as np
from pysr import PySRRegressor


def load_model(args, hyperparameter_set = 0):
    hyperparameter = {
        0: {'weight_add_node': 0.79,
            'weight_insert_node': 5.1,
            'weight_delete_node': 1.7,
            'weight_do_nothing': 0.21,
            'weight_mutate_constant': 0.048,
            'weight_mutate_operator': 0.47,
            'weight_swap_operands': 0.1,
            'weight_randomize': 0.00023,
            'weight_simplify': 0.0020,
            'weight_optimize': 0.0,
            'crossover_probability': 0.066},

        1: {'weight_add_node': 0.9,
            'weight_insert_node': 4.5,
            'weight_delete_node': 1.85,
            'weight_do_nothing': 0.15,
            'weight_mutate_constant': 0.08,
            'weight_mutate_operator': 0.37,
            'weight_swap_operands': 0.15,
            'weight_randomize': 0.00017,
            'weight_simplify': 0.003,
            'weight_optimize': 0.0,
            'crossover_probability': 0.075},

        2: {'weight_add_node': 1.0,
            'weight_insert_node': 4.0,
            'weight_delete_node': 2.0,
            'weight_do_nothing': 0.1,
            'weight_mutate_constant': 0.1,
            'weight_mutate_operator': 0.27,
            'weight_swap_operands': 0.2,
            'weight_randomize': 0.0001,
            'weight_simplify': 0.0040,
            'weight_optimize': 0.0,
            'crossover_probability': 0.09}
    }

    model = PySRRegressor(
        niterations=10,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        constraints={
            "square": 6,
            "cube": 6,
            "exp": 6,
            "sin": 6,
            "cos": 6
        },
        nested_constraints={
            "sin": {"sin": 0, "cos": 0, "exp": 0},
            "cos": {"sin": 0, "cos": 0, "exp": 0},
            "exp": {"sin": 0, "cos": 0, "exp": 0}
        },
        progress=False,
        # ^ Custom loss function (julia syntax)
        **hyperparameter[hyperparameter_set]
    )
    fitfunc = partial(call_pysr,
                      model=model,
                      args=args
                      )
    return fitfunc


def call_pysr(model, X_df, Y_df, args, info=None):
    try:
        with HiddenPrints():
            output = model.fit(X_df.to_numpy(), Y_df.to_numpy())
        best_index = np.argmax(output.equations_['score'].to_numpy())
        equation = output.equations_['equation'][best_index]
        equation = str(sympify(equation))
        clean_up_pysr()
        tree = map_equation_to_syntax_tree(args, equation, infix=True)
    except (SyntaxError, RuntimeError, FloatingPointError) as E:
        print(traceback.format_exc())
        return {}
    output = evaluate_equation(args, tree, X_df, Y_df)
    return output

def clean_up_pysr():
    files_to_delete = (ROOT_DIR).glob('hall_of_fame*')
    # Delete the files
    for file_path in files_to_delete:
        try:
            file_path.unlink()
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")



def run(args):
    print("Load Model")
    load_model_func = load_model
    run_experiments(args, load_model_func, 'PySR')


if __name__ == '__main__':
    print("Start")
    parser = ConfigSyntaxTree.arguments_parser()
    parser = ConfigPySR.arguments_parser(parser)
    parser = ConfigGeneral.arguments_parser(parser)
    args = parser.parse_args()

    run(args)
