import traceback
import warnings

from sympy import sympify

from src.configs.config_PySR import ConfigPySR
from src.configs.config_cvgp import ConfigCVGP
from src.configs.config_general import ConfigGeneral
from src.configs.config_syntax_tree import ConfigSyntaxTree
from src.cvgp.src.ctrl_var_gp import functions, gen_true_program, regress_task, control_variable_gp
from src.cvgp.src.ctrl_var_gp.const import ScipyMinimize
from src.cvgp.src.ctrl_var_gp.functions import _protected_power
from src.cvgp.src.ctrl_var_gp.library import Token, PlaceholderConstant, Library
from src.cvgp.src.ctrl_var_gp.program import Program
from src.evaluate_equation import map_equation_to_syntax_tree, evaluate_equation
from src.preprocess import get_datasets_files
from src.utils.utils import HiddenPrints

warnings.filterwarnings("ignore", category=UserWarning)
import random
from functools import partial

import numpy as np


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tree_to_cvgp(tree):
    program = []
    const_loc = []
    consts = []
    prefix = tree.rearrange_equation_prefix_notation()[1]
    prefix = prefix.replace(' x', ' X')
    prefix = prefix.split()
    for i, symbol in enumerate(prefix):
        if symbol == '+':
            program += ['add']
        elif symbol == 'sqrt':
            program += ['sqrt']
        elif symbol == '-':
            program += ['sub']
        elif symbol == '*':
            program += ['mul']
        elif symbol == '/':
            program += ['div']
        elif symbol == '**':
            program += ['pow']
        elif is_float(symbol):
            program += ['const']
            consts.append(float(symbol))
            const_loc.append(i)
        elif symbol == 'pi':
            program += ['const']
            consts.append(3.143)
            const_loc.append(i)
        else:
            program += [symbol]

    prog = {'preorder': program,
            'const_loc': const_loc,
            'consts': consts
            }
    return prog


def cvgp_to_infix(cvgp):
    infix = str(cvgp[0]).replace('X_', 'x_')
    return infix

def cvgp_to_prefix(cvgp):
    prefix = ''
    # cvgp =cvgp.split()
    for i, symbol in enumerate(cvgp):
        symbol = str(symbol)
        if symbol == 'add':
            prefix += ' + '
        elif symbol == 'sqrt':
            prefix += ' sqrt '
        elif symbol == 'sub':
            prefix += ' - '
        elif symbol == 'mul':
            prefix += ' * '
        elif symbol == 'div':
            prefix += ' / '
        elif symbol == 'pow':
            prefix += ' ** '
        elif symbol == 'inv':
            prefix += ' 1 / '
        elif symbol.startswith('X_'):
            symbol = symbol.replace('X_', 'x_')
            prefix += f" {symbol} "
        else:
            prefix += f" {symbol} "

    return prefix


class CVGP_Regressor():
    def __init__(self, hyper_parameter):
        self.hyper_parameter = hyper_parameter


def load_model(args, hyperparameter_set=0):
    # cxpb: probability of mate
    # mutpb: probability of mutations
    # maxdepth: the maxdepth of the tree during mutation
    # population_size: the size of the selected populations (at the end of each generation)
    # tour_size: the size of the tournament for selection
    # hof_size: the size of the best programs retained
    # n_generations: number of generations per Variables
    hyperparameter = {
        0: {'cxpb': 0.5, 'mutpb': 0.5, 'maxdepth': 2, 'tour_size': 3,
            'hof_size': 10, 'population_size' : 25, 'n_generations' : 30 },  # the difference to 1 is no mutation
        1: {'cxpb': 0.4, 'mutpb': 0.6, 'maxdepth': 2, 'tour_size': 3,
            'hof_size': 10,'population_size' : 25, 'n_generations' : 30},
        2: {'cxpb': 0.6, 'mutpb': 0.4, 'maxdepth': 2, 'tour_size': 3,
            'hof_size': 10,'population_size' : 25, 'n_generations' : 30}
    }
    model = CVGP_Regressor(
        hyper_parameter=hyperparameter[hyperparameter_set]
    )

    fitfunc = partial(call_cvgp,
                      model=model,
                      args=args
                      )
    return fitfunc


def call_cvgp(model, X_df, Y_df, info, args):
    try:
        var_to_x = {}
        equation = info['Formula']
        for i in range(1, 11, 1):
            if f'v{i}_name' in info:
                v_name = info[f'v{i}_name']
                if isinstance(v_name, str):
                    var_to_x[v_name] = f"x_{i - 1}"
                    v_low = info[f'v{i}_low']
                    v_high = info[f'v{i}_high']
        variables = list(var_to_x.keys())
        while len(variables) > 0:
            variable = variables.pop(0)
            if any(variable in string for string in variables):
                variables.append(variable)
            else:
                equation = equation.replace(variable, var_to_x[variable])
        tree = map_equation_to_syntax_tree(args, equation, infix=True)
        prog = tree_to_cvgp(tree)

        nvar = X_df.shape[1]
        regress_batchsize = 256
        opt_num_expr = 5
        noise_std = 0

        expr_obj_thres = 0.001
        expr_consts_thres = 0.01


        # get all the functions and variables ready
        var_x = []
        for i in range(nvar):
            xi = Token(None, 'X_' + str(i), 0, 0., i)
            var_x.append(xi)

        ops = [
            # Binary operators
            Token(np.add, "add", arity=2, complexity=1),
            Token(np.subtract, "sub", arity=2, complexity=1),
            Token(np.multiply, "mul", arity=2, complexity=1),
            Token(np.sin, "sin", arity=1, complexity=3),
            Token(np.cos, "cos", arity=1, complexity=3),
            Token(_protected_power, "pow", arity=2, complexity=2),
            Token(np.divide, "div", arity=2, complexity=2),
            Token(np.exp, "exp", arity=1, complexity=4),
            # functions.protected_ops[0],  # 'div'
            functions.protected_ops[5]  # 'inv' '1/x'
        ]
        named_const = [PlaceholderConstant(1.0)]
        protected_library = Library(ops + var_x + named_const)

        protected_library.print_library()

        # get program ready
        Program.library = protected_library
        Program.opt_num_expr = opt_num_expr
        Program.expr_obj_thres = expr_obj_thres
        Program.expr_consts_thres = expr_consts_thres

        Program.set_execute(True)  # protected = True

        # set const_optimizer
        Program.const_optimizer = ScipyMinimize()
        Program.noise_std = noise_std

        # read the program
        true_pr = gen_true_program.build_program(prog, protected_library, 0)

        # set the task
        allowed_input_tokens = np.zeros(nvar, dtype=np.int32)  # set it for now. Will change in gp.run
        Program.task = regress_task.RegressTaskV1(regress_batchsize,
                                                  allowed_input_tokens,
                                                  true_pr,
                                                  noise_std,
                                                  metric="neg_mse")

        # set gp helper
        gp_helper = control_variable_gp.GPHelper()
        gp_helper.library = protected_library

        # set GP
        control_variable_gp.ControlVariableGeneticProgram.library = protected_library
        control_variable_gp.ControlVariableGeneticProgram.gp_helper = gp_helper
        gp = control_variable_gp.ControlVariableGeneticProgram(nvar= nvar, **model.hyper_parameter)

        # run GP
        with HiddenPrints():
            gp.run()

        # print
        print('final hof=')

        gp.print_hof()
        print('gp.timer_log=', gp.timer_log)
        equation = gp.hof[0]
        infix = cvgp_to_infix(equation.sympy_expr)
        tree = map_equation_to_syntax_tree(args, infix, infix=True)
        output = evaluate_equation(args, tree, X_df, Y_df)
    except Exception as e:
        print(f'Fitting in CVGP not successful: {e}')
        traceback.format_exc()
        output={}
    return output


def run(args):
    from src.experiment_schedule import run_experiments
    print("Load Model")
    load_model_func = load_model
    run_experiments(args, load_model_func, 'CVGP')


if __name__ == '__main__':
    print("Start")
    parser = ConfigSyntaxTree.arguments_parser()
    parser = ConfigCVGP.arguments_parser(parser)
    parser = ConfigGeneral.arguments_parser(parser)
    parser = ConfigPySR.arguments_parser(parser)
    args = parser.parse_args()

    run(args)
