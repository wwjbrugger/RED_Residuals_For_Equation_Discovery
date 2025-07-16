import unittest

import pandas as pd

from src.syntax_tree.state import State
from src.syntax_tree.syntax_tree import  SyntaxTree
from pcfg import PCFG
from src.post_processing_methods.residual.get_residual_of_equation import  get_residual_of_equation
import numpy as np
from src.utils.logging import get_log_obj


def get_empty_list(arg):
    return []


class Get_residual_of_equation(unittest.TestCase):
    def setUp(self) -> None:
        grammar_string = """
                    S -> '*' Constant  S [0.1]
                                S -> '+' Constant  S [0.1]
                                S -> '<*' Variable S_   [0]
                                S -> '*' A A  [0]
                                S -> '-' '*' A A '1' [0]
                                S -> '+' A A [0]
                                S -> Value [0.8]
                                S_ -> '+' '*' Variable S_ '1' [0.5]
                                S_ -> '+' Variable '1'  [0.5]

                                Variable -> 'x_0' [0.2]
                                Variable -> 'x_1' [0.2]
                                Variable -> 'x_2' [0.2]
                                Variable -> 'x_3' [0.2]
                                Variable -> 'x_4' [0.2]

                                Constant -> 'c' [1]



                                A -> Trig [0.3]
                                A -> Value [0.3] 
                                A -> 'log' Value  [0.4] 
                                Value -> Variable [0.2]
                                Value -> '**' '2' Variable [0.2]
                                Value -> '+' Variable '1' [0.2]
                                Value -> '+' '**' '2' Variable '1' [0.1]
                                Value -> '+' '**' '2' Variable Variable [0.1]
                                Value -> '**' '0.5' Variable [0.1]
                                Value -> '2' [0.1]
                                Trig -> 'sin'  Value [0.5]| 'cos' Value [0.5]
                    """

        self.grammar = PCFG.fromstring(grammar_string)

        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 10
        self.args.max_constants_in_tree = 5
        self.args.number_equations = 10
        self.args.max_num_nodes_in_syntax_tree = 30
        self.args.num_calls_sampling = 10
        self.args.sample_with_noise = False
        self.args.how_to_select_node_to_delete = 'random'

        self.args.precision = 'float32'
        self.logger = get_log_obj(args=self.args, name='test_logger')

    # def test_print(self):
    #     syntax_tree = SyntaxTree(grammar=None, args=self.args)
    #     syntax_tree.prefix_to_syntax_tree(prefix='+ 1 cos S_'.split())
    #     syntax_tree.print()
    #     y, equation = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=-1)
    #     print(equation)

    def test_0(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ c S'.split())
        syntax_tree.nodes_to_expand.append(1025)
        dataset = np.array([
            [- 10.08902, - 10.17091, 0.215567, 0.144374, 10.0676, - 9.089025],
            [- 7.930019, - 9.765913, 1.442015, 0.152672, 8.89528, - 6.930019],
            [- 5.511989, - 9.767252, 2.458630, 0.353149, 7.76481, - 4.511989],
            [- 3.271325, - 9.527846, 3.300389, 0.583482, 6.78344, - 2.271325],
            [- 1.169702, - 9.483120, 4.553673, 0.426493, 5.84703, - 0.169702],
            [1.076235, - 9.438329, 5.722373, 0.766701, 4.58014, 2.076235],
        ])
        data_frame = pd.DataFrame(dataset, columns=['x_4', 'x_3', 'x_0', 'x_2', 'x_1', 'y'])

        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1025,
            'last_symbol': 'S'
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        # The fitted constant should be the average of all y values
        mean_y = np.mean(dataset[ :, -1])
        soll_residual_y = dataset[ :, -1] - mean_y
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertEqual('S'.split(), reduced_state.syntax_tree.rearrange_equation_prefix_notation(-1)[1].split())
        np.testing.assert_almost_equal(soll_residual_y, reduced_state.observation['data_frame'].to_numpy()[ :, -1], decimal=4)


