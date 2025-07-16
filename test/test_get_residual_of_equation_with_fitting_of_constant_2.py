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
                   S -> '*' S S [0.2]
                    S -> '+' S S [0.2]
                    S -> '/' S S [0.18]
                    S -> '-' S S [0.1]
                    S -> Variable  [0.1]
                    S -> Constant [0.02]
                    S -> Trig [0.1] 
                    S -> 'log' Value  [0.1] 
                    
                    Trig -> 'sin'  Value [0.5]| 'cos' Value [0.5]
                   
                    Value -> Variable [0.2]
                    Value -> '**' '2' Variable [0.2]
                    Value -> '+' Variable Constant [0.2]
                    Value -> '+' '**' '2' Variable Constant [0.2]
                    Value -> '+' '**' '2' Variable Variable [0.2]
                    
                    Constant -> 'c' [1]
                  
                    

                    
                    Variable -> 'x_0' [0.2]
                    Variable -> 'x_1' [0.2]
                    Variable -> 'x_2' [0.2]
                    Variable -> 'x_3' [0.2]
                    Variable -> 'x_4' [0.2]
                    """

        self.grammar = PCFG.fromstring(grammar_string)

        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 30
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 7
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



    def test_reset_unset_variables(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix=' -  log   +  ** 2 x_0  c    S '.split())
        syntax_tree.nodes_to_expand.append(130)
        syntax_tree.nodes_to_expand.append(194)

        dataset = np.array([
            [0.3, 10., 0.1, -10., -10., -0.86196005],
            [1.3777778, 8.922222, 0.2, -9.888889, -7.7777777, -100],
            [2.4555554, 7.8444443, 0.3, -9.777778, -5.5555553, -2.2842891],
            [3.5333333, 6.766667, 0.4, -9.666667, -3.3333333, 2.0207396],
            [4.611111, 5.688889, 0.5, -9.555555, -1.1111112, 0.48979977],
            [5.688889, 4.611111, 0.6, -9.444445, 1.1111112, 0.5532453],
            [6.766667, 3.5333333, 0.7, -9.333333, 3.3333333, -1.7410758],
            [7.8444443, 2.4555554, 0.8, -9.222222, 5.5555553, 1.402989],
            [8.922222, 1.3777778, 0.9, -9.111111, 7.7777777, 0.0797143],
            [10., 0.3, 1., -9., 10., 0.83989394]])
        data_frame =pd.DataFrame(dataset, columns = ['x_4', 'x_3', 'x_0', 'x_2', 'x_1', 'y'])

        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1025,
            'last_symbol': 'S'
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        # The fitted constant should be the average of all y values
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertEqual(state.syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
                         ' -  log   +  ** 2 x_0  c   S')


    def test_fitting_with_two_constants(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix=' -    +  ** 2 x_0  c    S' .split())

        dataset = np.array([
            [0.3, 10., 0.1, -10., -10., -0.86196005],
            [1.3777778, 8.922222, 0.2, -9.888889, -7.7777777, -100],
            [2.4555554, 7.8444443, 0.3, -9.777778, -5.5555553, -2.2842891],
            [3.5333333, 6.766667, 0.4, -9.666667, -3.3333333, 2.0207396],
            [4.611111, 5.688889, 0.5, -9.555555, -1.1111112, 0.48979977],
            [5.688889, 4.611111, 0.6, -9.444445, 1.1111112, 0.5532453],
            [6.766667, 3.5333333, 0.7, -9.333333, 3.3333333, -1.7410758],
            [7.8444443, 2.4555554, 0.8, -9.222222, 5.5555553, 1.402989],
            [8.922222, 1.3777778, 0.9, -9.111111, 7.7777777, 0.0797143],
            [10., 0.3, 1., -9., 10., 0.83989394]])

        observation = {
            'data_frame': pd.DataFrame(dataset, columns = ['x_4', 'x_3', 'x_0', 'x_2', 'x_1', 'y'] ),
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1025,
            'last_symbol': 'S'
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        # The fitted constant should be the average of all y values

        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        reduced_state.syntax_tree.expand_node_with_action(node_id=syntax_tree.nodes_to_expand[0], action=5)
        reduced_state.syntax_tree.expand_node_with_action(node_id=syntax_tree.nodes_to_expand[0], action=15)
        result = reduced_state.syntax_tree.evaluate_subtree(
            node_id=reduced_state.syntax_tree.start_node.node_id,
            dataset=pd.DataFrame(dataset, columns = ['x_4', 'x_3', 'x_0', 'x_2', 'x_1', 'y'])
        )
        np.testing.assert_almost_equal(
            reduced_state.syntax_tree.constants_in_tree['c_0']['value'], -10.335, decimal=4)
        np.testing.assert_almost_equal(
            reduced_state.syntax_tree.constants_in_tree['c_1']['value'], - 9.9500, decimal=4)






