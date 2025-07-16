import unittest

import pandas as pd

from src.evaluate_equation import map_equation_to_syntax_tree
from src.syntax_tree.state import State
from src.syntax_tree.syntax_tree import SyntaxTree
from pcfg import PCFG
from src.post_processing_methods.residual.get_residual_of_equation import get_residual_of_equation
import numpy as np
from src.utils.logging import get_log_obj


def get_empty_list(arg):
    return []


class Get_residual_of_equation(unittest.TestCase):
    def setUp(self) -> None:
        grammar_string = grammar_string = """
                    S -> Constant [0.2]
                    S -> Variable [0.1]
                    S -> '+' S S [0.2]
                    S -> '-' S S [0.1]
                    S -> '*' S S [0.1]
                    S -> '/' S S [0.1]
                    S -> 'sin' S_ [0.1]
                    S -> 'cos' S_ [0.1]
                    Constant -> '0.5' [0.15]
                    Constant -> '2' [0.2]
                    Constant -> '3' [0.2]
                    Constant -> '4' [0.15]
                    Constant -> '5' [0.15]
                    Constant -> '6' [0.15]
                    Variable -> 'x_1' [1.0]
                    S_ -> Constant [0.2]
                    S_ -> '+' S_ S_ [0.4]
                    S_ -> '-' S_ S_ [0.2]
                    S_ -> '*' S_ S_ [0.1]
                    S_ -> '/' S_ S_ [0.1]
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
        self.args.num_calls_sampling = 10
        self.args.max_num_nodes_in_syntax_tree = 30
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
        # y = (x + 1) + (x + 1)
        syntax_tree.prefix_to_syntax_tree(prefix='+ + x 1 S'.split())
        data_frame = pd.DataFrame(list(zip([0, 1, 2, 3], [2, 4, 6, 8])), columns=['x', 'y'])
        residual = np.array(list(zip([0, 1, 2, 3], [1, 2, 3, 4])))
        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1025,
            'last_symbol': 'S'
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertEqual('S'.split(), reduced_state.syntax_tree.rearrange_equation_prefix_notation(-1)[1].split())
        self.assertTrue(np.array_equal(reduced_state.observation['data_frame'].to_numpy(), residual))

    def test_1(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        # y = (x + 1) + (x + 1)
        syntax_tree.prefix_to_syntax_tree(prefix='+ + x 1 S'.split())
        data_frame = pd.DataFrame(list(zip([0, 1, 2, 3], [2, 4, 6, 8])), columns=['x', 'y'])
        residual = np.array(list(zip([0, 1, 2, 3], [1, 2, 3, 4])))
        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1025,
            'last_symbol': 'S'
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)

        self.assertTrue(np.array_equal(reduced_state.observation['data_frame'].to_numpy(), residual))

    def test_2(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        # y = (x + 1) + (x + (0 * x))
        syntax_tree.prefix_to_syntax_tree(prefix='+ + x 1 + x S'.split())
        data_frame = pd.DataFrame(list(zip([0, 1, 2, 3], [1, 3, 5, 7])), columns=['x', 'y'])
        residual = np.array(list(zip([0, 1, 2, 3], [0, 0, 0, 0])))
        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1538,
            'last_symbol': ''
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertTrue('+ x S'.split(), reduced_state.syntax_tree.rearrange_equation_prefix_notation(-1)[0].split())
        reduced_state_2 = get_residual_of_equation(state=reduced_state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertTrue(np.array_equal(reduced_state_2.observation['data_frame'].to_numpy(), residual))

    def test_3(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        # y = (x + 1) + (x + 1)
        syntax_tree.prefix_to_syntax_tree(prefix='+ 1 cos S_'.split())
        data_frame = pd.DataFrame(list(zip([0, 1, 2, 3], [2, 4, 6, 8])), columns=['x', 'y'])
        residual = np.array(list(zip([0, 1, 2, 3], [1, 2, 3, 4])))
        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 513,
            'last_symbol': 'S_'
        }
        self.assertEqual(syntax_tree.nodes_to_expand, [513])

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        reduced_str = reduced_state.syntax_tree.rearrange_equation_prefix_notation(-1)[1]
        self.assertEqual(reduced_str.split(), 'cos S_'.split())
        self.assertFalse(reduced_state.syntax_tree.dict_of_nodes[513].invertible)
        reduced_state_2 = get_residual_of_equation(state=reduced_state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertEqual(reduced_state, reduced_state_2)

    def test_logarithm(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        # y = (x + 1) + (x + 1)
        syntax_tree.prefix_to_syntax_tree(prefix='log S'.split())
        data_frame = pd.DataFrame(list(zip([5, 1, 2, 3], [2000000000, 4, 6, 8])), columns=['x', 'y'])
        observation = {
            'data_frame': data_frame,
            'current_tree_representation_str': syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1],
            'current_tree_representation_int': [0, 0, 1],
            'id_last_node': 1026,
            'last_symbol': 'S_'
        }

        state = State(syntax_tree=syntax_tree, observation=observation, done=False)
        reduced_state = get_residual_of_equation(state=state, function_to_get_current_tree_representation_int=get_empty_list, logger=self.logger)
        self.assertEqual(state, reduced_state)

    def test_abs(self):
        dataset = pd.DataFrame({'x_0': [1, 2, 3],
                                'x_1': [1, 2, 3],
                                'x_2': [1, 2, 3],
                                'y': [1, 2, 3]})
        tree = map_equation_to_syntax_tree(
            self.args,
            '(  ( x_0 )  /  abs (  ( x_1 )  )  )',
            infix=True,
            catch_exceptions=True
        )
        tree.print()
        print(tree.rearrange_equation_infix_notation(512))
        tree.residual(512, dataset=dataset)
        self.assertEqual(
            ('abs', '(  ( x_0 )  /  ( y )  ) '),
            tree.rearrange_equation_infix_notation(512))
        tree.residual(513, dataset=dataset)
        self.assertEqual(
            ('x_1', '(  ( x_0 )  /  ( y )  ) '),
            tree.rearrange_equation_infix_notation(513))


