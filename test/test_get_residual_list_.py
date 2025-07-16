import unittest

from src.evaluate_equation import map_equation_to_syntax_tree
from src.post_processing_methods.residual.get_list_residual_nodes import get_list_residual_nodes
from src.utils.logging import get_log_obj


def get_empty_list(arg):
    return []


class TestSyntaxTree(unittest.TestCase):
    def setUp(self) -> None:
        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 14
        self.args.max_constants_in_tree = 5
        self.args.number_equations = 10
        self.args.num_calls_sampling = 10
        self.args.max_num_nodes_in_syntax_tree = 30
        self.args.sample_with_noise = False
        self.args.how_to_select_node_to_delete = 'random'

        self.args.precision = 'float32'
        self.logger = get_log_obj(args=self.args, name='test_logger')


    def test_tree(self):
        string = ' abs  abs  *  sin x_2  **  abs  +  +  cos x_2  abs  +  **  / -32.43219838248628 x_2  0.5   cos x_2   x_1   abs  sin x_2  '
        tree =  map_equation_to_syntax_tree(self.args, string, infix=False,
                                           catch_exceptions=True)
        list_residual_nodes = get_list_residual_nodes(tree,last_changed_node_id=0)
        self.assertEqual(
            list_residual_nodes,
            [1.0, 2.0, 3.0, 2050.0, 4.0, 2051.0, 3074.0, 2052.0, 3075.0, 2053.0, 2308.0, 3076.0, 2054.0, 2181.0, 2055.0, 2182.0, 2183.0, 2214.0, 2184.0, 2199.0, 2215.0, 2185.0, 2192.0])


    def test_tree_2(self):
        string = ' abs  abs  *  sin x_2  **  abs  +  +  cos x_2  abs  +  **  / -32.43219838248628 x_2  0.5   cos x_2   x_1   abs  sin x_2  '
        tree =  map_equation_to_syntax_tree(self.args, string, infix=False,
                                           catch_exceptions=True)
        list_residual_nodes = get_list_residual_nodes(tree,last_changed_node_id=2050.0)
        self.assertEqual(
            list_residual_nodes,
            [3.0, 4.0, 2051.0, 3074.0, 2052.0, 3075.0, 2053.0, 2308.0, 3076.0, 2054.0, 2181.0, 2055.0, 2182.0, 2183.0, 2214.0, 2184.0, 2199.0, 2215.0, 2185.0, 2192.0])


        
