import pandas as pd

from src.evaluate_equation import infix_to_prefix, map_equation_to_syntax_tree
import unittest

from src.fit_func_cvgp import cvgp_to_prefix


class TestCVGP(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_cvgp_to_prefix(self):
        """
        Test that it can sum a list of integers
        """
        cvpg = ["mul","mul","mul","pow","4.826768564397918","pow","3.3099267942475197"
            ,"mul","7.671691189551916","X_0","div","X_2","X_2","X_1","mul",
         "pow","mul","pow","mul","mul","pow","0.4508583245629594","pow","X_1",
         "3.8001612629599837","X_0","4.351134549822245","pow",
         "1.199447230634666","X_0","4.164075975296017","add","sin","mul",
         "mul","pow","2.9058990922032955","pow","1.7393141302629833","X_1",
         "mul","mul","pow","8.667925469052044","pow","0.11653898681776464",
         "X_1","X_0","3.102641","0.6555449068731534","X_1","1.6862292095938514"]

        prefix = cvgp_to_prefix(cvpg)
        tree = map_equation_to_syntax_tree(self.args, prefix, infix=False,
                                           catch_exceptions=True)


#
    def test_cvgp_to_prefix_2(self):
        """
        Test that it can sum a list of integers
        """
        cvpg = ["add","exp","exp","sin","add","exp","exp","2.666423015340686","sub","exp","mul","exp","X_0","4.579859857320913","add","7.8091436470006625","7.239589568892376","sub","exp","mul","exp","X_0","exp","pow","div","exp","7.295312824160691","3.1900677206475523","exp","8.777541960990426","add","6.842478236115619","1.2310332564721582"]

        prefix = cvgp_to_prefix(cvpg)
        tree =  map_equation_to_syntax_tree(self.args, prefix, infix=False,
                                           catch_exceptions=True)

    def test_cvgp_to_prefix_3(self):
        """
        Test that it can sum a list of integers
        """
        cvpg = ["mul","add","mul","sin","inv","X_0","mul","div","X_2","X_0","mul","X_1","-5.412403954477635e-10","X_1","mul","X_0","X_2"]

        prefix = cvgp_to_prefix(cvpg)
        tree =  map_equation_to_syntax_tree(self.args, prefix, infix=False,
                                           catch_exceptions=True)
        dataset = pd.DataFrame({'x_0':[1,2,3],
                      'x_1':[1,2,3],
                      'x_2':[1,2,3]})
        result = tree.evaluate_subtree(0,dataset)
        self.assertEqual(
            list(result),
            [-1554708390.8445423, -6218833559.378169, -13992375499.60088])

if __name__ == '__main__':
    unittest.main()
