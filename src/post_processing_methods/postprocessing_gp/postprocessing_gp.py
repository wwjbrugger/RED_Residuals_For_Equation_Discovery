from sympy import sympify

from src.fit_func_gplearn import load_model
from src.syntax_tree.syntax_tree import SyntaxTree
from src.utils.error import NotInvertibleError, NoSolutionFoundError
from src.utils.logging import logging_gp
import traceback
import time


def gp_for_one_node(X, y, args, node_id, tree):
    output={}
    start_time_gp = time.time()
    try:
        eq_with_placeholder_prefix,_ = tree.get_eq_with_placeholder(node_id)
        start_equation = prefix_to_gp_learn_string(eq_with_placeholder_prefix)
        output = gp_get_old_tree_repr(args, node_id, output, tree)

        gp_model = load_model(args)
        output_gp = gp_model(
            X_df=X,
            Y_df=y,
            start_population=[start_equation for i in range(500)]
        )
        output.update(output_gp)

    except Exception as e:
        output['equation'] = f'Error {e}'
        traceback.print_exc()
        print(f"Tree is:  {tree.get_subtree_in_prefix_notion(0)}")
    output['time'] = time.time() - start_time_gp
    return output


def prefix_to_gp_learn_string(prefix):
    prefix = prefix.replace('+', 'add' )
    prefix = prefix.replace('-', 'sub')
    prefix = prefix.replace('**', 'pow')
    prefix = prefix.replace('*', 'mul')
    prefix = prefix.replace('/', 'div')
    prefix = prefix.replace('X', '1')
    prefix = prefix.replace('abs', 'Abs')
    return prefix



def gp_get_old_tree_repr(args, node_id, output, tree):
    output['original_equation_infix'] = tree.rearrange_equation_infix_notation(-1)[1]
    output['eq_placeholder_prefix'], output['eq_placeholder_infix'] =\
            tree.get_eq_with_placeholder(node_id=node_id)
    output['old_X'] = tree.get_subtree_in_prefix_notion(node_id=node_id)
    return output