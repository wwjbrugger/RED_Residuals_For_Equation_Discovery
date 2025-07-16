import pandas as pd
from sympy import sympify

from src.evaluate_equation import infix_to_prefix
from src.utils.error import NotInvertibleError, NoSolutionFoundError
from src.utils.logging import logging_residuals
from src.syntax_tree.syntax_tree import SyntaxTree
import traceback
import time

def residual_for_one_node(X, y, args, model, node_id, num_residuals, tree):
    output = {}
    start_time_residual = time.time()
    try:
        num_residuals += 1
        res = tree.residual(node_id, dataset=pd.concat([X, y], axis=1))

        output = model(X_df=X, Y_df=pd.Series(res, name='y', index = X.index))
        if output == {}:
            output['equation'] = f'Error model did not return a suggestion'
        else:
            output= get_tree_representations(
                args,
                node_id,
                output,
                tree
            )

    except (Exception) as e:
        output['equation'] =  f'Error {e}'
        traceback.print_exc()
        print(f"Tree is:  {tree.get_subtree_in_prefix_notion(0)}")
    output['time'] =  time.time() - start_time_residual
    return num_residuals, output


def get_tree_representations(args, node_id, output, tree):
    output['original_equation_infix'] = tree.rearrange_equation_infix_notation(-1)[1]
    output['residual_equation'] = tree.rearrange_equation_infix_notation(node_id)[1]
    output['old_X'] = tree.get_subtree_in_prefix_notion(node_id=node_id)
    output['new_X_infix']= output['infix']
    output['new_X_prefix'] = output['prefix']

    output['eq_placeholder_prefix'], output['eq_placeholder_infix'] =\
        tree.get_eq_with_placeholder(node_id=node_id)
    output['infix_before_sympy'] = output['eq_placeholder_infix'].replace('X', output['infix'])
    output['infix'] = str(sympify(output['infix_before_sympy']))

    output['prefix'] = infix_to_prefix(output['infix'], args)

    complete_res_tree = SyntaxTree(args=args, grammar=None)
    complete_res_tree.prefix_to_syntax_tree(output['prefix'].split())
    output['num_operations'] =complete_res_tree.num_inner_nodes()
    return output


