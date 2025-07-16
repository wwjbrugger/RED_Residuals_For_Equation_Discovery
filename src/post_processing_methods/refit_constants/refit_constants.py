import pandas as pd
import re

from src.evaluate_equation import evaluate_equation
from src.syntax_tree.syntax_tree import SyntaxTree
import time

def refit_constants(args, prefix, X_df, Y_df):
    output = {}
    start_time_refit_constant = time.time()
    try:
        tree_refit = SyntaxTree(args=args, grammar=None)
        prefix_c = replace_numbers_with_c(prefix)
        tree_refit.prefix_to_syntax_tree(prefix_c.split())
        output = evaluate_equation(args, tree_refit, X_df, Y_df)
        replace_c_with_numbers(output, tree_refit)
    except Exception as e:
        output['equation'] = f'Error: {e}'
    output['time'] = time.time() -  start_time_refit_constant
    return output

def replace_numbers_with_c(text):
    return re.sub(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b', 'c', text)

def replace_c_with_numbers(output, tree_refit):
    output['prefix_c'] =  output['prefix']
    prefix = tree_refit.start_node.math_class.prefix_notation(
        call_node_id=-1,
        kwargs=tree_refit.constants_in_tree
    )
    output['prefix'] = prefix