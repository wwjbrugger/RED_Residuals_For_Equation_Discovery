import traceback

from sklearn.model_selection import train_test_split

from src.post_processing_methods.residual.calculate_residual import residual_for_one_node
import numpy as np
import time

from src.evaluate_equation import map_equation_to_syntax_tree, test_equation
from src.post_processing_methods.postprocessing_gp.postprocessing_gp import gp_for_one_node
from src.post_processing_methods.residual.get_list_residual_nodes import get_list_residual_nodes


def perm_experiment(X, y, dataset_name, model, num_residuals, results):
    results[dataset_name]['permute']['train'] = {}
    successful_classic_runs = 0
    retries = 10
    begin_time_single_experiment = time.time()

    while successful_classic_runs < num_residuals and retries > 0:
        if begin_time_single_experiment + 240 < time.time():
            print("Permute experiment stopt after 4 min")
            return
        try:
            start_time_permute = time.time()
            shuffled_indices = np.random.permutation(len(X))
            shuffled_X = X.iloc[shuffled_indices].reset_index(drop=True)
            shuffled_y = y.iloc[shuffled_indices].reset_index(drop=True)
            output = model(X_df=shuffled_X, Y_df=shuffled_y)
            output['time'] =  time.time() - start_time_permute
            results[dataset_name]['permute']['train'][successful_classic_runs] = output
            successful_classic_runs += 1
        except:
            print('Rerun permute experiment')
            retries -= 1


def residual_experiments_all_nodes(X, y, args, dataset_name, model, results, prefix_equ):
    results[dataset_name]['residuals']['train'] = {}
    tree = map_equation_to_syntax_tree(args, prefix_equ, infix=False)
    num_residuals = 0
    begin_time_single_experiment = time.time()
    for node_id in tree.dict_of_nodes.keys():
        if begin_time_single_experiment + 240 < time.time():
            print("Residual experiment stopt after 4 min")
            return num_residuals
        start_time_residual = time.time() 
        if not node_id in [-1, 0]:
            try:
                print(f'               residuals for node {node_id}')
                num_residuals, output = residual_for_one_node(
                    X, y,
                    args,
                    model,
                    node_id,
                    num_residuals,
                    tree
                )
                output['time'] = time.time() - start_time_residual
                results[dataset_name]['residuals']['train'][node_id] = output
            except (SyntaxError, RuntimeError):
                print(traceback.format_exc())
                continue


    return num_residuals

def residual_experiments_operator(X, y, args, dataset_name, model, results,
                                  prefix_equ,X_val=None, y_val=None):
    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42
        )
    else:
        X_train = X
        y_train = y

    results[dataset_name]['residuals']['train'] = {}
    num_residuals, last_changed_node = 0, 0
    output={}
    try:
        tree = map_equation_to_syntax_tree(args, prefix_equ, infix=False)
    except (SyntaxError, RuntimeError) as E:
        print(traceback.format_exc())
        results[dataset_name]['residuals']['train'][0] = ('Translation to syntax'
                                                          'tree not possible')
        return num_residuals
    lowest_error = results[dataset_name]['classic']['train'][0]['error']
    next_residual_nodes = get_list_residual_nodes(tree, last_changed_node)
    start_time_residual = time.time()

    while ((num_residuals < args.max_num_residuals) and (lowest_error > 1E-10) and
           (len(next_residual_nodes) > 0)):
        try:
            print(f"Residual run number: {num_residuals}")
            node_id = next_residual_nodes.pop(0)
            num_residuals, output = residual_for_one_node(
                X_train, y_train,
                args,
                model,
                node_id,
                num_residuals,
                tree
            )
            if 'error' not in output:
                continue

            val_output = test_equation(
                args=args,
                prefix_equ=output['prefix'],
                X_test=X_val,
                y_test=y_val
            )
            if not 'error' in val_output:
                continue
            if val_output['error'] < lowest_error:
                lowest_error = val_output['error']
                tree = map_equation_to_syntax_tree(args, output['prefix'], infix=False)
                last_changed_node = node_id
                next_residual_nodes =get_list_residual_nodes( tree,last_changed_node)
                output['time'] = time.time() - start_time_residual
                results[dataset_name]['residuals']['train'][num_residuals] = output
                start_time_residual = time.time()
        except (SyntaxError, RuntimeError) as E:
            print(traceback.format_exc())
            continue
    output['time'] = time.time() - start_time_residual
    results[dataset_name]['residuals']['train']['last'] = output

    return num_residuals


def postprocessing_gp_experiment(X, y, args, dataset_name, results, prefix_equ):
    results[dataset_name]['post_gp']['train'] = {}
    try:
        tree = map_equation_to_syntax_tree(args, prefix_equ, infix=False)
    except (SyntaxError, RuntimeError) as E:
        print(traceback.format_exc())
        results[dataset_name]['post_gp']['train'][0] = ('Parsing to syntax'
                                                          'tree not possible')

    begin_time_single_experiment = time.time()
    num_gp = 0
    for node_id in tree.dict_of_nodes.keys():
        if begin_time_single_experiment + 240 < time.time():
            print("postprocessing gp experiment stopt after 4 min")
            return num_gp
        if not node_id in [-1, 0]:
            print(f'         gp for node {node_id}')
            try:
                results[dataset_name]['post_gp']['train'][node_id] = \
                    gp_for_one_node(X, y, args, node_id, tree)
            except (SyntaxError, RuntimeError) as E:
                print(traceback.format_exc())
                results[dataset_name]['residuals']['train'][node_id] = E
                
