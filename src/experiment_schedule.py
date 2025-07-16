from sklearn.model_selection import train_test_split

from src.evaluate_equation import test_approach
from src.experiments import (perm_experiment,
                             postprocessing_gp_experiment,
                             residual_experiments_operator)
from src.post_processing_methods.hyperparameter.hyperparameter import different_hyperparameter
from src.preprocess import preprocess_data, get_info_of_equation, get_datasets_files
from src.post_processing_methods.refit_constants.refit_constants import refit_constants
from src.utils.utils import save_result_dict, load_result_dict
import time
import numpy as np
import torch
import random


def run_experiments(args, load_model_func, approach):
    files_path, equation_info = get_datasets_files(args)
    model = load_model_func(args)
    if len(args.path_to_old_results) > 0:
        results = load_result_dict(args.path_to_old_results)
        previous_time = results['runtime']
    else:
        results = {}
        previous_time = 0
    add_args_to_results(args, results)
    time_stamp = time.strftime('%d_%b_%Y_%H:%M:%S')
    start_time = time.time()
    for i, file_path in enumerate(files_path):
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(i)
        X, y, dataset_name = preprocess_data(args, file_path)
        if dataset_name in results:
            continue
        if not X is None:
            X, X_test, y, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42
            )
            info = get_info_of_equation(args, dataset_name, equation_info)
            print(f"Run experiment {file_path}")
            results[dataset_name] = {}
            print('Begin: Classic')

            start_time_classic = time.time()
            output_classic = model(X_df=X, Y_df=y, info=info)
            if not 'prefix' in output_classic:
                results[dataset_name] = {}
                results[dataset_name]['equation'] = 'Error'
                continue
            results[dataset_name]['classic'] = {}
            results[dataset_name]['classic']['train'] = {}
            results[dataset_name]['classic']['train'][0] = output_classic
            results[dataset_name]['classic']['train'][0]['time']= \
                time.time() - start_time_classic
            print(f"Classic Equation: {output_classic['infix']}")

            test_approach(
                results,
                dataset_name,
                X_test,
                y_test,
                args,
                approach='classic'
            )
            if approach == 'CVGP' or args.only_classic:
                # For the Control Variables Approach we don't want to test all approaches.
                results['runtime'] = time.time() - start_time + previous_time
                save_result_dict(args, time_stamp, approach, results)
                continue

            if output_classic['error'] > 0.001:
                ###############################################################
                ################### Post GP ###################################
                ###############################################################
                results[dataset_name]['post_gp'] = {}
                print('Begin: Post_gp ')
                postprocessing_gp_experiment(
                    X, y,
                    args,
                    dataset_name,
                    results,
                    output_classic['prefix']
                )

                test_approach(
                    results,
                    dataset_name,
                    X_test,
                    y_test,
                    args,
                    approach='post_gp'
                )
                ###############################################################
                ################### Hyperparameter ############################
                ###############################################################
                results[dataset_name]['hyperparameter'] = {}
                print('Begin: Hyperparameter ')
                output_hyperparameter = different_hyperparameter(
                    args,
                    load_model_func,
                    X,
                    y,
                    equation_info=info
                )
                results[dataset_name]['hyperparameter']['train'] = output_hyperparameter
                test_approach(
                    results,
                    dataset_name,
                    X_test,
                    y_test,
                    args,
                    approach='hyperparameter'
                )

                ###############################################################
                ################### Refit Constant ############################
                ###############################################################
                results[dataset_name]['refit_constants'] = {}
                print('Begin: Refit Constant ')
                results[dataset_name]['refit_constants']['train'] = {}
                output_refit = refit_constants(args, output_classic['prefix'], X_df=X, Y_df=y)
                results[dataset_name]['refit_constants']['train'][0] = output_refit
                test_approach(
                    results,
                    dataset_name,
                    X_test,
                    y_test,
                    args,
                    approach='refit_constants'
                )

                ###############################################################
                ################### Residuals #################################
                ###############################################################
                results[dataset_name]['residuals'] = {}
                print('Begin: Residuals ')
                num_residuals = residual_experiments_operator(
                    X_train,
                    y_train,
                    args,
                    dataset_name,
                    model,
                    results,
                    output_classic['prefix'],
                    X_val=X_val,
                    y_val=y_val
                )
                test_approach(
                    results,
                    dataset_name,
                    X_test,
                    y_test,
                    args,
                    approach='residuals'
                )

                if num_residuals > 0:
                    ###############################################################
                    ################### Permute ###################################
                    ###############################################################
                    results[dataset_name]['permute'] = {}
                    print('Begin: Permute ')
                    perm_experiment(
                        X, y,
                        dataset_name,
                        model,
                        num_residuals,
                        results
                    )
                    test_approach(
                        results,
                        dataset_name,
                        X_test,
                        y_test,
                        args,
                        approach='permute'
                    )

            results['runtime'] = time.time() - start_time + previous_time
            save_result_dict(args, time_stamp, approach, results)
        else:
            results[dataset_name] = {}
            results[dataset_name]['equation'] = 'To many variables'
    results['finished'] = 'True'
    save_result_dict(args, time_stamp, approach, results)


def add_args_to_results(args, results):
    for parameter, value in vars(args).items():
        args_parameter = f"args_{parameter}"
        if args_parameter in results and not parameter == 'path_to_old_results':
            if str(value) != results[args_parameter]:
                raise AssertionError(f'{parameter}: Loaded parameter {results[args_parameter]} '
                                     f' do not match with given parameter {str(value)} ')
        results[args_parameter] = str(value)
