import json

import numpy as np
import pandas as pd
from definitions import ROOT_DIR
from pathlib import Path
import os


def get_datasets_files(args):
    if args.dataset_folder == 'datasets_srbench':
        files_path, pd_equations = read_srbench_data(args)
    elif args.dataset_folder == 'datasets_dso':
        files_path, pd_equations = read_dso_data(args)
    elif args.dataset_folder == 'datasets_dso_1000':
        files_path, pd_equations = read_dso_1000_data(args)
    else:
        raise NotImplementedError
    return files_path,  pd_equations.to_dict('index')


def read_srbench_data(args):
    file_path = f'{ROOT_DIR}/{args.dataset_folder}'
    files_path = [Path(file_path) / f / f"{f}.tsv.gz" for f in os.listdir(file_path)
                  if os.path.isdir(os.path.join(file_path, f))]
    pd_bonus_equation = pd.read_csv(f'{ROOT_DIR}/datasets_srbench/BonusEquations.csv')
    pd_feynman_equation = pd.read_csv(f'{ROOT_DIR}/datasets_srbench/FeynmanEquations.csv')
    pd_equations = pd.concat([pd_bonus_equation, pd_feynman_equation], axis=0)
    pd_equations = pd_equations.set_index('Filename')
    return files_path, pd_equations

def read_dso_data(args):
    file_path = f'{ROOT_DIR}/{args.dataset_folder}'
    files_path = [Path(file_path) / f for f in os.listdir(file_path)
                  if f.startswith('data')]
    pd_equation_info = pd.read_csv(f'{ROOT_DIR}/datasets_dso/benchmarks.csv')
    pd_equations = pd_equation_info.set_index('name')
    return files_path, pd_equations

def read_dso_1000_data(args):
    file_path = f'{ROOT_DIR}/{args.dataset_folder}'
    files_path = [Path(file_path) / f for f in os.listdir(file_path)
                  if f.startswith('data')]
    pd_equation_info = pd.read_csv(f'{ROOT_DIR}/datasets_dso_1000/benchmarks.csv')
    pd_equations = pd_equation_info.set_index('name')
    return files_path, pd_equations


def add_noise(args, df):
    if args.noise_factor > 0:
        noise_factor = args.noise_factor
        noise = np.random.uniform(-noise_factor, noise_factor, df.shape)
        df_noisy = df * (1 + noise)
        return df_noisy
    else:
        return df



def preprocess_data(args, file_path):
    dataset_name, df = read_data_set_from_disc(file_path)
    if len(list(df.columns)) > 4:
        return None, None, dataset_name
    rename_columns(df)
    df = df.sample(min(df.shape[0],args.max_dataset_size))
    df = add_noise(args, df)
    X, y = split_data_set(df)
    return X, y, dataset_name


def split_data_set(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def rename_columns(df):
    new_columns = [f"x_{i}" for i in range(-1 + len(list(df.columns)))]
    new_columns.append('y')
    df.columns = new_columns


def read_data_set_from_disc(file_path):
    dataset_name = file_path.name
    if '.gz' in dataset_name:
        df = pd.read_csv(file_path, compression='gzip', sep='\t')
    else:
        df = pd.read_csv(file_path, header=None)

    return dataset_name, df


def get_info_of_equation(args, dataset_name, equation_info):
    if args.dataset_folder == 'datasets_srbench':
        pd_index = dataset_name.replace('feynman_', '')
        pd_index = pd_index.replace('.tsv.gz', '')
        pd_index = pd_index.replace('_', '.')
        pd_index = pd_index.replace('test.', 'test_')
        info = equation_info[pd_index]   # 'I.15.10'
    elif args.dataset_folder == 'datasets_dso' or args.dataset_folder == 'datasets_dso_1000':
        id = dataset_name.split('_')[1]
        info = equation_info[id]
        info['Formula'] = info['expression']
        variable_dict = json.loads(info['train_spec'])
        if 'all' in variable_dict:
            for i in range(info['variables']):
                k = list(variable_dict['all'].keys())[0]
                range_list = variable_dict['all'][k]
                info[f'v{i + 1}_name'] = f'x{i+1}'
                info[f'v{i+1}_low'] = range_list[0]
                info[f'v{i+1}_high'] = range_list[1]
        else:
            for i, variable_info in enumerate(variable_dict.values()):
                k = list(variable_info.keys())[0]
                range_list = variable_info[k]
                info[f'v{i+1}_name'] = f'v{i}'
                info[f'v{i+1}_low'] = range_list[0]
                info[f'v{i+1}_high'] = range_list[1]
    return info
