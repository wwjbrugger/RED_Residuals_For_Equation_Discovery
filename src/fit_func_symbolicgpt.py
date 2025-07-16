import traceback
from functools import partial
import numpy as np
import torch
from scipy.optimize import minimize
from sympy import sympify
from definitions import ROOT_DIR
from src.configs.config_PySR import ConfigPySR
from src.configs.config_SymbolicGPT import ConfigSymbolicGPT
from src.configs.config_general import ConfigGeneral
from src.configs.config_syntax_tree import ConfigSyntaxTree
from src.evaluate_equation import evaluate_equation, map_equation_to_syntax_tree
from src.experiment_schedule import run_experiments
from src.preprocess import get_datasets_files
from symbolicgpt_master.models import GPT, GPTConfig, PointNetConfig
from symbolicgpt_master.utils import sample_from_model, CharDataset, lossFunc


def load_model(args, hyperparameter_set = 0):
    hyperparameter = {
        0: {'steps':200},  # This parameter is a tradeoff between accuracy and fitting time
        1: {'steps':100},
        2: {'steps':300}
    }
    # where to save model
    ckptPath = f'{ROOT_DIR}/src/symbolicgpt_master/models/XYE_9Var_20-250Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt'
    device = 'gpu'
    scratch = True  # if you want to ignore the cache and start for scratch
    numEpochs = 40  # number of epochs to train the GPT+PT model
    embeddingSize = 512  # the hidden dimension of the representation of both GPT and PT
    numPoints = [20, 250]  # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
    numVars = 9  # the dimenstion of input points x, if you don't know then use the maximum
    numYs = 1  # the dimension of output points y = f(x), if you don't know then use the maximum
    blockSize = 200  # spatial extent of the model for its context
    testBlockSize = 400
    batchSize = 128  # batch size of training data
    target = 'Skeleton'  # 'Skeleton' #'EQ'
    method = 'EMB_SUM'
    variableEmbedding = 'NOT_VAR'  # NOT_VAR/LEA_EMB/STR_VAR
    addVars = True if variableEmbedding == 'STR_VAR' else False
    chars = sorted(
        ['\n', ' ', '"', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ':', '<', '>', 'C', 'E', 'Q', 'S', 'T', 'X', 'Y', '[', ']', '_', 'a', 'b', 'c',
         'e', 'g', 'h', 'i', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's', 't', 'x', '{', '}'])  # extract unique characters from the text before converting the text to a list, # T is for the test data

    train_dataset = CharDataset([None], blockSize, chars, numVars=numVars,
                                numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                                const_range=None, xRange=None, decimals=None, augment=False)
    pconf = PointNetConfig(embeddingSize=embeddingSize,
                           numberofPoints=numPoints[1] - 1,
                           numberofVars=numVars,
                           numberofYs=numYs,
                           method=method,
                           variableEmbedding=variableEmbedding)
    mconf = GPTConfig(len(chars), blockSize,
                      n_layer=8, n_head=8, n_embd=embeddingSize,
                      padding_idx=train_dataset.paddingID)
    model = GPT(mconf, pconf)
    model.load_state_dict(torch.load(ckptPath))
    model = model.eval().to('cuda:0')
    model.hyperparameter = hyperparameter[hyperparameter_set]
    fitfunc = partial(call_symbolic_gpt,
                      model=model,
                      train_dataset=train_dataset,
                      args=args
                      )
    return fitfunc


def call_symbolic_gpt(model, train_dataset, X_df, Y_df, args, info=None):
    X = X_df.to_numpy()
    Y = Y_df.to_numpy().squeeze()
    points = np.zeros((X.shape[0], 10))
    points[:, 0:X.shape[1]] = X
    points[:, -1] = Y
    points = torch.Tensor(np.array([points.T])).to(device='cuda:0')
    np.seterr(all='raise')
    try:
        outputsHat = sample_from_model(
            model=model,
            x=torch.tensor([[23]]).to(device='cuda:0'),
            points=points,
            variables=torch.tensor([X.shape[1]]).to(device='cuda:0'),
            temperature=1.0,
            sample=True,
            top_k=0.0,
            top_p=0.7,
            **model.hyperparameter
        )[0]
        # filter out predicted
        predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])
        variableEmbedding = 'NOT_VAR'
        if variableEmbedding == 'STR_VAR':
            predicted = predicted.split(':')[-1]

        predicted = predicted.strip(train_dataset.paddingToken).split('>')
        predicted = predicted[0]  # if len(predicted[0])>=1 else predicted[1]
        predicted = predicted.strip('<').strip(">")
    except:
        print(f"Generating an equation failed")
        return {}
    try:
        c = [1.0 for i, x in enumerate(predicted) if x == 'C']  # initialize coefficients as 1
        # c[-1] = 0 # initialize the constant as zero
        b = [(-5, 5) for i, x in enumerate(predicted) if x == 'C']  # bounds on variables
        if len(c) != 0:
            # This is the bottleneck in our algorithm
            # for easier comparison, we are using minimize package
            cHat = minimize(lossFunc, c, bounds=b,
                            args=(predicted, X, Y))

            predicted_with_constant = predicted.replace('C', '{}').format(*cHat.x)
        else:
            predicted_with_constant = predicted
    except Exception as e:
        print(f'Error in Constant Fitting : Wrong Equation {predicted}, Err: {e}')
        return {}
    try:
        predicted_with_constant = str(sympify(predicted_with_constant))
    except Exception as e:
        print(f'Error in Sympify calculation  {predicted_with_constant}, Err: {e}')
        return {}
    for i in range(10):
        predicted_with_constant = predicted_with_constant.replace(f'x_{i + 1}', f' x_{i} ')
        predicted_with_constant = predicted_with_constant.replace(f'x{i + 1}', f' x_{i} ')
    try:
        tree = map_equation_to_syntax_tree(args, predicted_with_constant, infix=True)
    except (SyntaxError, RuntimeError) as E:
        print(traceback.format_exc())
        return {}
    output = evaluate_equation(args, tree, X_df, Y_df)
    return output


def run(args):
    print("Load Model")
    load_model_func = load_model
    run_experiments(args, load_model_func, 'symbolicgpt')


if __name__ == '__main__':
    print("Start")
    parser = ConfigSyntaxTree.arguments_parser()
    parser = ConfigSymbolicGPT.arguments_parser(parser)
    parser = ConfigGeneral.arguments_parser(parser)
    parser = ConfigPySR.arguments_parser(parser)
    args = parser.parse_args()

    run(args)
