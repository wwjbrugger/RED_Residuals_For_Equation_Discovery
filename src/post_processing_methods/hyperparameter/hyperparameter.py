from src.evaluate_equation import test_equation
import time

def different_hyperparameter(args, load_model_func, X_df, Y_df,equation_info):
    output={}
    for i in range(1,3,1):
        start_time_hyperparameter = time.time()
        model =load_model_func(args, hyperparameter_set=i )
        output_fit = model(X_df=X_df, Y_df=Y_df, info=equation_info)
        if output_fit == {}:
            output['equation'] = 'Error no equation returned'
            continue
        output[i] = output_fit
        output[i]['time'] = time.time() - start_time_hyperparameter
    return output

