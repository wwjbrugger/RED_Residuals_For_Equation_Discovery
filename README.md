# Prompting Neural-Guided Equation Discovery Based on Residuals
This code accompanies the paper "Prompting Neural-Guided Equation Discovery Based on Residuals" accepted at Discovery Science 2025.

Residuals for Equation Discovery (RED) is a post-processing method for Equation Discovery that
improves a given equation in a targeted manner, based on its residuals

We test the method on five equation discovery systems (EDS) NeSymReS, SymbolicGPT, E2E, PySR and GPLearn. 

In total we implemented six methods to postprocess the initial equation suggested by the EDS  (RED, Permute, Hyper Parameter Grid, Constant Fitting, CVGP, Seeded GPlearn)

### Installation
All packages are specified for Python version 3.10
```
pip install -r requirements.txt
```

### Experiments
The experiments can be reproduced using the commands specified in experiments.sh. 

The scripts `src/fit_func_[model].py` start the experiments for the respective model. 
The postprocessing methods are located in `src/post_processing_methods`.

### Add new Equation Discovery Model 

We only need a method to load the pretrained model and a method which handels the call of the model.  

```
def load_model():
    model = ...
    fitfunc = partial(call,
                      model=model,
                      args=args
                      )
    return fitfunc


def call(model, X_df, Y_df, args):
    equation = model.fit(X_df.to_numpy(), Y_df.to_numpy())
    equation = str(sympify(equation))
    output = evaluate_equation(args, equation, X_df, Y_df)
    return output
```

### Add a new Operator
New Operator can be added in `src/equation_class/math_class`.  
Afterward, they have to be added in  `src/syntax_tree/syntax_tree.py` in the dictionary  `self.operator_to_class` and 
in  `src/equation_classes/infix_to_prefix.py` in the method `getPriority` 
