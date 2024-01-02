import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families Gaussian # or any other family you need
from statsmodels.genmod.families Binomial # or any other family you need


def svymean(data, var_name, weight_name):
    weighted_data = data[var_name] * data[weight_name]
    return weighted_data.sum() / data[weight_name].sum()

def svytotal(data, var_name, weight_name):
    return (data[var_name] * data[weight_name]).sum()

def svyvar(data, var_name, weight_name):
    mean = svymean(data, var_name, weight_name)
    variance = ((data[var_name] - mean)**2 * data[weight_name]).sum() / data[weight_name].sum()
    return variance

def svyglm(formula, data, family, weight_name):
    if weight_name not in data.columns:
        raise ValueError(f"Weight column '{weight_name}' not found in data.")

    weights = data[weight_name]
    if not np.issubdtype(weights.dtype, np.number):
        raise ValueError("Weights must be numeric.")

    glm_model = GLM.from_formula(formula, data, family=family, freq_weights=weights)
    result = glm_model.fit()
    return result


from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial  # or any other family as needed

# Assuming 'data' is your DataFrame and 'weights' is the column with frequency weights
# formula = 'response_variable ~ predictor_variable1 + predictor_variable2'
# weights = data['weights']  # replace 'weights' with the actual column name for frequency weights

# Creating the model
# glm_model = GLM.from_formula(formula, data, family=Binomial(), freq_weights=weights)

# Fitting the model
# result = glm_model.fit()

# Outputting the results
# print(result.summary())
