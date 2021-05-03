from typing import Callable, Union
from graphs import graph_1, graph_2, graph_3, graph_4, graph_5
from parametric_models import erraticfunc, halvingconcatfunc, bernoullifunc, \
                              xorfunc, sum3func, sum5func
from functional_models import ParametricModelNonUniformProbs

# In the following, all models use constant  vals of: noise, len
# and all root nodes have the same prob dist
def model_factory(graph_name:Union[str, int], func_name:str, shift:Union[bool, int]):
    graph_name_dict = {
        1 : graph_1,
        2 : graph_2,
        3 : graph_3,
        4 : graph_4,
        5 : graph_5,
        'Y' : graph_1,
        'instrum' : graph_2,
        'tridente': graph_3,
        'trid_inst': graph_4,
        'W':graph_5
    }
    structure_function_name_dict = {
        'halvingconcatfunc' : halvingconcatfunc,
        'bernoullifunc' : bernoullifunc,
        'xorfunc' : xorfunc,
        'sum3func' : sum3func,
        'sum5func' : sum5func,
        'erraticfunc' : erraticfunc,
    }
    if func_name == 'sum3func':
        probs = [0.4, 0.35, 0.25]
        alphabet = ['0', '1', '2']
    elif func_name == 'sum5func':
        probs = [0.3, 0.15, 0.1, 0.15, 0.3]
        alphabet = ['0', '1', '2', '3', '4']
    else:
        probs = 0.4 # modify def of random_string so that this isn't so ugly!
        alphabet = ['0', '1']
    ################################ adhoc change to shift parameter
    if isinstance(shift, bool):
        shift = {'X':1, 'Y':2} if shift else 0
    else: # shift is assumed to be an int that multiplies shift lengths
        shift = {'X':shift, 'Y':2*shift}
    ################################

    model_dict = {
        'class' : ParametricModelNonUniformProbs,
        'kwargs': {
            'graph' : graph_name_dict[graph_name],
            'structure_function' : structure_function_name_dict[func_name],
            'alphabet' : alphabet,
            'probs' : probs,
            'shift' : shift,
            'direction' : "Forward",
            'len_strings' : 20000,
            'noise_level' : 0.1,
        }
    }
    return model_from_dict(model_dict)

def model_from_dict(model_dict):
    model = model_dict['class'](**model_dict['kwargs'])
    return model