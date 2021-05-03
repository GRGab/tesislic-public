from independence_tests.cmitest import CompressionTest
from graphs import graph_1, graph_2, graph_3, graph_4, graph_5
from ic_runs import ICMultiRun
from independence_tests.kscontingencytest import KSContingencyTest
from parametric_models import halvingconcatfunc, bernoullifunc, \
                              modularsumfunc_factory, xorfunc
from utils import timer
from functional_models import ParametricModel, ParametricModelUniformProbs, \
                              ParametricModelRandomProbs, ParametricModelNonUniformProbs

# Choice of model
model_params = {
    'class' : ParametricModelNonUniformProbs,
    'kwargs': {
        'graph' : graph_1,
        'structure_function' : modularsumfunc_factory(3),
        'alphabet' : ['0', '1', '2'],
        'probs' : [0.4, 0.35, 0.25],
        'shift' : 0,
        'direction' : "Forward",
        'len_strings' : 20000,
        'noise_level' : 0.1,
    }
}

# Choice of Independence Tests
independence_test_specifications = [
    # {
    #     'class'  : KSContingencyTest,
    #     'kwargs' : {
    #         'alpha'        : 0.05,
    #         'kind'         : 'chi2',
    #         'n_subsamples' : 1,
    #         'alphabet'     : model_params['kwargs']['alphabet'],
    #     }
    # },
    {
        'class'  : CompressionTest,
        'kwargs' : {
            'criterion' : 'fixed_threshold',
            'compressor': 'I',
            'pairing'   : 'interleave',
            'threshold' : t
        }
    }
# ]
for t in [380, 390, 394, 400]]

# Choice of MultiRun options
multirun_params = {
    'n_repetitions' : 1,
    'visualize' : True,
    'verbose' : True,
    'save_everything' : True,
    'save_test_results' : True,
    'send_telegram' : False,
    'save_to_disk' : True
}

# Multirun
independence_test = [t['class'](**t['kwargs']) for t in independence_test_specifications]
model = model_params['class'](**model_params['kwargs'])
mr = ICMultiRun(model, independence_test, **multirun_params)
timer()(mr.run)()
