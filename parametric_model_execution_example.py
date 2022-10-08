from syntactic_causal_discovery.independence_tests.cmitest import CompressionTest
from syntactic_causal_discovery.graphs import graph_1, graph_2, graph_3, graph_4, graph_5
from syntactic_causal_discovery.ic_runs import ICMultiRun
from syntactic_causal_discovery.independence_tests.kscontingencytest import KSContingencyTest
from syntactic_causal_discovery.parametric_models import halvingconcatfunc, bernoullifunc, \
                              modularsumfunc_factory, xorfunc
from syntactic_causal_discovery.utils import timer
from syntactic_causal_discovery.functional_models import ParametricModel, ParametricModelUniformProbs, \
                              ParametricModelRandomProbs, ParametricModelNonUniformProbs

# Choice of model
model_params = {
    'class' : ParametricModelNonUniformProbs,
    'kwargs': {
        'graph' : graph_1,
        'structure_function' : halvingconcatfunc,
        'alphabet' : ['0', '1'],
        'probs' : 0.4,
        'shift' : 0,
        'direction' : "Forward",
        'len_strings' : 20000,
        'noise_level' : 0.1,
    }
}

# Choice of Independence Tests
independence_test_specifications = [
    {
        'class'  : KSContingencyTest,
        'kwargs' : {
            'alpha'        : 0.05,
            'kind'         : 'chi2',
            'n_subsamples' : 1,
            'alphabet'     : model_params['kwargs']['alphabet'],
        }
    },
    {
        'class'  : CompressionTest,
        'kwargs' : {
            'criterion' : 'fixed_threshold',
            'compressor': 'I',
            'pairing'   : 'concat',
            'threshold' : 50
        }
    }
]

# Choice of MultiRun options
multirun_params = {
    'n_repetitions' : 1,
    'visualize' : True,
    'verbose' : True,
    'save_everything' : True,
    'save_test_results' : True,
    'save_to_disk' : True,
    'savepath': 'examples/example_multirun'
}

# Multirun
independence_test = [t['class'](**t['kwargs']) for t in independence_test_specifications]
model = model_params['class'](**model_params['kwargs'])
mr = ICMultiRun(model, independence_test, **multirun_params)
timer()(mr.run)()
