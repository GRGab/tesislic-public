import dill
import itertools
import networkx as nx
from joblib import Parallel, delayed
from pathlib import Path

from analyze_runs import load_multirun
from analysis.resultados_scripts.models import model_factory
from cmitest import CompressionTest
from ic_runs import ICMultiRun
from kscontingencytest import KSContingencyTest
from utils import timer, timestamp
from visualization import visualize_graph, visualize_multirun

def multirun_factory(model, tests, n_repetitions, savepath, visualize=False):
    test_specs = []
    for test in tests:
        if test == 'statistical':
            test_spec = {
                'class'  : KSContingencyTest,
                'kwargs' : {
                    'alpha'        : 0.05,
                    'kind'         : 'chi2',
                    'n_subsamples' : 1,
                    'alphabet'     : model.alphabet,
                }
            }
        else: # must be of form (information_metric, pairing_func, threshold)
            test_spec = {
                'class' : CompressionTest,
                'kwargs' : {
                    'criterion' : 'fixed_threshold',
                    'compressor': test[0],
                    'pairing'   : test[1],
                    'threshold' : test[2]
                }
            }
        test_specs.append(test_spec)
    multirun_specs = {
        'n_repetitions' : n_repetitions,
        'visualize' : visualize,
        'verbose' : False,
        'save_everything' : True,
        'save_test_results' : True,
        'send_telegram' : False,
        'save_to_disk' : True,
        'savepath' : savepath
    }
    mr = multirun_from_dicts(model, test_specs, multirun_specs)
    return mr

def multirun_from_dicts(model, test_dicts, multirun_dict):    
    independence_test = [t['class'](**t['kwargs']) for t in test_dicts]
    mr = ICMultiRun(model, independence_test, **multirun_dict)
    return mr


### Main functions for executing a lot of multiruns
@timer(telegram=True)
def run_test_on_some_models(test, model_tuples, n_repetitions=1, n_jobs=4):
    """
    model_tuples is list with elements of form (graph, func, bool)
    test is "statistical" or is tuple of form (metric, pairing_func, threshold) 
    """
    execution_instructions = (delayed(_run_test_on_one_model)(test,
                                                              model_tuple,
                                                              n_repetitions)
                              for model_tuple in model_tuples)
    Parallel(n_jobs=n_jobs)(execution_instructions)

def _run_test_on_one_model(test, model_tuple, n_repetitions):
    graph_number, func, shift = model_tuple
    print(f'graph_{graph_number} / {func} / shift={shift}')
    path = make_path((graph_number, func, shift), test, n_repetitions)
    mr = multirun_factory(model_factory(graph_number, func, shift),
                            [test],
                            n_repetitions,
                            path)
    mr.run()
        
def run_multiple_tests_on_model(tests, model_tuple, n_repetitions=1):
    # call multirun_factory with savepath = None. This makes sure that
    # visualize_multirun is not called from within ICMultiRun instance
    # (this should be refactored thouroughly!)
    graph_number, func, shift = model_tuple
    multirun = multirun_factory(model_factory(graph_number, func, shift),
                                              tests,
                                              n_repetitions,
                                              None)
    multirun.run()
    directory = make_path(model_tuple)
    # Save pkl
    with open(f"{directory}/multitests_{timestamp()}.pkl", 'wb') as f:
        dill.dump(multirun, f)
    # Now, save pngs
    for i, test in enumerate(tests):
        if test == 'statistical':
            path = directory + f'/statistical-rep{n_repetitions}'
        else:
            path = directory + f'/{test[0]}-{test[1]}-thres{test[2]}-rep{n_repetitions}'
        # following lines are literal copy from visualize_multirun (refactor needed)
        g = nx.Graph()
        g.add_nodes_from(multirun.model.nodes)
        g.add_edges_from(itertools.combinations(multirun.model.nodes, 2))
        g.remove_edges_from([e for e in g.edges if multirun.edge_counts[i][e]==0])
        edge_labels = {e: f'{multirun.edge_counts[i][e]/multirun.n_repetitions:.3g}' for e in g.edges}
        visualize_graph(g, edge_labels=edge_labels, title=None, savepath=path,
                only_save=True)


### Visualization
def visualize_results(test, model_tuples, n_repetitions=1): # se pisa un poco con visualize_multirun
    for i, func, shift in model_tuples:
        mr = load_result(test, (i, func, shift), n_repetitions)
        print(f"graph_{i} / {func} / {shift}")
        visualize_multirun(mr)

def load_result(test, model_tuple, n_repetitions): # se pisa un poco con load_multirun
    path = make_path(model_tuple, test, n_repetitions, relative_to='analysis')
    return load_multirun(f'{path}.pkl')
    
###############################################################################
# create paths for use in other functions
###############################################################################

def make_path(model_tuple, test=None, n_repetitions=None, relative_to='tesislic'):
    graph_number, func, shift = model_tuple
    shift_str = shift_data_to_string(shift)
    if relative_to == 'tesislic':
        directory = f'analysis/resultados/graph_{graph_number}/{func}{shift_str}'
    elif relative_to == 'analysis':
        directory = f'resultados/graph_{graph_number}/{func}{shift_str}'
    else:
        raise ValueError
    Path(directory).mkdir(parents=True, exist_ok=True)
    if test is not None and n_repetitions is not None:
        if test == 'statistical':
            path = directory + f'/statistical-rep{n_repetitions}'
        else:
            path = directory + f'/{test[0]}-{test[1]}-thres{test[2]}-rep{n_repetitions}'
    else:
        path = directory
    return path

def shift_data_to_string(shift_data):
    if isinstance(shift_data, bool):
        if shift_data:
            shift_str = '-shift'
        else:
            shift_str = ''
    else: # shift is an int
        shift_str = f'-shift{shift_data}'
    return shift_str