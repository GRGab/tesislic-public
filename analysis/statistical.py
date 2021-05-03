from pathlib import Path

from utils import timer
from analysis.models import model_factory
from analysis.multiruns import multirun_factory

@timer(telegram=True)
def run_statistical_test_all_models(n_repetitions=1):
    for graph_number in [1, 2, 3, 4, 5]:
        for func in ['xorfunc', 'sum3func', 'sum5func', 'bernoullifunc', 'halvingconcatfunc']:
            for shift in [False, True]:
                print(f'graph_{graph_number} / {func} / shift={shift}')
                directory = f'analysis/resultados/graph_{graph_number}/{func}{"-shift" if shift else ""}'
                Path(directory).mkdir(parents=True, exist_ok=True)
                path = directory + f'/statistical-rep{n_repetitions}'
                mr = multirun_factory(model_factory(graph_number, func, shift),
                                      ['statistical'],
                                      n_repetitions,
                                      path)
                mr.run()