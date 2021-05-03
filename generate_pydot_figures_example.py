import dill

from generacion_imagenes import draw_multirun_pydot

filename = 'examples/example_multirun.pkl'

with open(filename, 'rb') as f:
    mr = dill.load(f)

for i in range(len(mr.independence_test)):
    draw_multirun_pydot(mr, f'examples/example_multirun-test{i}',
                        test_number=i,
                        fmt='pdf')