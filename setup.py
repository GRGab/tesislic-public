import setuptools

with open("DESCRIPTION.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='syntactic_causal_discovery',
    version='1.0.0',
    author='Gabriel Goren',
    description='A package for causal discovery using compression as an estimate for algorithmic mutual information',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gabrielgoren/tesislic-public',
    license='GNUv3',
    packages=['syntactic_causal_discovery'],
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'networkx',
                      'graphviz',
                      'pydot',
                      'dill',
                      'rpy2'],
)