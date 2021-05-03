import numpy as np
import rpy2
# Importing the top-level sub-package is also initializing and starting R embedded in the current Python process
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
# Import R packages
base = rpackages.importr('base')
utils = rpackages.importr('utils')
stats = rpackages.importr('stats')

def fisher_test(table: np.array, workspace: int = 200000, simulate_pval: bool = False,
                replicate: int = 10000,) -> float:
    nrows = table.shape[0]
    v = robjects.FloatVector(np.ravel(table.T)) # transpose required for consistency when translating back to numpy
    m = robjects.r['matrix'](v, nrow = nrows)
    f = robjects.r["fisher.test"]
    p = f(m, workspace=workspace, simulate_p_value=simulate_pval, B=replicate)[0][0]
    return p

if __name__ == "__main__":
    table = np.array([[12, 1, 45], [1, 15, 15], [12, 4, 31]])
    p = fisher_test(table)
    print(p)
    """
    Results of code above:
    simulate_p_value == False
    -> 2.098731844789896e-07
    simulate_p_value == True, B = 2000 (default)
    -> 0.0004997501249375312
    simulate_p_value == True, B = 10000
    -> 9.999000099990002e-05
    simulate_p_value == True, B = 100000
    -> 9.99990000099999e-06
    """