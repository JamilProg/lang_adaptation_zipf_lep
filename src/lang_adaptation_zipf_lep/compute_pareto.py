# Author: Jamil Zaghir

""" Compute optimum """

import numpy as np
import oapackage


# Find the optimal points as coordinates and the correspondings text
def compute_pareto_front(x, y, texts):
    # OAPackage function works as a max-max optimization
    # As we wants to build Pareto in a min-min optimization, we put coordinates in negative
    k = [-i for i in x]
    j = [-i for i in y]
    datapoints = np.array([k, j], dtype=np.double)

    pareto = oapackage.ParetoDoubleLong()
    for ii in range(0, datapoints.shape[1]):
        w = oapackage.doubleVector((datapoints[0, ii], datapoints[1, ii]))
        pareto.addvalue(w, ii)

    pareto.show(verbose=1)

    lst = pareto.allindices()  # the indices of the Pareto optimal designs
    datapoints = np.array([[-i for i in l] for l in datapoints])
    optimal_datapoints1 = datapoints[:, lst]
    optimal_datapoints = [list(i) for i in zip(*optimal_datapoints1)]

    tuples = [(datapoint[0], datapoint[1]) for datapoint in optimal_datapoints]
    pareto_labels = []
    if texts:
        for i, txt in enumerate(texts):
            if (datapoints[0][i], datapoints[1][i]) in tuples:
                pareto_labels.append(txt)
    return optimal_datapoints, pareto_labels
