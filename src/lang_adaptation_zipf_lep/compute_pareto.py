# Author: Jamil Zaghir

""" Compute optimum """

import numpy as np
import oapackage
import ast
import os
import pandas as pd


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

def main(input_data: str = os.path.join('.', 'data', 'out_prefixes_ranks.csv'), output_data: str = os.path.join('.', 'data', 'optimum_theorique.csv')):
    df = pd.read_csv(input_data, sep='\t', encoding="latin-1")
    optidict = dict()
    count = 0
    for i, row in df.iterrows():
        # pour chaque terme dans excel
        label = row['LABEL']
        hugid = row['HUGID']
        fulllabel = str(hugid) + '_' + label
        prefixes = ast.literal_eval(row['PREFIXES'])
        values_j = ast.literal_eval(row['RANKS'])
        values_k = []
        for p in prefixes:
            values_k.append(len(p))
        pareto_points, pareto_labels = compute_pareto_front(
            values_k, values_j, prefixes, fulllabel, True)
        optidict[fulllabel] = (hugid, label, pareto_points, pareto_labels)
        print(count)
        count += 1
    df = pd.DataFrame(
        columns=["HUGID", "LABEL", "PARETO_OPTI", "PARETO_OPTI_COORD"])
    for _, (hugid, label, paretop, paretol) in optidict.items():
        df = df.append({'HUGID': hugid, 'LABEL': label, "PARETO_OPTI": paretol,
                       "PARETO_OPTI_COORD": paretop}, ignore_index=True)
    df.to_csv(output_data, sep='\t', encoding='latin-1', index=False)

main()
