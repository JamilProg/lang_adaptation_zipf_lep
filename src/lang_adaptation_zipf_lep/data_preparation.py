# Author: Jamil Zaghir

""" Data Preparation to pickle """


import argparse
import os
import pandas as pd
import re
import datetime
import ast
from scipy.spatial import distance


def compute_pareto_distance(df, hugid, prefix, k, j):
    df = df[df["HUGID"] == int(hugid)]
    if len(df) == 1:
        pareto_pref = ast.literal_eval(df.iloc[0]["PARETO_OPTI"])
        if prefix in pareto_pref:
            return 0
        pareto_coord = ast.literal_eval(df.iloc[0]["PARETO_OPTI_COORD"])
        # distance max possible: 456
        min_dist = 456
        mypoint = (k,j)
        for coord in pareto_coord:
            lpoint = (coord[0],coord[1])
            dist = distance.euclidean(mypoint,lpoint)
            if dist < min_dist:
                min_dist = dist
    else:
        return None
    return min_dist


def rank_j(prefix, hugid, label, date, df):
    df = df[df["HUGID"] == int(hugid)]
    if len(df) == 1:
        prefixes = ast.literal_eval(df.iloc[0]["PREFIXES"])
        ranks = ast.literal_eval(df.iloc[0]["RANKS"])
        for p, rank in zip(prefixes, ranks):
            if p == prefix:
                return rank
    else:
        print(len(df),"ERROR", prefix, hugid, label)
        return -1
    return -1


def cleaner(input):
    # cleaner function
    input = input.lower()
    input = re.sub(r'<b>','', input) 
    input = re.sub(r'</b>','', input)
    input = re.sub(r'\x92','\'', input)
    input = re.sub(r'\xa0',' ', input)
    return input


def filter_lph(df):
    # keep data from the problem list terminology
    df = df[df["SELECTED_TERMINOLOGY"] == 'HUG']
    return df


if __name__ == "__main__":
    # parse argument (data path)
    parser = argparse.ArgumentParser()

    ## Private data: internal to the hospitals of Geneva
    # unstamped data
    parser.add_argument('-nostamp_data_path', type=str, default=os.path.join('.','data','Unstamped_completerRaw.csv'))
    # stamped data
    parser.add_argument('-datapath', type=str, default=os.path.join('.','data','completerRaw.csv'))
    # obsolete labels (labels that are no longer suggested by the autocomplete tool)
    parser.add_argument('-obsoletelabels', type=str, default=os.path.join('.','data','obsoleteLabels.csv')) 
    # data which contains the suggested labels for all queries and the corresponding ranks
    parser.add_argument('-ranks', type=str, default=os.path.join("./data", "ranks.csv"))

    ## Data which contains the pareto fronts computed in the previous script 01_compute_pareto
    parser.add_argument('-optimum', type=str, default=os.path.join("./data", "optimum_theorique.csv"))
    ARGS = parser.parse_args()
    
    # Load stamped data
    df = pd.read_csv(ARGS.data_path)
    df = df.dropna()
    df = filter_lph(df)
    # Load non stamped data
    dfns = pd.read_csv(ARGS.nostamp_data_path)
    dfns = dfns.dropna()
    dfns = filter_lph(dfns)
    print("Data loaded")

    # Toss off unused labels
    unusedIDs = []
    with open(ARGS.obsoletelabels, "r") as fr:
        for l in fr.readlines():
            l = l.replace('\n', '')
            unusedIDs.append(l)
    df = df[~df["SELECTED_ID"].isin(unusedIDs)]

    # Toss off users from non temporal data in our dataset to ensure that we keep track of actual beginners
    df = df[~df["USER_LOGIN"].isin(dfns["USER_LOGIN"])]
    print("Dataset created: with only users who began the use of autocomplete since we track the time stamp.")
    
    # Selected Labels: Normalization
    df["SELECTED_LABEL"] = df['SELECTED_LABEL'].str.replace(r'<b>', '')
    df["SELECTED_LABEL"] = df['SELECTED_LABEL'].str.replace(r'</b>', '')
    df["SELECTED_LABEL"] = df['SELECTED_LABEL'].apply(str.strip)
    df["SELECTED_LABEL"] = df['SELECTED_LABEL'].apply((lambda x: cleaner(x)))

    # Vital stats
    print("There are: " + str(len(df)) + " rows.")
    print("There are", df["SELECTED_LABEL"].nunique(), "labels in the terminology")
    print("There are", df["USER_LOGIN"].nunique(), "users")
    duration = datetime.datetime.strptime(df["DATETIME"].max(), '%Y-%m-%d') - datetime.datetime.strptime(df["DATETIME"].min(), '%Y-%m-%d')
    print("The duration of this dataset:", duration)

    # Compute K (query length): Efficiency
    df["K"] = df['PREFIX'].str.len()
    print(df.head())

    # Compute J (term rank): Effectiveness
    outcsv = pd.read_csv(ARGS.ranks, sep='\t', encoding="latin-1")
    df["J"] = df.apply(lambda x: rank_j(x.PREFIX, x.SELECTED_ID, x.SELECTED_LABEL, x.DATETIME, outcsv), axis=1)
    df = df[df["J"] != -1]
    print(df.head())

    # Compute User-Label Seniority: time axis
    # We compute the seniority of terms
    df_sen = df
    df_sen = df_sen.sort_values(by=['DATETIME','USER_LOGIN'])
    df_sen["un"] = 1
    df_sen['SENIORITY_ITEM'] = df_sen.groupby(['USER_LOGIN', 'SELECTED_LABEL'])['un'].cumsum()
    df_sen = df_sen.drop(columns='un')
    df_sen_sorted = df_sen.sort_values(by=['USER_LOGIN','SELECTED_LABEL', 'SENIORITY_ITEM'])
    print(df_sen_sorted.head(20))
    df = df_sen_sorted

    # Compute the distance to the Pareto front
    outcsv = pd.read_csv(ARGS.optimum, sep='\t', encoding="latin-1")
    df["dist_pareto"] = df.apply(lambda x: compute_pareto_distance(outcsv, x.SELECTED_ID, x.PREFIX, x.K, x.J), axis=1)
    print(df.head())

    # Save to pickle
    df.to_pickle(os.path.join(".", "data", "pkl", "my_distances_empirical.pkl"))
    