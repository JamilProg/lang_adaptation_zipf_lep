# Author: Jamil Zaghir

""" Evolution of queries effectiveness and efficiency """


import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import signal
import math


# https://en.wikipedia.org/wiki/Sample_size_determination
def sample_size(N, z, e):
    p = 0.5
    numerator = (z**2) * p * (1-p)
    denominator = e**2 + ((z**2) * p * (1-p)) / N
    n = numerator / denominator
    return math.ceil(n)


def filter_seniority_max(senmax, df):
    df_test = df.groupby(['SELECTED_ID']).agg({'SENIORITY_ITEM': ['max']})
    df_test = df_test[df_test["SENIORITY_ITEM"]["max"] <= senmax]
    ids_to_remove = df_test.index.tolist()
    df = df[~df["SELECTED_ID"].isin(ids_to_remove)]
    userlabel_amount = df['SENID'].nunique()
    return df, userlabel_amount


if __name__ == "__main__":
    # read data
    df = pd.read_pickle(os.path.join(".", "data", "pkl", "my_distances_empirical.pkl"))
    print(df.head())

    # define ID for user-label pairs
    df['SENID'] = df['SELECTED_ID'] + df['USER_LOGIN']
    populationsize = df['SENID'].nunique()
    print('Number of unique user-label pairs:', populationsize)
    print('Number of rows:',len(df))

    # to avoid terminology bias, we filter out labels that do not reach a given seniority value
    # meaning that the more the max seniority, the more we delete user-label pairs, but we can lose statistical relevance
    # to ensure the statistical robustness of the results, we make sure that there are enough of user-label pairs at senmax
    senmax_candidate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    # We want the sample to have at least the following settings: confidence interval = 1.96 (95%), margin of error = 0.03 (3%)
    minimal_sample_size = sample_size(populationsize, 1.96, 0.03)
    print('Minimal sample size for a population of size', populationsize, ':', minimal_sample_size)
    # We find the best senmax so that our sample size does not go below that estimate
    senmax = 1
    for sen in senmax_candidate:
        _, candidate_size = filter_seniority_max(sen, df)
        print('At seniority', sen, ', there are', candidate_size, 'user-label pairs.')
        if candidate_size >= minimal_sample_size:
            if sen > senmax:
                senmax = sen
        else:
            break
    print('Selected senmax:', senmax)

    # filter labels given the ones kept at seniority max for fair comparisons
    df, usrlabel_amount = filter_seniority_max(senmax, df)
    print('Number of unique user-label pairs:',usrlabel_amount)

    # prepare data for plotting
    df = df[df['SENIORITY_ITEM'] <= senmax]
    result = df.groupby('SENIORITY_ITEM').agg({'dist_pareto': ['mean', 'std']})
    dist_pareto = result['dist_pareto']['mean'].tolist()
    dist_pareto_std = result['dist_pareto']['std'].tolist()

    # Plot 1: distance from the Pareto front
    x = list(range(1,len(dist_pareto)+1))
    np.random.seed(1)

    fig = go.Figure(layout=go.Layout(
            title=go.layout.Title(text="Evolution of the distance from the Pareto front")
        ))
    fig.add_trace(go.Scatter(
        x=x,
        y=dist_pareto,
        mode='markers',
        marker=dict(size=2, color='black'),
        name='Mean Euclidean Distance'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=signal.savgol_filter(dist_pareto,
                            9, # window size used for filtering
                            3), # order of fitted polynomial
        mode='markers',
        marker=dict(
            size=6,
            color='mediumpurple',
            symbol='x'
        ),
        name='Mean Euclidean Distance (Savitzky-Golay Smoothing)'
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=dist_pareto_std,
        marker=dict(color='lightgray'),
        name='Standard Deviation'
    ))

    fig.update_layout(legend=dict(x=0.4, 
                                y=1,
                                font=dict(
                                    size=16,
                                    color="black"
                                    ),
                                groupclick="toggleitem", 
                                bordercolor="LightGray", 
                                borderwidth=2),
                            xaxis_title="User-Label Seniority",
                            template='plotly_white',
                            font=dict(
                                size=16
                            ))


    fig.show()
    # fig.write_image('./avg_dist_pareto.png', 'png', width=1000, height=600, scale=2)

    # Plot 2: Effectiveness and efficiency
    result = df.groupby('SENIORITY_ITEM').agg({'K': ['mean', 'std']})
    kmean = result['K']['mean'].tolist()
    kstd = result['K']['std'].tolist()

    result = df.groupby('SENIORITY_ITEM').agg({'J': ['mean', 'std']})
    jmean = result['J']['mean'].tolist()
    jstd = result['J']['std'].tolist()

    fig = go.Figure(layout=go.Layout(
            title=go.layout.Title(text="Separate evolution of the two objectives")
        ))

    fig.add_trace(go.Scatter(
        x=x,
        y=signal.savgol_filter(kmean,
                            53, # window size used for filtering
                            3), # order of fitted polynomial
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            symbol='x'
        ),
        name='Avg query length K (Savitzky-Golay Smoothing)'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=signal.savgol_filter(jmean,
                            53, # window size used for filtering
                            3), # order of fitted polynomial
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            symbol='x'
        ),
        name='Avg rank J (Savitzky-Golay Smoothing)'
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=kstd,
        marker=dict(color='pink'),
        name='Query length K Standard deviation'
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=jstd,
        marker=dict(color='lightblue'),
        name='Rank J Standard deviation'
    ))

    fig.update_layout(legend=dict(x=0.45, 
                                y=1,
                                font=dict(
                                    size=16,
                                    color="black"
                                    ),
                                groupclick="toggleitem", 
                                bordercolor="LightGray", 
                                borderwidth=2),
                            xaxis_title="User-Label Seniority",
                            template='plotly_white',
                            font=dict(
                                size=16
                            ))


    fig.show()
    # fig.write_image('./monitor_two_variables.png', 'png', width=1000, height=600, scale=2)
