# Language adaptation and Zipf's Least Effort Principle

This repository is the implementation code for the paper - Language adaptation large scale observation study: accuracy is the main driver.


## Computing the Pareto front
compute_pareto.py: 

This code needs the (x,y) coordinates and a label for each coordinate.

Returns the Pareto optimal points as a csv file - there is also the option of returning the Pareto plots (Fig 6, and the Pareto plots in the Supplementary Information).

## Data preparation: from the dataset to pickle 
data_preparation.py: 

This code takes the dataset with datetime, the old dataset without datetime (to exclude users who where included in that dataset for fair estimation of seniorities), and the optimum computed in the previous script.

Returns the pickled data which includes the efficiency (query length), the accuracy (label rank) and the user-seniorities.

## Plot the evolution of queries effectiveness and efficiency with respect to the seniorities
evolution_wrt_pareto.py:

This code reads the pickle from the output of the previous script.

Returns the Fig 2 and 3 of the article.

## Medical idiom plot
medical_idiom.py:

This code also takes the pickle data, and path to french dictionaries (usually the ones from pyenchant which is based on several spellcheckers) as inputs.

Returns the Fig 4 and 5 of the article.

## Zipf plots
zipfdistribution.py:

This code takes the pickle data as input.

Returns the plot the frequency distribution on a normal or log-log scale (see Supplementary Information).

