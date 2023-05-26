# Author: Jamil Zaghir

""" Use of medical idiom """


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import os
import enchant as enc
from wordcloud import WordCloud
import re
import string


dico_fr = []


def clean_accents(input):
    accents = "àâéèêëïîôùûç"
    non_accents = "aaeeeeiiouuc"
    for acc, non_acc in zip(accents, non_accents):
        input = re.sub(acc, non_acc, input)
    input = re.sub("œ", "oe", input)
    return input


def remove_punc(input):
    input = re.sub("'", " ", input)
    input = re.sub("-", " ", input)
    input = re.sub("_", " ", input)
    input = re.sub("\(", " ", input)
    input = re.sub("\)", " ", input)
    input = re.sub(":", " ", input)
    input = input.translate(str.maketrans('', '', string.punctuation))
    input = re.sub(" +", " ", input)
    return input


def is_it_French_word(input):
    # Inclusion cases that might be excluded by pyenchant
    if input.lower() == 'avc' or input.lower() == 'ecg' or input.lower() == 'hta':
        return input
    
    if checkword.check(input) == True:
        return None
    elif checkword.check(input.capitalize()) == True:
        return None
    elif checkword.check(input.upper()) == True:
        return None
    else:
        # starts with in dictionary, accent-less and capitalization-less
        input = clean_accents(input)
        input = input.lower()
        # take tokens
        for w in dico_fr:
            if w.startswith(input):
                break
        else:
            return input
        return None


def is_it_French_label(input):
    input = remove_punc(input)
    tokens = nltk.word_tokenize(input)
    bad_tokens = []
    for token in tokens:
        r = is_it_French_word(token)
        if r is not None:
            bad_tokens.append(r)
    return bad_tokens


if __name__ == "__main__":
    # parse argument (data path)
    parser = argparse.ArgumentParser()
    parser.add_argument('-enchant_dico_path', type=str, default=os.path.join(".", "data", "dico.txt"))
    parser.add_argument('-hospitals_dico_path', type=str, default=os.path.join(".", "data", "no_fr_old.txt"))

    ARGS = parser.parse_args()
    # read data
    df = pd.read_pickle(os.path.join(".", "data", "pkl", "my_distances_empirical.pkl"))
    print(df.head())

    # find non french prefixes and words
    # instantiate the pyenchant dictionary
    checkword = enc.Dict('fr')

    with open(ARGS.enchant_dico_path, 'r', encoding='utf-8') as fread:
        for l in fread.readlines():
            l = l.replace('\n', '')
            l = remove_punc(l)
            l = clean_accents(l)
            l = l.lower()
            dico_fr.append(l)

    # takes about 5 minutes
    df["notFrench"] = df["PREFIX"].apply(is_it_French_label)
    print(df.head())

    df["len"] = df['notFrench'].apply(len)
    df_test = df[df['len'] >= 1]
    dico_values = dict()
    for i, row in df_test.iterrows():
        if row['len'] == 0:
            continue
        for el in row['notFrench']:
            if el not in dico_values.keys():
                dico_values[el] = 1
            else:
                dico_values[el] += 1

    sociolect_words = dict()
    # Abbreviations / non-morphologically french words
    dicokeys = dico_values.keys()
    with open(ARGS.hospitals_dico_path, 'r', encoding='utf-8') as fread:
        for l in fread.readlines():
            l = l.replace('\n', '')
            sociolect_words[l] = 0
            for k in dicokeys:
                if k.lower() == l:
                    sociolect_words[l] += dico_values[k]
            # sociolect_words[l] = dico_values[l]
    print(len(sociolect_words))

    # remove words which are not used by at least two users => avoid typos
    keys_to_remove = []
    df['notFrenchs'] = df['notFrench'].apply(lambda x: ' '.join(map(str,x)))
    for w, c in sociolect_words.items():
        if c <= 1:
            keys_to_remove.append(w)
        else:
            users=df[df['notFrenchs'].str.contains(r'^'+w, regex=True)]['USER_LOGIN'].nunique()
            if users < 2:
                keys_to_remove.append(w)
    for k in keys_to_remove:
        del sociolect_words[k]
    print(len(sociolect_words))

    wordcloud = WordCloud(background_color='white', width = 1000, height = 500).generate_from_frequencies(sociolect_words)

    plt.figure(figsize=(15,8))
    plt.axis("off")
    plt.imshow(wordcloud)

    # new column: isJargon True if any word from jargon is in notFrench, else False
    sociolect = set(sociolect_words.keys())
    df['isJargon'] = False
    for index, row in df.iterrows():
        res = bool(set(row['notFrench']) & sociolect)
        df.at[index, 'isJargon'] = res

    # DATA VISUALIZATION: stacked percentage bar
    df['USERLABEL'] = df['SELECTED_ID'] + df['USER_LOGIN']
    df_test = df[df["isJargon"] == True]
    df_no = df[df["isJargon"] == False]
    listusrlb = df_test['USERLABEL'].unique().tolist()
    df_t = df[df['USERLABEL'].isin(listusrlb)]
    df_t= df_t[df_t['SENIORITY_ITEM']<=15]
    df_t['isNotJargon'] = (df_t['isJargon'] == False)

    cross_tab_prop = pd.crosstab(index=df_t['SENIORITY_ITEM'],
                                columns=df_t['isNotJargon'],
                                normalize="index")
    cross_tab_prop = cross_tab_prop.rename(columns={False:'medical idiom', True:'others'})
    cross_tab_prop['medical idiom'] = cross_tab_prop['medical idiom'] * 100
    cross_tab_prop['others'] = cross_tab_prop['others'] * 100

    fig = plt.figure()
    ax = cross_tab_prop.plot(kind='bar', 
                            stacked=True, 
                            colormap='Set1', 
                            figsize=(10, 6))

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()

    df_test= df_test[df_test['SENIORITY_ITEM']<=15]
    counter = df_test.groupby('SENIORITY_ITEM').agg({'SENIORITY_ITEM': ['count']})
    counts = counter['SENIORITY_ITEM']['count'].tolist()
    # make a plot with different y-axis using second axis object
    ax2.plot([x-1 for x in df_t['SENIORITY_ITEM'].unique().tolist()], counts,color="blue",marker="o")
    ax2.set_ylabel("Number of autocomplete usages using the medical idiom",color="blue",fontsize=14)
    ax2.set_yscale("log")

    ax.legend(loc="upper right", ncol=2)
    ax.set_xlabel("Seniority Level",fontsize=10)
    ax.set_ylabel("Proportion",fontsize=14)
    plt.title("Use of medical idiom vs non-medical idiom with respect to seniority")

    plt.show()
    # plt.savefig('fig.png',
    #             format='png',
    #             dpi=300)

    breakpoint()
    # top 30 most used medical idiom
    jargons = pd.DataFrame.from_dict(sociolect_words, orient='index', columns=['count'])
    print(jargons.sort_values(by=['count'], ascending=False).head(30))
    print(len(jargons))

    # the case of HTA: sen 1-5, 6-10, 11-15
    print(df[(df['SELECTED_LABEL'] == "hypertension artérielle")&(df['SENIORITY_ITEM']  >= 1)&(df['SENIORITY_ITEM']  < 6)]['PREFIX'].value_counts())
    print(df[(df['SELECTED_LABEL'] == "hypertension artérielle")&(df['SENIORITY_ITEM']  >= 6)&(df['SENIORITY_ITEM']  < 11)]['PREFIX'].value_counts())
    print(df[(df['SELECTED_LABEL'] == "hypertension artérielle")&(df['SENIORITY_ITEM']  >= 11)&(df['SENIORITY_ITEM']  < 16)]['PREFIX'].value_counts())
