# Author: Jamil Zaghir

""" Compute rank frequency distribution of the dataset """


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import enchant as enc
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as cm
import matplotlib as mpl
import os


# instantiate the pyenchant dictionary
checkword = enc.Dict('fr')


############################################################# Inspired by https://varunver.wordpress.com/2019/08/27/data-structure-trie-radix-tree/ #############################################################
class Node:
    def __init__(self):
        self.children = {}
        self.endOfWord = False


def insertWord(root, word):
    '''
    Loop through characters in a word and keep adding them at a new node, linking them together
    If char already in node, pass
    Increment the current to the child with the character
    After the characters in word are over, mark current as EOW
    '''
    current = root
    for char in word:
        if char in current.children.keys():
            pass
        else:
            current.children[char] = Node()
        current = current.children[char]
    current.endOfWord = True


def allWords(prefix, node, results):
    '''
    Recursively call the loop
    Prefix will be prefix + current character
    Node will be position of char's child
    results are passed by reference to keep storing result

    Eventually, when we reach EOW, the prefix will have all the chars from starting and will be the word that we need. We add this word to the result
    '''
    if node.endOfWord:
        results.append(prefix)
    for char in node.children.keys():
        #print char, node, node.children
        allWords(prefix + char, node.children[char], results)
  

def searchWord(root, word):
    '''
    Loop through chars of the word in the trie
    If char in word is not in trie.children(), return
    If char found, keep iterating
    After iteration for word is done, we should be at the end of word. If not, then word doesn't exist and we return false.
    '''
    current = root
    search_result = True
    for char in word:
        if char in current.children.keys():
            pass
        else:
            search_result = False
            break
        current = current.children[char]
    if not current.endOfWord:
        search_result = False
    return search_result


def getWordsWithPrefix(prefix, node, prefix_result):
    '''
    We loop through charcters in the prefix along with trie
    If mismatch, return
    If no mismatch during iteration, we have reached the end of prefix. Now we need to get words from current to end with the previx that we passed. So call allWords with prefix
    '''
    current = node
    for char in prefix:
        if char in current.children.keys():
            pass
        else:
            return
        current = current.children[char]
    allWords(prefix, current, prefix_result)
##################################################################################################################################################################################################################


def plotTrie(root, withlabels=True, wordvaluepairs=None, initial=''):
    # plt.figure(figsize=(250,10))
    plt.figure(figsize=(120,10))

    G = nx.DiGraph()

    def traverse(node, prefix):
        if node.endOfWord:
            G.add_node(prefix, end=True)
        else:
            G.add_node(prefix)

        for char, child in node.children.items():
            G.add_edge(prefix, prefix + char)
            traverse(child, prefix + char)

    traverse(root, " ")

    # write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    write_dot(G,'test.dot')

    # same layout using matplotlib with no labels
    plt.title('Trie ' + initial)
    pos = graphviz_layout(G, prog='dot')
    if wordvaluepairs == None:
        nx.draw(G, pos, with_labels=withlabels, arrows=True, font_size=8, node_color='#54a9e3', node_size=200)
    else:
        # Normalization
        maxvalue = 1000
        for k in wordvaluepairs.keys():
            if wordvaluepairs[k] >= maxvalue:
                wordvaluepairs[k] = 100
            else:
                wordvaluepairs[k] = wordvaluepairs[k] / (maxvalue/100)
        nodesdisplayed = [' ' + w for w in wordvaluepairs.keys()]
        nodesdisplayed = [n for n in nodesdisplayed if n in G.nodes()]
        colors = [cm.Wistia(wordvaluepairs[word.strip()]) if word.strip() in wordvaluepairs.keys() else 'white' for word in nodesdisplayed]
        # nx.draw(G, pos, with_labels=withlabels, arrows=False, font_size=8, node_color=colors, alpha=1, node_size=200, nodelist=nodesdisplayed)
        # create the scalar mappable
        cmap = cm.get_cmap('Wistia')
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=1000))

        nx.draw(G, pos, with_labels=withlabels, arrows=False, font_size=8, node_color=colors, alpha=0.8, node_size=100, nodelist=nodesdisplayed)

        # add colorbar
        cbar = plt.colorbar(sm)
        cbar.set_label('Word frequency')
        cbar.set_ticks([1, 200, 400, 600, 800, 1000])
        cbar.set_ticklabels(['1', '200', '400', '600', '800', '1000+'])
    plt.savefig(os.path.join('.', 'tries', initial+'Graph'+'.png'))
    # plt.show()
    plt.close()


def is_it_French_word(input):
    if checkword.check(input) == True:
        return True
    elif checkword.check(input.capitalize()) == True:
        return True
    else:
        return False


if __name__ == "__main__":
    df = pd.read_pickle("./data/pkl/my_distances_empirical.pkl")

    # dict token -> boolPareto
    paretodict = dict()
    prefixes_all = df['PREFIX'].unique().tolist()
    dft = df[df['dist_pareto'] == 0]
    prefixes_pareto = dft['PREFIX'].unique().tolist()
    for p in prefixes_all:
        p = p.strip()
        lp = p.split()
        for token in lp:
            paretodict[token] = False
    for p in prefixes_pareto:
        p = p.strip()
        lp = p.split()
        for token in lp:
            paretodict[token] = True

    # dict token -> numberUse
    nUsesdict = dict()
    prefcounts = df['PREFIX'].value_counts()
    dico = dict(zip(prefcounts.index.tolist(), prefcounts.values.tolist()))
    for k, v in dico.items():
        k = k.strip()
        lk = k.split()
        for tk in lk:
            if tk not in nUsesdict.keys():
                nUsesdict[tk] = v
            else:
                nUsesdict[tk] += v

    # add lower
    dicozipf = dict()
    for k, v in nUsesdict.items():
        keyy = k.lower()
        if keyy in dicozipf.keys():
            dicozipf[keyy] += v
        else:
            dicozipf[keyy] = v
    dftest_temp = pd.DataFrame.from_dict(dicozipf, orient='index', columns=['countUsages'])

    ## WORD LEVEL
    # sort the dictionary by values in descending order
    sorted_freq = sorted(dicozipf.items(), key=lambda x: x[1], reverse=True)

    # extract the word and frequency values as separate lists
    words = [w[0] for w in sorted_freq]
    freqs = [w[1] for w in sorted_freq]

    # plot the frequency distribution on a log-log scale
    fig, ax = plt.subplots(figsize=(8, 6))
    ranks = range(1,len(freqs)+1)
    ax.scatter(ranks, freqs, color='red', s=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Word Frequency Rank')
    ax.set_ylabel('Word Frequency Value')
    ax.set_title('Plot of frequency versus rank of words used by autocomplete users')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    # plt.show()
    plt.close()

    # Show top 50 words
    print(dftest_temp.sort_values(by=['countUsages'], ascending=False).head(50))

    ## QUERY LEVEL
    # sort the dictionary by values in descending order
    prefixes_index, prefixes_values = df['PREFIX'].value_counts().index.tolist(), df['PREFIX'].value_counts().values.tolist()

    dicozipf = dict()
    for k, v in zip(prefixes_index, prefixes_values):
        keyy = k.lower()
        if keyy in dicozipf.keys():
            dicozipf[keyy] += v
        else:
            dicozipf[keyy] = v
    dftest_temp = pd.DataFrame.from_dict(dicozipf, orient='index', columns=['countUsages'])

    # extract the word and frequency values as separate lists
    sorted_freq = sorted(dicozipf.items(), key=lambda x: x[1], reverse=True)
    prefs = [w[0] for w in sorted_freq]
    freqs = [w[1] for w in sorted_freq]

    # plot the frequency distribution on a log-log scale
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(range(len(freqs)), freqs, color='red', s=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Query Frequency Rank')
    ax.set_ylabel('Query Frequency Value')
    ax.set_title('Plot of frequency versus rank of queries used by autocomplete users')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    # plt.show()
    plt.close()

    # Show top 50 queries
    print(dftest_temp.sort_values(by=['countUsages'], ascending=False).head(50))

    ## SELECTED LABEL LEVEL
    # sort the dictionary by values in descending order
    prefixes_index, prefixes_values = df['SELECTED_LABEL'].value_counts().index.tolist(), df['SELECTED_LABEL'].value_counts().values.tolist()

    dicozipf = dict()
    for k, v in zip(prefixes_index, prefixes_values):
        keyy = k.lower()
        if keyy in dicozipf.keys():
            dicozipf[keyy] += v
        else:
            dicozipf[keyy] = v
    dftest_temp = pd.DataFrame.from_dict(dicozipf, orient='index', columns=['countUsages'])

    # extract the word and frequency values as separate lists
    sorted_freq = sorted(dicozipf.items(), key=lambda x: x[1], reverse=True)
    prefs = [w[0] for w in sorted_freq]
    freqs = [w[1] for w in sorted_freq]

    # plot the frequency distribution on a log-log scale
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(range(len(freqs)), freqs, color='red', s=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Label Frequency Rank')
    ax.set_ylabel('Label Frequency Value')
    ax.set_title('Log-log plot of frequency versus rank of label used by autocomplete users')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    # plt.show()
    plt.close()

    # Show top 50 selected labels
    print(dftest_temp.sort_values(by=['countUsages'], ascending=False).head(50))

    # Radix tree
    dicozipf = dict()
    for k, v in nUsesdict.items():
        keyy = k.lower()
        if keyy in dicozipf.keys():
            dicozipf[keyy] += v
        else:
            dicozipf[keyy] = v

    # # 1- Build radix tree for each initial
    # allwords = [w for w in dicozipf.keys() if is_it_French_word(w)]
    # initials = set([w[0] for w in allwords])
    # for initial in initials:
    #     print("Initial:", initial)
    #     words = [w for w in allwords if w.startswith(initial)]
    #     root = Node()
    #     for word in words:
    #         insertWord(root, word)
        
    #     print("There are", len(words), "words in total.")
        
    #     results = []
    #     prefix = ''
    #     allWords(prefix, root, results)
    #     # prefix will be added to every word found in the result, so we start with ''
    #     # results is empty, passed as reference so all results are stored in this list
    #     print('All words in trie: {}\n\n'.format(results))
    #     subdico = dict()
    #     for k in dicozipf.keys():
    #         if k.startswith(initial):
    #             subdico[k] = dicozipf[k]
    #     plotTrie(root, True, subdico, initial)
    
    # 2- Or, build radix tree for all words but without displaying labels
    words = [w for w in dicozipf.keys() if is_it_French_word(w)]
    root = Node()
    for word in words:
        insertWord(root, word)
    
    print("There are", len(words), "words in total.")
    
    results = []
    prefix = ''
    allWords(prefix, root, results)
    plotTrie(root, False, dicozipf)

    # # 3- Diverse function: search if a word exists
    # search_word = 'hypertension'
    # search_result = searchWord(root, search_word)
    # print('Search {}: {}'.format(search_word, search_result))

    # # 4- Diverse function: search a list of results given a prefix
    # prefix_result = []
    # prefix = 'hyp'
    # getWordsWithPrefix(prefix, root, prefix_result)
    # print('\n\nWords starting with {}: {}'.format(prefix, prefix_result))

