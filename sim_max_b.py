from nltk.corpus import wordnet

threshold = 8


def custom_metric(word1, word2):
    hypo_hyper = 0.7
    holo_mero = 0.7
    syno_anto = 0.9  # todo: consider synonyms-antonyms
    graph1 = {}
    graph2 = {}
    w1 = wordnet.synsets(word1)
    w2 = wordnet.synsets(word2)
    if set(w1).intersection(w2):
        return 0

    graph1, graph2, path_len = search_path(graph1, graph2, w1, w2, 1)

    # DEBUG
    print('PATHLEN: ', path_len)
    print()
    for key in graph1.keys():
        print('At the distance of ' + key + ' from \'' + word1 + '\': ' + str(graph1[key]))
    print()
    for key in graph2.keys():
        print('At the distance of ' + key + ' from \'' + word2 + '\': ' + str(graph2[key]))
    print()
    print()

    return path_len


flatten = lambda l: [item for sublist in l for item in sublist]


def search_path(graph1, graph2, w1, w2, depth):
    if depth > threshold:
        return graph1, graph2, float('Inf')

    w1_hypernyms = flatten([w.hypernyms() for w in w1])
    w2_hypernyms = flatten([w.hypernyms() for w in w2])
    w1_holonyms = flatten([w.part_holonyms() for w in w1])
    w2_holonyms = flatten([w.part_holonyms() for w in w2])
    w1_meronyms = flatten([w.member_meronyms() for w in w1])
    w2_meronyms = flatten([w.member_meronyms() for w in w2])

    graph1[str(depth)] = w1_hypernyms + w1_holonyms + w1_meronyms
    graph2[str(depth)] = w2_hypernyms + w2_holonyms + w2_meronyms
    path_len = 0
    flag = False
    for key1 in graph1.keys():
        for key2 in graph2.keys():
            if len(set(graph1[key1]).intersection(set(graph2[key2]))) > 0:
                path_len = int(key1) + int(key2)
                flag = True

    if not flag:
        graph1, graph2, path_len = search_path(graph1, graph2,  graph1[str(depth)],  graph2[str(depth)], depth + 1)
    return graph1, graph2, path_len


custom_metric('cat', 'dog')
custom_metric('cat', 'pillow')
custom_metric('dog', 'domestic_animal')

