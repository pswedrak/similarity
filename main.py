import sys
from nltk.corpus import wordnet as wn


def main():
    word1 = sys.argv[1]
    word2 = sys.argv[2]

    concepts = wn.synsets(word1) + wn.synsets(word2)
    root = concepts[0].root_hypernyms()[0]

    graph = {root: []}

    for c in concepts:
        graph[c] = []

    for c in concepts:
        insert_hypernyms(graph, c, root)

    for c in concepts:
        for hyponym in c.hyponyms():
            if hyponym not in graph.keys():
                graph[hyponym] = []
            if hyponym not in graph[c]:
                graph[c].append(hyponym)

        for meronym in c.part_meronyms() + c.substance_meronyms():
            if meronym not in graph.keys():
                graph[meronym] = []
            if meronym not in graph[c]:
                graph[c].append(meronym)

        for holonym in c.part_holonyms() + c.substance_holonyms():
            if holonym not in graph.keys():
                graph[holonym] = []
            if holonym not in graph[c]:
                graph[c].append(holonym)

    for c in concepts:
        for cc in c.hyponyms() + c.part_meronyms() + c.substance_meronyms() + c.part_holonyms() + c.substance_holonyms():
            insert_hypernyms(graph, cc, root)

    print(graph)


def insert_hypernyms(graph, c, root):
        if c == root:
            return
        else:
            for hypernym in c.hypernyms():
                if hypernym in graph.keys():
                    if c not in graph[hypernym]:
                        graph[hypernym].append(c)
                else:
                    graph[hypernym] = [c]
                insert_hypernyms(graph, hypernym, root)


if __name__ == '__main__':
    main()

