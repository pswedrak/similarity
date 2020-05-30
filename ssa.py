import sys
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Node:
    def __init__(self, name):
        self.name = name
        self.depth = 0


def main():
    if len(sys.argv) == 3:
        word1 = sys.argv[1]
        word2 = sys.argv[2]
        g, dist = calculate_similarity(word1, word2)
        print(dist)
        draw_graph(g, word1, word2)
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        filename_results = filename.split(".")[0] + '_results.txt'
        fp2 = open(filename_results, 'w')

        with open(filename) as fp:
            line = fp.readline()
            while line:
                split_line = line.split(' ')
                word1 = split_line[0]
                word2 = split_line[1][:-1]
                _, dist = calculate_similarity(word1, word2)
                fp2.write(word1 + ' ' + word2 + ' ' + str(round(dist*10, 2)))
                fp2.write('\n')
                line = fp.readline()
        fp.close()
        fp2.close()
    else:
        print('wrong number of arguments')


def calculate_similarity(word1, word2):
    concepts = wn.synsets(word1, pos='n') + wn.synsets(word2, pos='n')
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

    g, max_depth = build_networx_graph(graph, root)
    dists = []

    for c_i in wn.synsets(word1, pos='n'):
        for c_j in wn.synsets(word2, pos='n'):
            if g.has_node(str(c_i.name())) & g.has_node(str(c_j.name())):
                lch = c_i.lowest_common_hypernyms(c_j)[0]
                lch_node = g.nodes[str(lch.name())]
                lch_value = lch_node['depth']
                pl = shortest_path_length(g, str(lch.name()), str(c_i.name())) + shortest_path_length(g, str(lch.name()),
                                                                                              str(c_j.name()))
                gloss = gloss_value(c_i, c_j)
                dist = pl * (1 - lch_value / max_depth) * (1 + gloss)
                dists.append(dist)

    if len(dists) == 0:
        dist = 0
        return g, dist
    else:
        dist = min(dists)
        dist = np.exp(-dist / 8)
        return g, dist


def gloss_value(c_i, c_j):
    stop_words = set(stopwords.words('english'))
    filtered_words1 = [w for w in word_tokenize(c_i.definition()) if w not in stop_words and w.isalpha()]
    filtered_words2 = [w for w in word_tokenize(c_j.definition()) if w not in stop_words and w.isalpha()]
    common_words = [w for w in filtered_words1 + filtered_words2 if w in filtered_words1 and w in filtered_words2]
    max_value = max(len(filtered_words1), len(filtered_words2))
    return 1 - len(common_words) / max_value


def shortest_path_length(g, c_i, c_j):
    return nx.shortest_path_length(g, c_i, c_j)


def nearest_common_ancestor(g, c_i, c_j):
    return nx.lowest_common_ancestor(g, c_i, c_j)


def insert_hypernyms(graph, c, root):
        if c == root:
            return
        else:
            for hypernym in c.hypernyms() + c.instance_hypernyms():
                if hypernym in graph.keys():
                    if c not in graph[hypernym]:
                        graph[hypernym].append(c)
                else:
                    graph[hypernym] = [c]
                insert_hypernyms(graph, hypernym, root)


def build_unweighted_graph(graph, root):
    g = nx.DiGraph()
    MAX = 20

    for node in graph.keys():
        g.add_node(str(node.name()), depth=0)

    rootnode = None
    for node in g.nodes():
        if node == root.name():
            rootnode = node

    pos = nx.spring_layout(g)

    for node in g.nodes():
        g.nodes[node]['pos'] = pos[node]

    for node_from in graph.keys():
        for node_to in graph[node_from]:
            g.add_edge(str(node_from.name()), str(node_to.name()), weight=0)

    return g


def build_networx_graph(graph, root):
    g = nx.DiGraph()
    MAX = 20

    for node in graph.keys():
        g.add_node(str(node.name()))

    rootnode = None
    for node in g.nodes():
        if node == root.name():
            rootnode = node

    pos = nx.spring_layout(g)

    for node in g.nodes():
        g.nodes[node]['pos'] = pos[node]

    for node_from in graph.keys():
        for node_to in graph[node_from]:
            g.add_edge(str(node_from.name()), str(node_to.name()), weight=0)

    max_depth = 0
    for node in g.nodes():
        depth = nx.shortest_path_length(g, rootnode, node)
        if depth > max_depth:
            max_depth = depth
        g.nodes[node]['depth'] = depth

    for edge in g.edges:
        depth_from = g.nodes()[edge[0]]['depth']
        depth_to = g.nodes()[edge[1]]['depth']
        g.edges[edge]['weight'] = 1 - (depth_from + depth_to)/2*MAX

    return g, max_depth


def draw_graph(G, word1, word2):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        textposition='top right',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_adjacencies = []
    node_text = []

    for node in G.nodes.items():
        node_text.append(node[0] + ' depth: ' + str(node[1]['depth']))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        #mode='markers+text',
        mode='markers',
        #text=node_text,
        #textposition='top right',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Rainbow',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Graph created for words: ' + word1 + ' and ' + word2,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()


if __name__ == '__main__':
    main()

