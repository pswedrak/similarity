from nltk.corpus import wordnet
import networkx as nx
import plotly.graph_objects as go

threshold = 8


def custom_metric(word1, word2):
    hypo_hyper = 0.7
    holo_mero = 0.7
    syno_anto = 0.9  # todo: consider synonyms-antonyms
    graph = nx.DiGraph()
    graph1 = {}
    graph2 = {}
    print("Name: ", wordnet.synsets(word1)[0].name())
    w1 = wordnet.synsets(word1)
    w2 = wordnet.synsets(word2)
    graph1['0'] = [w1[0]]
    graph2['0'] = [w2[0]]
    if set(w1).intersection(w2):
        return 0
    common_val = []
    graph1, graph2, path_len, graph, common = search_path(graph1, graph2, [w1[0]], [w2[0]], 1, nx_graph=graph, common=common_val)

    # DEBUG
    print('PATHLEN: ', path_len)
    print('COMMON: ', common)

    for key in graph1.keys():
        print('At the distance of ' + key + ' from \'' + word1 + '\': ' + str(graph1[key]))
    print()
    for key in graph2.keys():
        print('At the distance of ' + key + ' from \'' + word2 + '\': ' + str(graph2[key]))
    print()
    print()

    path = find_path(graph1, graph2, path_len, common)
    print('Path between concepts is: ', path)
    pos = nx.planar_layout(graph)

    for node in graph.nodes():
        graph.nodes[node]['pos'] = pos[node]
    return graph1, graph2, path_len, graph, common


flatten = lambda l: [item for sublist in l for item in sublist]


def find_path(graph1, graph2, path_len, common):
    list1 = []
    list2 = []
    conc = common
    for i in range(path_len[0]-1, 0, -1):
        key = str(i)
        for syns in conc:
            print(syns.hyponyms())
            print(syns.hypernyms())
            hypo_hyper = set(graph1[key]).intersection(set(syns.hyponyms() + syns.hypernyms()))
            holo_mero = set(graph1[key]).intersection(set(syns.part_holonyms() + syns.member_meronyms()))
            conc = list(hypo_hyper.union(holo_mero))
            list1.insert(0, conc)
    conc = common
    for i in range(path_len[1]-1, 0, -1):
        key = str(i)
        for syns in conc:
            hypo_hyper = set(graph2[key]).intersection(set(syns.hyponyms() + syns.hypernyms()))
            holo_mero = set(graph2[key]).intersection(set(syns.part_holonyms() + syns.member_meronyms()))
            conc = list(hypo_hyper.union(holo_mero))
            list2.append(conc)
    print('Whole path: ', flatten(list1)+list(common)+flatten(list2))
    return flatten(list1)+list(common)+flatten(list2)


def add_to_graph(synsetsU, synsetsV, nx_graph):
    nx_graph.add_nodes_from([synU.name() for synU in synsetsU])
    nx_graph.add_nodes_from(synV.name() for synV in flatten(synsetsV))
    for i in range(0, len(synsetsU)):
        nx_graph.add_edges_from([(synsetsU[i].name(), synV.name()) for synV in synsetsV[i]])

    return nx_graph


def get_connected_synsets(w1, w2, nx_graph=None):
    w1_hypernyms = [w.hypernyms() for w in w1]
    w2_hypernyms = [w.hypernyms() for w in w2]
    w1_hyponyms = [w.hyponyms() for w in w1]
    w2_hyponyms = [w.hyponyms() for w in w2]
    w1_holonyms = [w.member_holonyms() + w.part_holonyms() for w in w1]
    w2_holonyms = [w.member_holonyms() + w.part_holonyms() for w in w2]
    w1_meronyms = [w.part_meronyms() + w.member_meronyms() for w in w1]
    w2_meronyms = [w.part_meronyms() + w.member_meronyms() for w in w2]

    if nx_graph is not None:
        nx_graph = add_to_graph(w1, w1_hypernyms, nx_graph)
        nx_graph = add_to_graph(w1, w1_hyponyms, nx_graph)
        nx_graph = add_to_graph(w1, w1_holonyms, nx_graph)
        nx_graph = add_to_graph(w1, w1_meronyms, nx_graph)
        nx_graph = add_to_graph(w2, w2_hypernyms, nx_graph)
        nx_graph = add_to_graph(w2, w2_hyponyms, nx_graph)
        nx_graph = add_to_graph(w2, w2_holonyms, nx_graph)
        nx_graph = add_to_graph(w2, w2_meronyms, nx_graph)
        return [w1_hypernyms, w1_hyponyms, w1_holonyms, w1_meronyms], [w2_hypernyms, w2_hyponyms, w2_holonyms, w2_meronyms], nx_graph

    return [w1_hypernyms, w1_hyponyms, w1_holonyms, w1_meronyms], [w2_hypernyms, w2_hyponyms, w2_holonyms, w2_meronyms]


def search_path(graph1, graph2, w1, w2, depth, nx_graph=None, common=None):
    if depth > threshold:
        return graph1, graph2, float('Inf')

    w1_syns, w2_syns, nx_graph = get_connected_synsets(w1, w2, nx_graph=nx_graph)
    graph1[str(depth)] = list(set(flatten(flatten(w1_syns))))
    graph2[str(depth)] = list(set(flatten(flatten(w2_syns))))
    path_len = 0
    flag = False
    for key1 in graph1.keys():
        if flag:
            break
        for key2 in graph2.keys():
            l = set(graph1[key1]).intersection(set(graph2[key2]))
            if len(set(graph1[key1]).intersection(set(graph2[key2]))) > 0:
                path_len = (int(key1), int(key2))
                common = set(graph1[key1]).intersection(set(graph2[key2]))
                flag = True
                break

    if not flag:
        graph1, graph2, path_len, nx_graph, common = search_path(graph1, graph2,  graph1[str(depth)],  graph2[str(depth)], depth + 1, nx_graph=nx_graph, common=common)
    return graph1, graph2, path_len, nx_graph, common


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
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_adjacencies = []
    node_text = []

    for node in G.nodes():
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top right',
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


g = nx.DiGraph()
# custom_metric('cat', 'dog')
# custom_metric('cat', 'pillow')
graph1, graph2, pathlen, nx_graph, comm = custom_metric('cat', 'dog')

draw_graph(nx_graph, 'dog', 'cat')

