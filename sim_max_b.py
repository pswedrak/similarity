from nltk.corpus import wordnet
import networkx as nx
import plotly.graph_objects as go

threshold = 6
hh_hm_depth_factor = 0.7
hh_hm_link_factor = 0.85


def sim_max_b(word1, word2):
    graph = nx.DiGraph()
    graph1 = {}
    graph2 = {}
    w1 = wordnet.synsets(word1)
    w2 = wordnet.synsets(word2)
    graph1['0'] = w1
    graph2['0'] = w2

    if word1 == word2:
        return graph1, graph2, 1., graph, set(w1).intersection(w2), None

    if set(w1).intersection(w2) or is_syn_or_ano(w1, w2):
        return graph1, graph2, 0.9, graph, None, None
    common_val = []
    graph1, graph2, path_len, graph, common = search_path(graph1, graph2, w1, w2, 1, nx_graph=graph, common=common_val)
    if path_len == float('Inf'):
        return graph1, graph2, 0., graph, None, None

    path = find_path(graph1, graph2, path_len, common)

    try:
        pos = nx.planar_layout(graph)
        for node in graph.nodes():
            graph.nodes[node]['pos'] = pos[node]
    except:
        print('Graph is not planar, so no visualization will be provided')

    sim = hh_hm_link_factor * hh_hm_depth_factor**(path_len[0] + path_len[1] - 1)

    return graph1, graph2, sim, graph, common, path


flatten = lambda l: [item for sublist in l for item in sublist]


def is_syn_or_ano(w1, w2):
    synonyms1 = flatten([w.lemmas() for w in w1])
    antonyms1 = flatten([w.antonyms() for w in synonyms1])
    synonyms2 = flatten([w.lemmas() for w in w2])
    antonyms2 = flatten([w.antonyms() for w in synonyms2])
    return set(synonyms1).intersection(set(synonyms2)) or set(synonyms1).intersection(set(antonyms2)) or \
        set(synonyms2).intersection(set(antonyms1)) or set(antonyms1).intersection(set(antonyms2))


def find_path(graph1, graph2, path_len, common):
    list1 = []
    list2 = []
    conc = common
    for i in range(path_len[0]-1, -1, -1):
        key = str(i)
        for syns in conc:
            hypo_hyper = set(graph1[key]).intersection(set(syns.hyponyms() + syns.hypernyms()))
            holo_mero = set(graph1[key]).intersection(set(syns.part_holonyms() + syns.part_meronyms() +
                                                          syns.member_holonyms() + syns.member_meronyms()))

            conc = list(hypo_hyper.union(holo_mero))
            if conc not in list1:
                list1.insert(0, conc)
    conc = common
    for i in range(path_len[1]-1, -1, -1):
        key = str(i)
        for syns in conc:
            hypo_hyper = set(graph2[key]).intersection(set(syns.hyponyms() + syns.hypernyms()))
            holo_mero = set(graph2[key]).intersection(set(syns.part_holonyms() + syns.part_meronyms() +
                                                          syns.member_holonyms() + syns.member_meronyms()))

            conc = list(hypo_hyper.union(holo_mero))
            if conc not in list2:
                list2.append(conc)

    return [syn.name() for syn in flatten(list1)+list(common)+flatten(list2)]


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
        # print('Too far from ')
        return graph1, graph2, float('Inf'), nx_graph, common

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


def draw_graph(G, paths, word1, word2):
    edge_x = []
    edge_y = []
    edge_path_x = []
    edge_path_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        print('Paths: ', paths)
        if edge[0] in paths and edge[1] in paths:
            print('Appending: ', edge)
            edge_path_x.append(x0)
            edge_path_x.append(x1)
            edge_path_x.append(None)
            edge_path_y.append(y0)
            edge_path_y.append(y1)
            edge_path_y.append(None)

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

    edge_trace_path = go.Scatter(
        x=edge_path_x, y=edge_path_y,
        line=dict(width=0.9, color='#555'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_path_x = []
    node_path_y = []
    node_start_end_x = []
    node_start_end_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        if node == paths[0] or node == paths[len(paths)-1]:
            node_start_end_x.append(x)
            node_start_end_y.append(y)
        elif node in paths:
            print('Found: ', node, ' in ', paths)
            node_path_x.append(x)
            node_path_y.append(y)
        else:
            node_x.append(x)
            node_y.append(y)

    node_adjacencies = []
    node_text = []
    node_text_path = []
    node_text_start_end = []
    for node in G.nodes():
        if node == paths[0] or node == paths[len(paths) - 1]:
            node_text_start_end.append(node)
        elif node in paths:
            node_text_path.append(node)
        else:
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

    node_trace_path = go.Scatter(
        x=node_path_x, y=node_path_y,
        mode='markers+text',
        text=node_text_path,
        textposition='top right',
        marker=dict(
            showscale=False,
            colorscale='Rainbow',
            reversescale=True,
            color=[],
            size=20,
            line_width=3))

    node_trace_path_start_end = go.Scatter(
        x=node_start_end_x, y=node_start_end_y,
        mode='markers+text',
        text=node_text_start_end,
        textposition='top right',
        marker=dict(
            showscale=False,
            colorscale='Rainbow',
            reversescale=True,
            color=[],
            size=20,
            line_width=3))

    for node, adjacencies in enumerate(G.adjacency()):
        a = adjacencies  # a[0] - nazwa, a[1] sąsiedzi
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace_path.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, edge_trace_path, node_trace, node_trace_path, node_trace_path_start_end],
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

graph1, graph2, pathlen, nx_graph, comm, path = sim_max_b('car', 'truck')
draw_graph(nx_graph, path, 'car', 'truck')

def results_for_file(file):
    res = []
    act_exp = []
    with open(file, 'r') as f:
        lines = f.readlines()
        # print('Word1  Word2  Original  Actual')
        for line in lines:
            tokens = line.split()
            _, _, result, _, _ = sim_max_b(tokens[0], tokens[1])
            # print(tokens[0], ' ', tokens[1], ' ', tokens[2], ' ', result)
            res.append([tokens[0], tokens[1], tokens[2], result*10])
            act_exp.append([float(tokens[2]), result*10])
    return res, act_exp

# res, act_exp = results_for_file(Path('set1.txt'))
# print(tabulate(res, headers=['Word1',  'Word2',  'Original',  'Actual']))

