import sys
import networkx as nx
import plotly.graph_objects as go
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

    graph = build_networx_graph(graph)
    draw_graph(graph, word1, word2)


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


def build_networx_graph(graph):
    g = nx.DiGraph()

    for node in graph.keys():
        g.add_node(str(node.name()))

    for node_from in graph.keys():
        for node_to in graph[node_from]:
            g.add_edge(str(node_from.name()), str(node_to.name()))

    pos = nx.planar_layout(g)

    for node in g.nodes():
        g.nodes[node]['pos'] = pos[node]

    return g


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


if __name__ == '__main__':
    main()

