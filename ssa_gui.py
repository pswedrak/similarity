import sys
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QLabel, QCheckBox
from PyQt5.QtCore import pyqtSlot


class Node:
    def __init__(self, name):
        self.name = name
        self.depth = 0


def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


def calculate_similarity(word1, word2):
    concepts = wn.synsets(word1, pos='n') + wn.synsets(word2, pos='n')
    if len(concepts) == 0:
        return None, 0
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
    max_depth = 20
    dist = np.finfo(np.float64).max
    path = None

    for c_i in wn.synsets(word1, pos='n'):
        for c_j in wn.synsets(word2, pos='n'):
            if g.has_node(str(c_i.name())) & g.has_node(str(c_j.name())):
                lch = c_i.lowest_common_hypernyms(c_j)[0]
                lch_value = g.nodes()[str(lch.name())]['depth']
                pl = nx.dijkstra_path_length(g, str(c_i.name()), str(c_j.name()))
                gloss = gloss_value(c_i, c_j)
                current_dist = pl * (1 - lch_value / max_depth) * (1 + gloss)
                if current_dist < dist:
                    dist = current_dist
                    path = nx.dijkstra_path(g, str(c_i.name()), str(c_j.name()))

    if dist == np.finfo(np.float64).max:
        dist = 0
        return g, dist, path
    else:
        dist = np.exp(-dist / 4)
        return g, dist, path


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
            g.add_edge(str(node_to.name()), str(node_from.name()), weight=0)

    max_depth = 0
    for node in g.nodes():
        depth = nx.shortest_path_length(g, rootnode, node)
        if depth > max_depth:
            max_depth = depth
        g.nodes[node]['depth'] = depth

    for edge in g.edges:
        depth_from = g.nodes()[edge[0]]['depth']
        depth_to = g.nodes()[edge[1]]['depth']
        weight = 1 - ((depth_from + depth_to)/(2*MAX))
        g.edges[edge]['weight'] = weight

    return g, max_depth


def draw_graph(G, word1, word2, path):
    fig = go.Figure(layout=go.Layout(
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

    concepts1 = wn.synsets(word1, pos='n')
    concepts1 = list(map(lambda concept: concept.name(), concepts1))

    concepts2 = wn.synsets(word2, pos='n')
    concepts2 = list(map(lambda concept: concept.name(), concepts2))

    edges_in_shortest_path = []
    for i in range(len(path)-1):
        edges_in_shortest_path.append((path[i], path[i+1]))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_y.append(y0)
        edge_x.append(x1)
        edge_y.append(y1)

        if edge in edges_in_shortest_path:
            edge_color = 'black'
        else:
            edge_color = 'lightblue'

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=edge_color),
            hoverinfo='text',
            textposition='top right',
            mode='lines')
        fig.add_trace(edge_trace)

    node_x = []
    node_y = []
    node_colours = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        if node in concepts1:
            node_colours.append('#ff0000')
        elif node in concepts2:
            node_colours.append('#000000')
        else:
            node_colours.append('#ffffff')

    node_text = []

    for node in G.nodes.items():
        node_text.append(node[0] + ' depth: ' + str(node[1]['depth']))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colours,
            size=10,
            line_width=2))
    fig.add_trace(node_trace)
    fig.show()


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Semantic similarity'
        self.left = 300
        self.top = 300
        self.width = 425
        self.height = 220
        self.textbox1 = QLineEdit(self)
        self.textbox2 = QLineEdit(self)
        self.textbox3 = QLineEdit(self)
        self.l1 = QLabel(self)
        self.l2 = QLabel(self)
        self.l3 = QLabel(self)
        self.l4 = QLabel(self)
        self.checkbox = QCheckBox(self)
        self.button = QPushButton('Calculate', self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.l1.move(20, 13)
        self.l1.setText("word 1: ")
        self.textbox1.move(120, 20)
        self.textbox1.resize(280, 20)

        self.l2.move(20, 43)
        self.l2.setText("word 2: ")
        self.textbox2.move(120, 50)
        self.textbox2.resize(280, 20)

        self.l3.move(20, 73)
        self.l3.setText("draw graph")

        self.checkbox.move(120, 73)
        self.button.move(20, 123)

        self.l4.move(20, 163)
        self.l4.setText("result:")
        self.textbox3.move(120, 170)
        self.textbox3.resize(280, 20)
        self.textbox3.setReadOnly(True)

        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        word1 = self.textbox1.text()
        word2 = self.textbox2.text()
        g, dist, path = calculate_similarity(word1, word2)
        self.textbox3.setText(str(np.round(dist * 10, 2)))
        if self.checkbox.isChecked():
            draw_graph(g, word1, word2, path)


if __name__ == '__main__':
    main()

