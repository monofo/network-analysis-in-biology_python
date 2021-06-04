import collections
import itertools
import math
import random
from itertools import combinations, groupby

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy import array
from scipy.special import comb
import statistics
from itertools import chain



def average_shortest_path_length(G):
    # This includes the isolated node!
    # https://stackoverflow.com/questions/48321963/how-to-generate-a-random-network-but-keep-the-original-node-degree-using-network/48323124

    path_lengths = (x.values() for x in dict(nx.shortest_path_length(G)).values())
    return statistics.mean(chain.from_iterable(path_lengths))

def max_degree(G):
    return max([d for n, d in G.degree()])


def gnp_random_connected_graph(n, p):
    """
    https://stackoverflow.com/questions/65157955/the-mean-distance-of-the-nodes-increases-by-log-of-the-number-of-nodes-python

    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


def plot_degree_distribution(
                            G, 
                            lines, 
                            title="", 
                            xlab="k", 
                            ylab="p(K)", 
                            xlim=None,
                            ylim=None,
                            log=True
                        ):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = np.array(cnt) / G.size()
    max_deg = max(degree_sequence)

    fig, ax = plt.subplots()
    plt.plot(deg, cnt, "bo")
    if lines is not None:
        plt.plot(range(max_deg+1), lines, "r")

    plt.title(title)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.xlim(ylim)

    if log:
        plt.yscale("log")
        plt.xscale("log")
    plt.show()
