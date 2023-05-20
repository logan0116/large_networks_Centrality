#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 下午4:30
# @Author  : liu yuhan
# @FileName: clossness_mp_v1.py
# @Software: PyCharm


import multiprocessing as mp
from functools import partial
from utils import *
import time
from networkx.algorithms.centrality.betweenness import _single_source_shortest_path_basic, _accumulate_basic, _rescale
from tqdm import tqdm


def remove_degree1(graph_sub, node2neighbor):
    # remove_degree1
    degree = nx.degree(graph_sub)
    node_degree1 = [node for node, deg in degree if deg == 1]
    graph_sub.remove_nodes_from(node_degree1)
    node_degree12neighbor = {node: list(node2neighbor[node])[0] for node in node_degree1}
    return graph_sub, node_degree12neighbor


def get_path_single(graph_min, node_list, node_degree12neighbor, s):
    betweenness_sub = dict.fromkeys(node_list, 0.0)  # b[v]=0 for v in G
    S, P, sigma, _ = _single_source_shortest_path_basic(graph_min, s)

    # 需要基于node_degree12neighbor修改S, P, sigma
    neighbors = []
    for node_degree1, neighbor in node_degree12neighbor.items():
        P[node_degree1] = [neighbor]
        sigma[node_degree1] = sigma[neighbor]
        S.append(node_degree1)
        if neighbor == s:
            neighbors.append(node_degree1)

    for node in [s] + neighbors:
        # print(node)
        P_ = P.copy()
        if node != s:
            P_[s] = [node]
            P_[node] = []

        betweenness_sub, _ = _accumulate_basic(betweenness_sub, S.copy(), P_, sigma, node)

    return betweenness_sub


def get_betweenness_mp(graph, node2neighbor):
    # 计算距离
    betweenness = dict.fromkeys(graph, 0.0)  # b[v]=0 for v in G

    for c in nx.connected_components(graph):
        # 对于每一个连通体育分别计算
        graph_sub = graph.subgraph(c).copy()
        n = graph_sub.number_of_nodes()
        node_list_sub = list(graph_sub.nodes)
        if n == 2:
            continue
        else:
            # 网络中有多个节点
            graph_sub_mini, node_degree12neighbor = remove_degree1(graph_sub, node2neighbor)
            node_list_sub_mini = list(graph_sub_mini.nodes)

            if graph_sub_mini.number_of_nodes() == 1:
                # 星形网络
                # 中心点
                node = node_list_sub_mini[0]
                num_neighbor = len(node2neighbor[node])
                betweenness[node] = num_neighbor * (num_neighbor - 1)
            else:
                # 优化内存
                # 2000个节点一组

                for bit in tqdm(range(0, n, 1000)):

                    pool = mp.Pool()
                    func = partial(get_path_single, graph_sub_mini, node_list_sub, node_degree12neighbor)
                    result = pool.map(func, node_list_sub_mini[bit:bit + 1000])
                    pool.close()
                    pool.join()

                    for betweenness_sub in result:
                        for node, bc in betweenness_sub.items():
                            betweenness[node] += bc

    betweenness = _rescale(
        betweenness,
        len(graph),
        normalized=True,
        directed=False,
        k=None,
        endpoints=False,
    )
    return betweenness


if __name__ == '__main__':
    for label in ['tech', 'industry', 'all']:
        link_list = get_network(label=label)
        graph = nx.Graph()
        graph.add_edges_from(link_list)
        node2neighbor = get_neighbor_unweighted(link_list)
        betweenness = get_betweenness_mp(graph, node2neighbor)
        with open('../data/output/betweenness_%s_1209.json' % label, 'w', encoding='UTF-8') as f:
            json.dump(betweenness, f)

    # with open('../data/test_2000.json', 'r') as f:
    #     link_list = json.load(f)
    #
    # graph_test = nx.Graph()
    # graph_test.add_edges_from(link_list)
    # node2neighbor = get_neighbor_unweighted(link_list)
    #
    # start = time.time()
    # clossness = get_betweenness_mp(graph_test, node2neighbor)
    # print(time.time() - start)
    #
    # start = time.time()
    # clossness_nx = nx.betweenness_centrality(graph_test)
    # print(time.time() - start)
    #
    # for node, cc in clossness.items():
    #     if abs(cc - clossness_nx[node]) > 1e-12:
    #         print(node, cc, clossness_nx[node])
