#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 下午4:30
# @Author  : liu yuhan
# @FileName: clossness_mp_v1.py
# @Software: PyCharm

import networkx as nx
import multiprocessing as mp
from functools import partial
from utils import *
import numpy as np
import time
from tqdm import tqdm


def remove_degree1(graph_sub, node2neighbor):
    # remove_degree1
    degree = nx.degree(graph_sub)
    node_degree1 = [node for node, deg in degree if deg == 1]
    graph_sub.remove_nodes_from(node_degree1)
    node_degree12neighbor = {node: list(node2neighbor[node])[0] for node in node_degree1}
    return graph_sub, node_degree12neighbor


def get_length_single(graph_sub, node_degree12neighbor, k, node):
    path = nx.single_source_shortest_path_length(graph_sub, node)
    for node_degree1, neighbor in node_degree12neighbor.items():
        path[node_degree1] = path[neighbor] + 1
    sun_length = np.sum([p[1] for p in path.items()])

    return node, k / sun_length


def get_clossness(graph, node2neighbor):
    N = graph.number_of_nodes()
    result = dict()
    # 计算距离
    for c in nx.connected_components(graph):
        # 对于每一个连通体育分别计算
        graph_sub = graph.subgraph(c).copy()
        n = graph_sub.number_of_nodes()
        if n == 2:
            # 网络中只有两个节点
            node_list_sub = list(graph_sub.nodes)
            result[node_list_sub[0]] = 1 / (N - 1)
            result[node_list_sub[1]] = 1 / (N - 1)
        else:
            # 网络中有多个节点
            graph_sub_mini, node_degree12neighbor = remove_degree1(graph_sub, node2neighbor)
            num_graph_sub_mini = graph_sub_mini.number_of_nodes()
            node_list_sub_mini = list(graph_sub_mini.nodes)

            if num_graph_sub_mini == 1:
                # 星形网络
                # 中心点
                node = node_list_sub_mini[0]
                # (n - 1) * (n - 1) / (N - 1) / (n - 1) = (n - 1) / (N - 1)
                result[node] = (n - 1) / (N - 1)
                # 外围点
                for node in node_degree12neighbor:
                    # (n - 1) * (n - 1) / (N - 1) / ((n - 2) * 2 + 1)
                    result[node] = (n - 1) * (n - 1) / (N - 1) / (2 * n - 3)

            else:
                # 一般网络
                k = (n - 1) * (n - 1) / (N - 1)
                pool = mp.Pool()
                func = partial(get_length_single, graph_sub_mini, node_degree12neighbor, k)
                result_ = pool.map(func, node_list_sub_mini)
                pool.close()
                pool.join()
                result_ = dict(result_)
                for node_degree1, neighbor in node_degree12neighbor.items():
                    result_[node_degree1] = k / (k / result_[neighbor] + n - 2)
                result.update(result_)

    return result


if __name__ == '__main__':
    for i in tqdm(range(500)):
        graph_test = get_test_graph(2000, 2000)
        link_list = list(graph_test.edges)
        node2neighbor = get_neighbor_unweighted(link_list)

        # prepare
        clossness = get_clossness(graph_test, node2neighbor)

        clossness_nx = nx.closeness_centrality(graph_test)

        for node, cc in clossness.items():
            if abs(cc - clossness_nx[node]) > 1e-12:
                print(node, cc, clossness_nx[node])
