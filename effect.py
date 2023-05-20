#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 上午9:27
# @Author  : liu yuhan
# @FileName: sh_effect.py
# @Software: PyCharm
import numpy as np
import networkx as nx
import multiprocessing as mp

from utils import *


def effect_single(p):
    """
    第一次获取i的邻域
    第二次获取j的领域
    :param i:
    :param node2neighbor:
    :return:
    """
    i, index, node2neighbor = p
    print(index)
    effect = 0
    i2j = node2neighbor[i]
    sum_i = np.sum(list(i2j.values()))

    for j in i2j:
        if i == j:
            continue
        sum_pm = 0
        j2q = node2neighbor[j]
        max_j = np.max(list(j2q.values()))
        for q, zjq in j2q.items():
            if q == i or q == j:
                continue
            try:
                # 判断q是不是在i的邻域
                ziq = i2j[q]
                piq = ziq / sum_i
                mjq = zjq / max_j
                sum_pm += piq * mjq
            except KeyError:
                continue
        effect += (1 - sum_pm)
    return i, effect


def effect_mp(label, link_list):
    """
    :param link_list:
    :return:
    """
    inst_graph = nx.Graph()
    inst_graph.add_edges_from(link_list)
    node_list = list(inst_graph.nodes)
    num_node = inst_graph.number_of_nodes()
    print(label, 'num_node', num_node, 'num_edge', len(link_list))

    node2neighbor = get_neighbor(link_list)

    pool = mp.Pool()
    effect = pool.map(effect_single,
                      zip(node_list,
                          [i for i in range(num_node)],
                          [node2neighbor for _ in range(num_node)])
                      )
    pool.close()
    return dict(effect)


if __name__ == '__main__':
    for label in ['tech', 'industry', 'all']:
        link_list = get_network(label=label)
        effect = effect_mp(label, link_list)
        with open('../data/output/effect_%s_1209.json' % label, 'w', encoding='UTF-8') as f:
            json.dump(effect, f)
