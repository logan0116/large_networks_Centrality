#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 上午9:27
# @Author  : liu yuhan
# @FileName: sh_effect.py
# @Software: PyCharm

import numpy as np
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm

from utils import *


def constraint_single(p):
    """
    第一次获取i的邻域
    第二次获取j的领域
    :param i:
    :param node2neighbor:
    :return:
    """
    i, node2neighbor = p
    print(i)
    constraint = 0
    i2j = node2neighbor[i]
    sum_i = np.sum(list(i2j.values()))

    for j, zij in i2j.items():
        if i == j:
            continue
        ll = zij / sum_i

        j2q = node2neighbor[j]
        for q in j2q:
            if q == i or q == j:
                continue
            try:
                # 判断q是不是在i的邻域
                ziq = i2j[q]
                piq = ziq / sum_i
                q2j = node2neighbor[q]
                sum_q = np.sum(list(q2j.values()))
                zqj = q2j[j]
                pqj = zqj / sum_q
                ll += piq * pqj
            except KeyError:
                continue
        constraint += ll ** 2
    return i, constraint


def constraint_mp(label, link_list):
    inst_graph = nx.Graph()
    inst_graph.add_edges_from(link_list)
    node_list = list(inst_graph.nodes)
    num_node = inst_graph.number_of_nodes()
    print(label, 'num_node', num_node, 'num_edge', len(link_list))

    node2neighbor = get_neighbor(link_list)

    node_list_list = [node_list[i:i + 10000] for i in range(0, len(node_list), 10000)]

    constraint = []

    for node_list in tqdm(node_list_list):
        pool = mp.Pool()
        constraint_ = pool.map(constraint_single,
                               zip(node_list,
                                   [node2neighbor for _ in range(len(node_list))]))
        pool.close()
        constraint.extend(constraint_)
    print(label, 'finish', len(constraint))
    return dict(constraint)


if __name__ == '__main__':
    for label in ['tech', 'industry', 'all']:
        link_list = get_network(label=label)
        effect = constraint_mp(label, link_list)
        with open('../data/output/constraint_%s_1209.json' % label, 'w', encoding='UTF-8') as f:
            json.dump(effect, f)
