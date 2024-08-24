import numpy as np
import networkx as nx

import os
import random
import sys

def build_full_graph(pathtofile, graphtype):
	node_dict = {}

	if graphtype == 'undirected':
		g = nx.Graph()
	elif graphtype == 'directed':
		g = nx.DiGraph()
	else:
		print('Unrecognized graph type .. aborting!')
		return -1

	times = []
	with open(pathtofile) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	content = content[1:]

	for line in content:
		entries = line.split()
		src_str = entries[1]
		dst_str = entries[2]

		if src_str not in node_dict:
			node_dict[src_str] = len(node_dict)
			g.add_node(node_dict[src_str])
		if dst_str not in node_dict:
			node_dict[dst_str] = len(node_dict)
			g.add_node(node_dict[dst_str])

		src_idx = node_dict[src_str]
		dst_idx = node_dict[dst_str]

		w = 0
		c = 0
		if g.has_edge(src_idx,dst_idx):
			w = g[src_idx][dst_idx]['weight']
			c = g[src_idx][dst_idx]['count']

		g.add_edge(src_idx,dst_idx,weight=w + 1.0/float(entries[-1]),count=c + 1)

		times.append(float(entries[-1]))

	for edge in g.edges_iter(data=True):
		src_idx = edge[0]
		dst_idx = edge[1]
		w = edge[2]['weight']
		c = edge[2]['count']
		g[src_idx][dst_idx]['weight'] = w/c

	return g, node_dict

def get_mvc_graph(ig,prob_quotient=10):
	g = ig.copy()
	# flip coin for each edge, remove it if coin has value > edge probability ('weight')
	for edge in g.edges_iter(data=True):
		src_idx = edge[0]
		dst_idx = edge[1]
		w = edge[2]['weight']
		coin = random.random()
		if coin > w/prob_quotient:
			g.remove_edge(src_idx,dst_idx)

	# get set of nodes in largest component
	cc = sorted(nx.connected_components(g), key = len, reverse=True)
	lcc = cc[0]

	# remove all nodes not in largest component
	numrealnodes = 0
	node_map = {}
	for node in g.nodes():
		if node not in lcc:
			g.remove_node(node)
			continue
		node_map[node] = numrealnodes
		numrealnodes += 1

	# re-create the largest component with nodes indexed from 0 sequentially
	g2 = nx.Graph()
	for edge in g.edges_iter(data=True):
		src_idx = node_map[edge[0]]
		dst_idx = node_map[edge[1]]
		w = edge[2]['weight']
		g2.add_edge(src_idx,dst_idx,weight=w)

	return g2

def get_scp_graph(ig,prob_quotient=10):
	g = ig.copy()
	# flip coin for each edge, remove it if coin has value > edge probability ('weight')
	for edge in g.edges_iter(data=True):
		src_idx = edge[0]
		dst_idx = edge[1]
		w = edge[2]['weight']
		coin = random.random()
		if coin > w/prob_quotient:
			g.remove_edge(src_idx,dst_idx)

	# remove nodes with in-degree and out-degree both 0
	numrealnodes = 0
	node_map = {}
	for node in g.nodes():
		if g.degree(node) == 0:
			g.remove_node(node)
			continue
		node_map[node] = numrealnodes
		numrealnodes += 1		

	# re-index nodes from 0 sequentially
	g2 = nx.DiGraph()
	for edge in g.edges_iter(data=True):
		src_idx = node_map[edge[0]]
		dst_idx = node_map[edge[1]]
		g2.add_edge(src_idx,dst_idx)

	# get each node's reachable set of descendants; keep track of number of sets and elements
	num_sets = 0
	num_elements = 0#nx.number_of_nodes(g2)
	set_map = {}
	element_map = {}
	node_desc = {}
	for node in g2.nodes():
		if g2.out_degree(node) > 0:
			descendants = nx.descendants(g2,node)
			node_desc[node] = descendants
			set_map[node] = num_sets
			num_sets += 1
		if g2.in_degree(node) > 0:
			element_map[node] = num_elements
			num_elements += 1

	# build bipartite graph
	a = range(num_sets)
	b = range(num_sets, num_sets + num_elements)
	bg = nx.Graph()
	bg.add_nodes_from(a, bipartite=0)
	bg.add_nodes_from(b, bipartite=1)
	for rset in node_desc:
		src_idx = set_map[rset]
		for desc in node_desc[rset]:
			dst_idx = element_map[desc] + num_sets
			bg.add_edge(src_idx,dst_idx)

	# if element is in only one set, add it to another random set
	for el in range(num_sets, num_sets + num_elements):
		if bg.degree(el) == 1:
			randset = np.random.randint(num_sets)
			while bg.has_edge(el,randset):
				randset = np.random.randint(num_sets)
			bg.add_edge(el,randset)

	# for s in range(num_sets):
	# 	print(bg.degree(s))
	# for el in range(num_sets, num_sets + num_elements):
	# 	if bg.degree(el) <= 1:
	# 		print('HERE')
	# 		print(bg.degree(el))

	return bg

def visualize(g,pdfname='graph.pdf'):
	import matplotlib.pyplot as plt
	pos=nx.spring_layout(g,iterations=100) # positions for all nodes
	nx.draw_networkx_nodes(g,pos,node_size=1)
	nx.draw_networkx_edges(g,pos)
	plt.axis('off')
	plt.savefig(pdfname,bbox_inches="tight")

def visualize_bipartite(g,pdfname='graph_scp.pdf'):
	import matplotlib.pyplot as plt
	X, Y = nx.bipartite.sets(g)
	pos = dict()
	pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
	pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
	nx.draw_networkx_nodes(g,pos,node_size=1)
	nx.draw_networkx_edges(g,pos)
	plt.axis('off')
	plt.savefig(pdfname,bbox_inches="tight")

if __name__ == '__main__':
	# build full graphs of both types
	print("Building undirected graph ...")
	g_undirected, node_dict = build_full_graph('InfoNet5000Q1000NEXP.txt','undirected')
	print("Building directed graph ...")
	g_directed, node_dict = build_full_graph('InfoNet5000Q1000NEXP.txt','directed')

	print(nx.number_of_nodes(g_undirected))
	print(nx.number_of_edges(g_undirected))

	print(nx.number_of_nodes(g_directed))
	print(nx.number_of_edges(g_directed))

