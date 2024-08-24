import networkx as nx
import data_utils
import argparse


def greedy(g):
    """
    Greedy heuristic for Max-IS 
    :param g: A networkx graph
    :return: An independent set of nodes
    """

    # get neighbours and degree for each node
    neighbours_degrees = [(n, set(nx.neighbors(g, n))) for n in g.nodes()]
    
    mis = set()
    while len(neighbours_degrees) != 0:
        # add node with lowest degree to set
        neighbours_degrees.sort(key=lambda x: len(x[1]))
        node, remove = neighbours_degrees[0]
        mis.add(node)

        # remove the node and its neighbours
        neighbours_degrees = [(n, neigh - remove) for n, neigh in neighbours_degrees[1:] if n not in remove]
    return mis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str,  help='the name of the data set')
    args = parser.parse_args()

    print('loading graphs...')
    names, graphs = data_utils.load_graphs(args.data_path)
    for n, g in zip(names, graphs):
        mis = greedy(g)
        print(f'IS size for instance {n}: {len(mis)}')


if __name__ == '__main__':
    main()
