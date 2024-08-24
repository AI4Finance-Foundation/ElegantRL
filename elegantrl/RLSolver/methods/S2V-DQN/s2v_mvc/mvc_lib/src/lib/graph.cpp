#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>

Graph::Graph() : num_nodes(0), num_edges(0)
{
    edge_list.clear();
    adj_list.clear();
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to)
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        adj_list[i].clear();

    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);
        adj_list[y].push_back(x);
        edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
    }
}

GSet::GSet()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0);

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid];
}

std::shared_ptr<Graph> GSet::Sample()
{
    assert(graph_pool.size());
    int gid = rand() % graph_pool.size();
    assert(graph_pool[gid]);
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest;