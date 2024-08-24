#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>

Graph::Graph() : num_nodes(0), num_primal(0), num_dual(0), num_edges(0)
{
    adj_list.clear();
}

bool Graph::is_primal(const int idx)
{
    assert(idx >= 0 && idx < num_nodes);
    return (idx < num_primal);
}

Graph::Graph(const int _num_primal, const int _num_dual, const int _num_edges, const int* edges_from, const int* edges_to)
        : num_primal(_num_primal), num_dual(_num_dual), num_edges(_num_edges)
{
    num_nodes = num_primal + num_dual;

    adj_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        adj_list[i].clear();
        
    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];

        // assert a bipartite graph
        assert( is_primal(x) ^ is_primal(y) );
        adj_list[x].push_back( y );
        adj_list[y].push_back( x );
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