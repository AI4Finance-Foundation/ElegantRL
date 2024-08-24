#include "graph.h"
#include "config.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

#define inf 2147483647/2

Graph::Graph() : num_nodes(0), num_edges(0)
{
    coor_x.clear();
    coor_y.clear();
    dist.clear();
    adj_set.clear();
}

double Graph::euc_dist(const int i, const int j)
{
    double dx = coor_x[i] - coor_x[j];
    double dy = coor_y[i] - coor_y[j];
    return sqrt(dx * dx + dy * dy);
}

Graph::Graph(const int _num_nodes, const double* _coor_x, const double* _coor_y)
        : num_nodes(_num_nodes)
{
    coor_x.resize(num_nodes);
    coor_y.resize(num_nodes);
    dist.resize(num_nodes);
    adj_set.resize(num_nodes);

    for (int i = 0; i < num_nodes; ++i)
    {
        dist[i].resize(num_nodes);
        coor_x[i] = _coor_x[i];
        coor_y[i] = _coor_y[i];
    }

    std::vector< std::pair<int, double> > neighbors;

    for (int i = 0; i < num_nodes; ++i)
    {     
        adj_set[i].clear();
        neighbors.clear();
        for (int j = 0; j < num_nodes; ++j)
        {
            dist[i][j] = euc_dist(i, j);
            if (j == i)
                continue;

            neighbors.push_back( std::make_pair(j, dist[i][j]) );
        }
        std::sort(neighbors.begin(), neighbors.end(), []
            (const std::pair<int, double>& x, const std::pair<int, double>& y){
            return x.second < y.second;
        });

        int n = neighbors.size();
        if (cfg::knn >= 0 && n > cfg::knn)
            n = cfg::knn;

        for (int j = 0; j < n; ++j)
        {
            adj_set[i].insert(neighbors[j].first);
            adj_set[neighbors[j].first].insert(i);
        }
    }

    num_edges = 0;
    for (int i = 0; i < num_nodes; ++i)
        num_edges += adj_set[i].size();
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