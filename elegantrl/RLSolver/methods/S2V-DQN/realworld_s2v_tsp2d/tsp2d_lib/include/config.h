#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>
#include "util/gnn_macros.h"

typedef float Dtype;

#ifdef GPU_MODE
    typedef gnn::GPU mode;
#else
    typedef gnn::CPU mode;
#endif

struct cfg
{
    static int max_bp_iter;
    static int embed_dim;
    static int batch_size;
    static int max_iter;
    static int dev_id;
    static int max_n, min_n;
    static int n_step;
    static int num_env;
    static int mem_size;
    static int reg_hidden;
    static int node_dim;
    static int knn;
    static int edge_dim;
    static int edge_embed_dim;
    static int aux_dim;
    static Dtype decay;
    static Dtype learning_rate;
    static Dtype l2_penalty;
    static Dtype momentum;    
    static Dtype w_scale;
    static const char *save_dir, *net_type; 

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
		    if (strcmp(argv[i], "-learning_rate") == 0)
		        learning_rate = atof(argv[i + 1]);
            if (strcmp(argv[i], "-max_bp_iter") == 0)
                max_bp_iter = atoi(argv[i + 1]);        
            if (strcmp(argv[i], "-dev_id") == 0)
                dev_id = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-embed_dim") == 0)
			    embed_dim = atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-knn") == 0)
			    knn = atoi(argv[i + 1]);                 
            if (strcmp(argv[i], "-edge_embed_dim") == 0)
			    edge_embed_dim = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-reg_hidden") == 0)
			    reg_hidden = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-max_n") == 0)
			    max_n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-min_n") == 0)
			    min_n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-mem_size") == 0)
			    mem_size = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-num_env") == 0)
			    num_env = atoi(argv[i + 1]);                
            if (strcmp(argv[i], "-n_step") == 0)
			    n_step = atoi(argv[i + 1]);                
    		if (strcmp(argv[i], "-batch_size") == 0)
	       		batch_size = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-max_iter") == 0)
	       		max_iter = atoi(argv[i + 1]);                   
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_penalty = atof(argv[i + 1]);      
            if (strcmp(argv[i], "-decay") == 0)
    			decay = atof(argv[i + 1]);      
            if (strcmp(argv[i], "-w_scale") == 0)
                w_scale = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-momentum") == 0)
    			momentum = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-save_dir") == 0)
    			save_dir = argv[i + 1];
            if (strcmp(argv[i], "-net_type") == 0)
    			net_type = argv[i + 1];                  
        }

        if (n_step <= 0)
            n_step = max_n;
        if (edge_embed_dim < 0)
            edge_embed_dim = embed_dim;
        std::cerr << "decay = " << decay << std::endl;
        std::cerr << "knn = " << knn << std::endl;
        std::cerr << "edge_embed_dim = " << edge_embed_dim << std::endl;
        std::cerr << "net_type = " << net_type << std::endl;
        std::cerr << "mem_size = " << mem_size << std::endl;
        std::cerr << "num_env = " << num_env << std::endl;    
        std::cerr << "n_step = " << n_step << std::endl;
        std::cerr << "min_n = " << min_n << std::endl;
        std::cerr << "max_n = " << max_n << std::endl;
        std::cerr << "max_iter = " << max_iter << std::endl;
        std::cerr << "dev_id = " << dev_id << std::endl;        
        std::cerr << "max_bp_iter = " << max_bp_iter << std::endl;
        std::cerr << "batch_size = " << batch_size << std::endl;        
        std::cerr << "embed_dim = " << embed_dim << std::endl;        
    	std::cerr << "learning_rate = " << learning_rate << std::endl;
        std::cerr << "w_scale = " << w_scale << std::endl;
    	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;    	
    }
};

#endif
