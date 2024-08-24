#include "config.h"

int cfg::max_bp_iter = 1;
int cfg::embed_dim = 64;
int cfg::dev_id = 0;
int cfg::batch_size = 32;
int cfg::max_iter = 1;
int cfg::reg_hidden = 32;
int cfg::node_dim = 0;
int cfg::aux_dim = 0;
int cfg::min_n = 0;
int cfg::max_n = 0;
int cfg::mem_size = 0;
int cfg::num_env = 0;
int cfg::n_step = -1;
int cfg::knn = -1;
int cfg::edge_dim = 4;
int cfg::edge_embed_dim = -1;
Dtype cfg::learning_rate = 0.0005;
Dtype cfg::decay = 1.0;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0;
Dtype cfg::w_scale = 0.01;
const char* cfg::save_dir = "./saved";
const char* cfg::net_type = "QNet";