#ifndef TSP2D_LIB_H
#define TSP2D_LIB_H

extern "C" int Init(const int argc, const char** argv);

extern "C" int InsertGraph(bool isTest, const int g_id, const int num_nodes, const double* coor_x, const double* coor_y);

extern "C" int LoadModel(const char* filename);

extern "C" int SaveModel(const char* filename);

extern "C" int UpdateSnapshot();

extern "C" int ClearTrainGraphs();

extern "C" int PlayGame(const int n_traj, const double eps);

extern "C" double Fit(const double lr);

extern "C" double FitWithFarthest(const double lr);

extern "C" double Test(const int gid);

extern "C" int SetSign(int s);

extern "C" int ClearMem();

extern "C" double GetSol(const int gid, int* sol);

#endif