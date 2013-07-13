/*
 * preproccessing
 */
int PreProcess(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data);
int CalculateVolume(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data);
int shared(int node_test, int index, std::vector<tet_struct> tet_data);
int DetermineAdjacancy(std::vector<node_struct> node_data,std::vector<tet_struct>tet_data);

int CudaPreProcess(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data);
