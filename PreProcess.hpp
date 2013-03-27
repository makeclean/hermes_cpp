/*
 * preproccessing
 */
int PreProcess(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data);

/*
 *  ReNumberTets
 */ 
int ReNumberTets(std::vector<tet_struct> &tet_data); 

/*
 *  ReNumberNodes
 */ 
int ReNumberNodes(std::vector<node_struct> &node_data); 