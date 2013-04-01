/*
 * preproccessing
 */
int PreProcess(std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data);

/*
 *  ReNumberTets
 */ 
int ReNumberTets(std::vector<node_struct> node_data_original, 
		 std::vector<tet_struct> &tet_data); 

/*
 *  ReNumberNodes
 */ 
int ReNumberNodes(std::vector<node_struct> &node_data, 
		  std::vector<node_struct> &node_data_original); 

/*
 * Calculate tet volumes on the basis of the tet vertices
 */
int CalculateTetVolume(std::vector<node_struct> node_data_original, 
		 std::vector<tet_struct> &tet_data);

struct vector
{
    double i;
    double j;
    double k;
};