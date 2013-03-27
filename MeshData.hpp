/* MeshData.hpp header file for mesh related filenames and fucntions */
#include <vector>

struct node_struct
{
  int node_number;
  double x_coord;
  double y_coord;
  double z_coord;
};

struct tet_struct
{
  int tet_num;
  int link1;
  int link2;
  int link3;
  int link4;
  int mat;
  double volume;
};


/*
 * Load the mesh file
 */
int LoadMeshFile(std::string meshfile, std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data , int &num_vols);

/*
 * convert string to node component structure
 */
node_struct NodeLineToNodeData(std::string line_read);

/*
 * convert string to tet component structure
 */
tet_struct TetLineToTetData(std::string line_read, int num_vols);

/*
 * Print mesh data to vtk file
 */
void PrintMeshVtk(std::string meshfile, std::vector<node_struct> &node_data, std::vector<tet_struct> &tet_data);

