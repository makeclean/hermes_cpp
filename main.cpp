/* Main program for the c++ version of hermes-gpu */
#include <iostream>

int main ( int argc, char *argv[])
{
  // parse the command line arguments
  for ( int argnum = 0 ; argnum <= argc ; argnum++ )
    {
      std::cout << argv[argnum] << std::endl;
    }

  return 0;
}


