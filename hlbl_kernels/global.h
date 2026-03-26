/* define some global variables */

#ifndef _GLOBAL_H
#define _GLOBAL_H

#include <mpi.h>

#define _GSI(x) 24*x
#define T_global 16
#define LX_global 16
#define LY_global 16
#define LZ_global 16


#define NPROCT 1
#define NPROCX 1
#define NPROCY 1
#define NPROCZ 1 

const int T = T_global/NPROCT;
const int LX = LX_global/NPROCX;
const int LY = LX_global/NPROCY;
const int LZ = LX_global/NPROCZ;

const int kernel_n=3;
const int kernel_n_geom=3;


// MPI
extern MPI_Comm g_cart_grid;
extern int g_cart_id;
extern int g_proc_coords[4];
#endif