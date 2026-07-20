#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include "global.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "mpi_init.h"
#include "Q_phi.h"
#include "cvc_utils.h"
#include "scalar_products.h"
#include "integration_bins.h"

namespace cvc {
    /***********************************************************
    * max lattice side length
    ***********************************************************/
    int get_Lmax()
    {
    int Lmax = 0;
    if ( T_global >= Lmax ) Lmax = T_global;
    if ( LX_global >= Lmax ) Lmax = LX_global;
    if ( LY_global >= Lmax ) Lmax = LY_global;
    if ( LZ_global >= Lmax ) Lmax = LZ_global;
    return Lmax;
    }


    /***********************************************************
    * x must be in { 0, ..., L-1 }
    * mapping as in 2006.16224, eq. 8
    ***********************************************************/
    

    void site_map (int xv[4], int const x[4])
    {
    xv[0] = ( x[0] >= T_global   / 2 ) ? (x[0] - T_global )  : x[0];
    xv[1] = ( x[1] >= LX_global  / 2 ) ? (x[1] - LX_global)  : x[1];
    xv[2] = ( x[2] >= LY_global  / 2 ) ? (x[2] - LY_global)  : x[2];
    xv[3] = ( x[3] >= LZ_global  / 2 ) ? (x[3] - LZ_global)  : x[3];

    return;
    }

    /***********************************************************
    * as above, but set L/2 to 0 and -L/2 to 0
    ***********************************************************/
    void site_map_zerohalf (int xv[4], int const x[4])
    {
    xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
    xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
    xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
    xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );

    return;
    }
    
}