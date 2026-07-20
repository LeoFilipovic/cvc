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
  * Calculate in which Rcut_bin (x,y) lies
  ***********************************************************/
  int get_Rcut_bin(int const xv[4], int const xv_mi_yv[4], const int* Rcut2_bins, unsigned const Rcut_n)
  {
    int const x2 = xv[0]*xv[0] + xv[1]*xv[1] + xv[2]*xv[2] + xv[3]*xv[3];
    int const xmy2 = xv_mi_yv[0]*xv_mi_yv[0] + xv_mi_yv[1]*xv_mi_yv[1] + xv_mi_yv[2]*xv_mi_yv[2] + xv_mi_yv[3]*xv_mi_yv[3];
    int const r2 = (x2 <= xmy2) ? x2 : xmy2;

    if (r2 <= Rcut2_bins[0])
    {
      return 0;
    }

    for (int iRcut = 1; iRcut < Rcut_n; iRcut++)
    {
      if (Rcut2_bins[iRcut-1] < r2 && r2 <= Rcut2_bins[iRcut])
      {
        return iRcut;
      }
    }
    return Rcut_n - 1;
  }

  /***********************************************************
  * Calculate in which Zcut_bin z lies
  ***********************************************************/
  int get_Zcut_bin(int const zv[4], const int * Zcut2_bins, int const Zcut_n)
    {
      int const z2 = zv[0]*zv[0] + zv[1]*zv[1] + zv[2]*zv[2] + zv[3]*zv[3];

      if (z2 <= Zcut2_bins[0])
      {
        return 0;
      }

      for (int iZcut = 1; iZcut < Zcut_n; iZcut++)
      {
        if (Zcut2_bins[iZcut-1] < z2 && z2 <= Zcut2_bins[iZcut])
        {
          return iZcut;
        }
      }
      return Zcut_n - 1;
    }
    
}
