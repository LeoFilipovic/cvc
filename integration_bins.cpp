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
  int get_Rcut_bin(int const xv[4], int const xv_mi_yv[4], const int * Rcut2_bins, int const Rcut_n)
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
/*********************************************
 * complex-valued 4-dim scalar product of two
 * spinor fields with the binning in z
 *********************************************/
  void spinor_scalar_product_co_cut ( complex * const w, double * const xi, double * const phi, unsigned int const V) //, const int * Xcut2_bins, int const Xcut_n)
  {
    complex paccum;
    
    #ifdef HAVE_MPI
    
    #endif
    #ifdef HAVE_OPENMP
    omp_lock_t writelock;
    #endif
    paccum.re = 0.;
    paccum.im = 0.;

    #ifdef HAVE_OPENMP
    omp_init_lock(&writelock);
    #pragma omp parallel default(shared)
    {
    #endif
    complex p2;
    p2.re = 0.;
    p2.im = 0.;

    int gsx[4] = {0,0,0,0};

    #ifdef HAVE_OPENMP
    #pragma omp for
    #endif
    for( unsigned int ix = 0; ix < V; ix ++ ) {
        // int const z[4] = {
        //   ( g_lexic2coords[iz][0] + g_proc_coords[0] * T  - gsx[0] + T_global  ) % T_global,
        //   ( g_lexic2coords[iz][1] + g_proc_coords[1] * LX - gsx[1] + LX_global ) % LX_global,
        //   ( g_lexic2coords[iz][2] + g_proc_coords[2] * LY - gsx[2] + LY_global ) % LY_global,
        //   ( g_lexic2coords[iz][3] + g_proc_coords[3] * LZ - gsx[3] + LZ_global ) % LZ_global };
        unsigned int const iix = _GSI( ix );
        _co_pl_eq_fv_dag_ti_fv(&p2, xi+iix, phi+iix);
    }
    #ifdef HAVE_OPENMP
    omp_set_lock(&writelock);
    #endif

    paccum.re += p2.re;
    paccum.im += p2.im;

    #ifdef HAVE_OPENMP
    omp_unset_lock(&writelock);
    }  /* end of parallel region */
    omp_destroy_lock(&writelock);
    #endif

    /* fprintf(stdout, "# [spinor_scalar_product_co_cut] %d local: %e %e\n", g_cart_id, paccum.re, paccum.im); */

    #ifdef HAVE_MPI
    complex pall;
    pall.re=0.; pall.im=0.;
    if ( MPI_Allreduce(&paccum, &pall, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid) != MPI_SUCCESS ) {
        if ( g_cart_id == 0 ) fprintf ( stderr, "[] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
        w->re = sqrt( -1. );
        w->im = sqrt( -1. );
    } else {
        w->re = pall.re;
        w->im = pall.im;
    }
    #else
    w->re = paccum.re;
    w->im = paccum.im;
    #endif
  }  /* end of spinor_scalar_product_co_cut */

}
