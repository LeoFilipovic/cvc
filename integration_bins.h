#ifndef _INTEGRATION_BINS_H
#define _INTEGRATION_BINS_H

namespace cvc {

/***********************************************************
* Calculate in which Rcut_bin (x,y) lies
***********************************************************/
int get_Rcut_bin(int const xv[4], int const xv_mi_yv[4], const int * Rcut2_bins, int const Rcut_n);
int get_Zcut_bin(int const zv[4], const int * Zcut2_bins, int const Zcut_n);
void spinor_scalar_product_co_cut ( complex * const w, double * const xi, double * const phi, unsigned int const V); //, const int * Xcut2_bins, int const Xcut_n);

}
#endif
