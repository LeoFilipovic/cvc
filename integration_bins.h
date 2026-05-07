#ifndef _INTEGRATION_BINS_H
#define _INTEGRATION_BINS_H

namespace cvc {

/***********************************************************
* Calculate in which Rcut_bin (x,y) lies
***********************************************************/
int get_Rcut_bin(int const xv[4], int const xv_mi_yv[4], const int * Rcut2_bins, int const Rcut_n);

}
#endif
