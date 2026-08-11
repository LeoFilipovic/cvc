#ifndef _INTEGRATION_BINS_H
#define _INTEGRATION_BINS_H


namespace cvc{
    int get_bin_0y(int const xv[4], int const xv_mi_yv[4], const int* Rcut2_bins, unsigned const Rcut_n);
    int get_bin_0(int const zv[4], const int * Zcut2_bins, int const Zcut_n);
}

#endif