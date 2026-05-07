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
}
