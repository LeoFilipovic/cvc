#ifndef _SITE_MAPPING_H
#define _SITE_MAPPING_H

namespace cvc {
    int get_Lmax();
    void site_map (int xv[4], int const x[4]);
    void site_map_zerohalf (int xv[4], int const x[4]);
}

#endif