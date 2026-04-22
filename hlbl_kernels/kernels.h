#ifndef _KERNELS_H_
#define _KERNELS_H_
extern "C" {
    #include "KQED.h"
}

# include "../table_init_d.h"
# include <mpi.h>

/* CPU 2P2 BREAKDOWN, the ones noted with _0 are the original functions */
void compute_pi_0(double *fwd_y, double * Pi, int iflavor, double ** spinor_work, unsigned VOLUME);

void compute_pi(double *fwd_y, double * Pi, int iflavor, unsigned VOLUME);

void integrate_p1_0(double * pimn, double *P1, int iflavor, int const * gsw, unsigned VOLUME, int const g_proc_coords[4], 
    unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

void integrate_p1(double const *Pi, double *P1, int iflavor,  int const * gsw, unsigned VOLUME, int const g_proc_coords[4], 
    unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

template <int kernel_n, int kernel_n_geom>
void compute_p23_0(double *pimn, double (*P23)[kernel_n*kernel_n_geom][4][4][4], const int*gsw, int n_y, const int *gycoords, 
    const double xunit[2], QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], 
    unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

void compute_p23(double const *pi, double *P23, const int*gsw, int n_y, const int *gycoords, const double xunit[2], QED_kernel_temps kqed_t, 
    unsigned VOLUME, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

void compute_2p2_cpu(double *fwd_y, double *P1, double *P23, const int* gsw, int iflavor, int n_y, const int * gycoords,
     const double xunit[2], QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid, 
     unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

/* CPU CHECKS */
void check_Pi(size_t vol);
void check_integral(size_t vol, int w0, int w1, int w2, int w3, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);
void check_p23(unsigned vol, const int* gsw, int n_y, const int *gycoords, const double xunit[2], int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

/* CUDA CHECKS */
void check_Pi_cuda(size_t const vol);
void check_P1_cuda(int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);
void check_P23_cuda(int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);



/* CPU 4PT */
void compute_4pt_0(const double * fwd_src, const double * fwd_y, double const g_dzu[6][4][12][24], double const g_dzsu[6][4][12][24], const int* gsx, 
    int iflavor, const double xunit[2], const int yv[4], double* kernel_sum, QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid, 
     unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

void compute_4pt(const double * fwd_src, const double * fwd_y, double const g_dzu[6][4][12][24], double const g_dzsu[6][4][12][24], const int* gsx, 
    int iflavor, const double xunit[2], const int yv[4], double* kernel_sum, QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid, 
     unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

void check_compute_4pt(size_t const VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid, unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

#endif /* _KERNELS_H_ */
