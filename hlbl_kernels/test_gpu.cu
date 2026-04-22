#include "kernels.cuh"
#include <mpi.h>
#include "global.h"
#include "cuda_lattice_test.cuh"

int main(int argc, char **argv) {
    // set up MPI cartesian
    MPI_Init(&argc, &argv);
    int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
    int const proc_dim[4] = {NPROCT, NPROCX, NPROCY, NPROCZ};
    const int period[4] = {0,0,0,0};
    MPI_Cart_create(MPI_COMM_WORLD, 4, proc_dim, period, 0, &g_cart_grid);
    MPI_Comm_rank(g_cart_grid, &g_cart_id);
    MPI_Cart_coords(g_cart_grid, g_cart_id, 4, g_proc_coords);

    int device_id=0;
    char* slurm_localid_str = getenv("SLURM_LOCALID");
    if (slurm_localid_str != NULL) {
        device_id = atoi(slurm_localid_str); 
    } else {
        device_id = g_cart_id % 4;
    }

    cudaSetDevice(device_id); 

    // --- Optional: Print for Debugging ---
    int device_id_check;
    cudaGetDevice(&device_id_check);
    printf("MPI Rank %d running on CUDA Device %d\n", g_cart_id, device_id_check);

    printf ("MPI cartesian dimension %dx%dx%dx%d\n", g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);

    int const VOL = T * LX * LY * LZ;
    double *fwd_y = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    double *fwd_src = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    
    srand(1234);
    for (int i=0; i<24 * _GSI(VOL); i++) {
        fwd_y[i] = rand()*2./RAND_MAX - 1.;
        fwd_src[i] = rand()*2./RAND_MAX - 1.;
    }
    double g_dzu[6][4][12][24];
    double g_dzsu[6][4][12][24];
    for (int k=0; k<6; k++)
    for (int nu=0; nu<4; nu++)
    for (int ia=0; ia<12; ia++)
    for (int ib=0; ib<24; ib++){
        g_dzu[k][nu][ia][ib] = rand() * 2. / RAND_MAX - 1;
        g_dzsu[k][nu][ia][ib] = rand() * 2. / RAND_MAX - 1;
    }

    //rearrange fwd_y to match the kernel expectation
    /* double *fwd_y_tmp = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    for (int ifl=0; ifl<2; ifl++)
        for (int ia=0; ia<12; ia++)
            for (int x=0; x<VOL; x++)
                for (int ib=0; ib<24; ib++)
                    fwd_y_tmp[ifl * 12 * 24 * VOL + x * 12 * 24 + ia * 24 + ib] = fwd_y[ifl * 12 * 24 * VOL + ia * 24 + x * 24 + ib];
 */
    const int n_y = 2;
    const int gsw[4] = {1,1,1,1};
    int *gycoords = (int *)malloc(sizeof(int) * 4 * n_y);
    for (int i=0; i<n_y; i++){
        gycoords[4*i +0] = (i+2)%T_global;
        gycoords[4*i +1] = (i+3)%LX_global;
        gycoords[4*i +2] = (i+4)%LY_global;
        gycoords[4*i +3] = (i+5)%LZ_global;
    }
    
    // set unit
    double const mmuon = 105.6583745 /* MeV */  / 197.3269804 /* MeV fm */;
    double const alat[2] = { 0.1, 0.00013 };  /* fm */
    //double const xunit[2] = { mmuon * alat[0], mmuon * alat[1] };
    double const xunit[2] = { .8, 0.1 };
    
    //allocate P1, P23 
    double *P1; // = (double *)malloc(sizeof(double) * 4 * 4 * 4 * T_global);
    double *P23; // = (double *)malloc(sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    cudaHostAlloc((void**)&P1, 64 * T_global * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&P23, 64 * n_y * kernel_n * kernel_n_geom * sizeof(double), cudaHostAllocDefault);
    struct QED_kernel_temps kqed_t;
    initialise(&kqed_t);
    //compute_2p2_gpu(fwd_y, P1, P23, 0, gsw, gycoords, n_y, xunit, kqed_t, VOL, g_proc_coords, g_cart_grid, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    //record_pi_cuda(fwd_y, VOL, 0, T_global, LX_global, LY_global, LZ_global);
    double *Pi = (double *) malloc(sizeof(double) * 16 * VOL);
    srand(1234);
    for (int i=0; i<16 * VOL; i++) Pi[i] = rand()*2./RAND_MAX - 1.;
    record_p23_cuda(Pi, n_y, gsw, gycoords, xunit, kqed_t, VOL, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    //record_p1_cuda(Pi, 0, gsw, VOL, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    //record_2p2_cuda(fwd_y, P1, P23, 0, gsw, gycoords, n_y, xunit, kqed_t, VOL, g_proc_coords, g_cart_grid, T_global, LX_global, LY_global, LZ_global, T, LX, LY, LZ);

    double **spinor_work;
    //compute_2p2_pieces(fwd_y, P1, P23, gsw, 0, 0, n_y, gycoords, xunit, spinor_work, kqed_t,VOL, 1);
    /*if (g_proc_coords[0]==0 && g_proc_coords[1]==0 && g_proc_coords[2]==0 && g_proc_coords[3]==0) {

        FILE *file23;
        for (int i=0; i< n_y * kernel_n * kernel_n_geom * 4 * 4 *4; i++) {
            file23 = fopen("2p2_p23_tej.dat", "a");
            fprintf(file23, "%.10e\n", P23[i]);
            fclose(file23);
        }
        FILE *file1;
        for (int i=0; i< 64 * T_global; i++) {
            file1 = fopen("2p2_p1_tej.dat", "a");
            fprintf(file1, "%.10e\n", P1[i]);
            fclose(file1);
        }
    } */

    int const y[4] = {1,2,3,4};
    int const gsx[4] = {0,0,0,0};
    double kernel_sum[3] = {0};
    compute_4pt_gpu(fwd_src, fwd_y, g_dzu[0][0][0], g_dzsu[0][0][0], gsx, 0, xunit, y, kernel_sum, kqed_t, VOL, g_proc_coords, g_cart_grid, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    
    //compute_4pt_contraction(fwd_src, fwd_y, g_dzu, g_dzsu, gsx, 0, xunit, y, kernel_sum, kqed_t, VOL);

    for (int i=0; i<3; i++) {
        printf("kernel_sum[%d] = %.16e\n", i, kernel_sum[i]);
    }
    cudaFreeHost(P1);
    cudaFreeHost(P23);
    free(gycoords);
    free(Pi);
    free(fwd_y);
    free(fwd_src);

    MPI_Finalize();
    return 0;
}
