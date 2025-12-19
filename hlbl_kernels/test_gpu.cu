#include "kernels.cuh"
#include <mpi.h>
#include "global.h"

int main(int argc, char **argv) {
    // allocate fwd_y on the host


   
    //check Pi correctness
    /* double *fwd_y = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    srand(1234);
    for (int i=0; i<24 * _GSI(VOL); i++) fwd_y[i] = rand()*2./RAND_MAX - 1.;
    record_pi_cuda(fwd_y, VOL, 0);
    free(fwd_y);
 */
    //check P1 correctness
    /* double *Pi = (double *) malloc(sizeof(double) * 16 * VOL);
    srand(1234);
    for (int i=0; i<16 * VOL; i++) Pi[i] = rand()*2./RAND_MAX - 1.;
    const int gsw[4] = {1, 1, 1, 1};
    record_p1_cuda(Pi, 0, gsw, VOL);
    free(Pi);    */

    //check P23 correctness
    /* const int n_y = 10;
    const int gsw[4] = {1,1,1,1};
    int *gycoords = (int *)malloc(sizeof(int) * 4 * n_y);
    for (int i=0; i<n_y; i++){
        gycoords[4*i +0] = (i+2)%T_global;
        gycoords[4*i +1] = (i+3)%LX_global;
        gycoords[4*i +2] = (i+4)%LY_global;
        gycoords[4*i +3] = (i+5)%LZ_global;
    }
    double xunit[2] = {0.1,0.2};
    double *Pi = (double *) malloc(sizeof(double) * 16 * VOL);
    srand(1234);
    for (int i=0; i<16 * VOL; i++) Pi[i] = rand()*2./RAND_MAX - 1.;
    record_p23_cuda(Pi, n_y, kernel_n, kernel_n_geom, gsw, gycoords, xunit, VOL);
    free(gycoords);
    free(Pi); */


    // set up MPI cartesian
    MPI_Init(&argc, &argv);
    int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int const proc_dim[4] = {NPROCT, NPROCX, NPROCY, NPROCZ};
    const int period[4] = {1,1,1,1};
    MPI_Cart_create(MPI_COMM_WORLD, 4, proc_dim, period, true, &g_cart_grid);
    MPI_Comm_rank(g_cart_grid, &g_cart_id);
    MPI_Cart_coords(g_cart_grid, g_cart_id, 4, g_proc_coords);
    int device_id=0;
    char* slurm_localid_str = getenv("SLURM_LOCALID");
    if (slurm_localid_str != NULL) {
        // Map based on the local index 0, 1, 2, 3
        device_id = atoi(slurm_localid_str); 
    } else {
        // Fallback or debug (less reliable on a cluster)
        device_id = rank % 4;
    }
    
    // Set the device
    cudaSetDevice(device_id); 

    // --- Optional: Print for Debugging ---
    printf("MPI Rank %d running on CUDA Device %d\n", rank, device_id);

    //cudaSetDevice(rank%4); // bind GPU to rank?
    //if (g_cart_id == 0) 
        printf ("MPI cartesian dimension %dx%dx%dx%d\n", g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);

    int const VOL = T * LX * LY * LZ;
    double *fwd_y = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    srand(1234);
    for (int i=0; i<24 * _GSI(VOL); i++) fwd_y[i] = rand()*2./RAND_MAX - 1.;

    //rearrange fwd_y to match the kernel expectation
    /* double *fwd_y_tmp = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    for (int ifl=0; ifl<2; ifl++)
        for (int ia=0; ia<12; ia++)
            for (int x=0; x<VOL; x++)
                for (int ib=0; ib<24; ib++)
                    fwd_y_tmp[ifl * 12 * 24 * VOL + x * 12 * 24 + ia * 24 + ib] = fwd_y[ifl * 12 * 24 * VOL + ia * 24 + x * 24 + ib];
 */
    const int n_y = 20;
    const int gsw[4] = {1,1,1,1};
    int *gycoords = (int *)malloc(sizeof(int) * 4 * n_y);
    for (int i=0; i<n_y; i++){
        gycoords[4*i +0] = (i+2)%T_global;
        gycoords[4*i +1] = (i+3)%LX_global;
        gycoords[4*i +2] = (i+4)%LY_global;
        gycoords[4*i +3] = (i+5)%LZ_global;
    }
    const double xunit[2] = {0.1,0.2};
    
    //allocate P1, P23 
    double *P1 = (double *)malloc(sizeof(double) * 4 * 4 * 4 * T_global);
    double *P23 = (double *)malloc(sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    struct QED_kernel_temps kqed_t;
    initialise(&kqed_t);
    //compute_2p2_gpu(fwd_y, P1, P23, 0, gsw, gycoords, n_y, xunit, kqed_t, VOL, g_proc_coords, g_cart_grid,  T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    //record_pi_cuda(fwd_y, VOL, 0, T_global, LX_global, LY_global, LZ_global);
    double *Pi = (double *) malloc(sizeof(double) * 16 * VOL);
    srand(1234);
    for (int i=0; i<16 * VOL; i++) Pi[i] = rand()*2./RAND_MAX - 1.;
    //record_p23_cuda(Pi, n_y, gsw, gycoords, xunit, kqed_t, VOL, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    //record_p1_cuda(Pi, 0, gsw, VOL, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    record_2p2_cuda(fwd_y, P1, P23, 0, gsw, gycoords, n_y, xunit, kqed_t, VOL, g_proc_coords, g_cart_grid, T_global, LX_global, LY_global, LZ_global, T, LX, LY, LZ);

    MPI_Finalize();
    return 0;
}