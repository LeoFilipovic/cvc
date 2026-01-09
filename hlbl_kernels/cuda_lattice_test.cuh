#include "../cuda_lattice.h"
#include "global.h"

inline void compute_2p2_pieces(
    const double * fwd_y, double * P1, double * P23x,
    const int* gsw, int iflavor, int io_proc, int n_y, const int * gycoords,
    const double xunit[2], double ** spinor_work, QED_kernel_temps kqed_t,
    unsigned VOLUME, int Nconf) {

  double* d_P1 = NULL;
  double* d_P23x = NULL;
  const int Lmax = T_global;
  const size_t n_P1 = 4 * 4 * 4 * Lmax;
  const size_t n_P23x = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
  const size_t sizeof_P1 = n_P1 * sizeof(double);
  const size_t sizeof_P23x = n_P23x * sizeof(double);
  checkCudaErrors(cudaMalloc((void**)&d_P1, sizeof_P1));
  checkCudaErrors(cudaMalloc((void**)&d_P23x, sizeof_P23x));
  checkCudaErrors(cudaMemset(d_P1, 0, sizeof_P1));
  checkCudaErrors(cudaMemset(d_P23x, 0, sizeof_P23x));
  
  Coord d_proc_coords {
    .t = g_proc_coords[0],
    .x = g_proc_coords[1],
    .y = g_proc_coords[2],
    .z = g_proc_coords[3]
  };
  Geom local_geom { .T = T, .LX = LX, .LY = LY, .LZ = LZ };
  Geom global_geom { .T = T_global, .LX = LX_global, .LY = LY_global, .LZ = LZ_global };
  Coord d_gsw = { .t = gsw[0], .x = gsw[1], .y = gsw[2], .z = gsw[3] };
  Pair d_xunit = { .a = xunit[0], .b = xunit[1] };
  Coord* gycoords_structs = (Coord*)malloc(n_y*sizeof(Coord));
  for ( int iy = 0; iy < n_y; iy++ )
  {
    gycoords_structs[iy].t = gycoords[4*iy + 0];
    gycoords_structs[iy].x = gycoords[4*iy + 1];
    gycoords_structs[iy].y = gycoords[4*iy + 2];
    gycoords_structs[iy].z = gycoords[4*iy + 3];
  }
  Coord* d_gycoords = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_gycoords, n_y*sizeof(Coord)));
  checkCudaErrors(cudaMemcpy(
      (void*)d_gycoords, (const void*)gycoords_structs, n_y*sizeof(Coord), cudaMemcpyHostToDevice));
  free(gycoords_structs);


  cu_2p2_pieces(
      d_P1, d_P23x, fwd_y, iflavor, d_proc_coords, d_gsw, n_y, d_gycoords,
      d_xunit, kqed_t, global_geom, local_geom);

  double* local_P1 = (double*)malloc(sizeof_P1);
  double* local_P23x = (double*)malloc(sizeof_P23x);
  double* all_P23x = (double *)malloc(sizeof_P23x);
  if ( local_P1 == NULL || local_P23x == NULL || all_P23x == NULL )
  {
    fprintf ( stderr, "Error alloc local_P1,23x or all_P23x\n" );
    exit ( 57 );
  }
  
  checkCudaErrors(cudaMemcpy(
      (void*)local_P1, (const void*)d_P1, sizeof_P1, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(
      (void*)local_P23x, (const void*)d_P23x, sizeof_P23x, cudaMemcpyDeviceToHost));

  // TODO: just MPI_Reduce?
  if ( MPI_Allreduce(local_P1, P1, n_P1, MPI_DOUBLE, MPI_SUM, g_cart_grid)
       != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
  }
  if ( MPI_Allreduce(local_P23x, all_P23x, n_P23x, MPI_DOUBLE, MPI_SUM, g_cart_grid)
       != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
  }

  // interleave data into output array
  memcpy((void *)P23x, all_P23x, sizeof_P23x);
  /*for ( int yi = 0; yi < n_y; yi++ )
  {
    for ( int ikernel = 0; ikernel < kernel_n; ikernel++ )
    {
      for ( int igeom = 0; igeom < kernel_n_geom; igeom++ )
      {
        memcpy(
            (void*)P23x[yi][kernel_n_geom*ikernel + igeom][iflavor][0][0],
            (void*)all_P23x[yi][kernel_n_geom*ikernel + igeom][0][0], sizeof(double)*4*4*4);
      }
    }
  }*/

  free(local_P1);
  free(local_P23x);
  //fini_5level_dtable ( &all_P23x );
  free(all_P23x);
  checkCudaErrors(cudaFree(d_P1));
  checkCudaErrors(cudaFree(d_P23x));
  checkCudaErrors(cudaFree(d_gycoords));

  if ( g_cart_id == 0 )
  {
    fprintf ( stdout, "[hlbl_mII_invert_contract] Finished 2+2 pieces for n_y = %d other y points\n", n_y );
  }
}