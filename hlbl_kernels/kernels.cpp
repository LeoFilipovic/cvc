

#include <omp.h>
#include <iostream>
#include <stdio.h> 
#include <stdlib.h>
#include "kernels.h"

#define kernel_n 3 // L0, L3, M2
#define kernel_n_geom 3 // P2_0, P2_1, P3

const int idx_comb[6][2] = {
  {0,1},
  {0,2},
  {0,3},
  {1,2},
  {1,3},
  {2,3} };

typedef void (*QED_kernel_LX_ptr)( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] );
static void QED_kernel_L0P4( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] )
{
  QED_Mkernel_L2(0.4, xv, yv, t, kerv);
}
static QED_kernel_LX_ptr KQED_LX[kernel_n] = {
  QED_kernel_L0,
  QED_kernel_L3,
  QED_kernel_L0P4,
};

inline int get_Lmax(unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
  int Lmax = 0;
  if ( T_global >= Lmax ) Lmax = T_global;
  if ( LX_global >= Lmax ) Lmax = LX_global;
  if ( LY_global >= Lmax ) Lmax = LY_global;
  if ( LZ_global >= Lmax ) Lmax = LZ_global;
  return Lmax;
}

inline void site_map_zerohalf (int xv[4], int const x[4], unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
  xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
  xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
  xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
  xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );

  return;
}

inline int prop_idx(int iflavor, int ia, unsigned int ix, int ib, unsigned VOLUME) {
  return iflavor * 12 * 24 * VOLUME + ia * VOLUME * 24 + ix *24 + ib;
}

//using namespace cvc;
/* computation of Pi[mu][nu] */
inline void compute_pi_0(double * fwd_y, double * Pi, int iflavor, double ** spinor_work, unsigned VOLUME) 
{
  double *** pimn = (double ***)malloc(sizeof(double **) *4);
  for (int i=0; i<4; i++){
    pimn[i] = (double **)malloc(sizeof(double *) *4);
    for (int j=0; j<4; j++){
      pimn[i][j] = (double *)calloc(VOLUME, sizeof(double));
    }
  }
  /* memset((void*)pimn[0][0], 0, sizeof(double)*4*4*VOLUME); */
  for (int i=0; i<4; i++)
  for (int j=0; j<4; j++)
  for (size_t k=0; k<VOLUME; k++){
    pimn[i][j][k] = 0.0;
  }
  
  for ( int nu = 0; nu < 4; nu++ )
  {
    for ( int mu = 0; mu < 4; mu++ )
    {
      for ( int ia = 0; ia < 12; ia++ )
      {
#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++ )
        {
          const double * _u = fwd_y + prop_idx(iflavor, ia, ix, 0, VOLUME);
          double * _t = spinor_work[0] + 24 * ix; 
          _fv_eq_gamma_ti_fv ( _t, mu, _u );
          _fv_ti_eq_g5 ( _t );
          double * _s = spinor_work[1] + 24 * ix;
          for ( int ib = 0; ib < 12; ib++ )
          {
            const double * _d = fwd_y + prop_idx(1 - iflavor, ib, ix, 0, VOLUME);
            complex w;
            _co_eq_fv_dag_ti_fv ( &w, _d, _t );
            _s[2*ib]   = w.re;
            _s[2*ib+1] = w.im;
          }
          _fv_ti_eq_g5 ( _s );
          _fv_eq_gamma_ti_fv ( _t, nu, _s );
          // real part
          pimn[mu][nu][ix] += _t[2*ia];
        }
      }
    }
  }

  /* copy to p1 */
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int ix=0; ix<VOLUME; ix++) {
    Pi[mu*4*VOLUME + nu*VOLUME + ix] = pimn[mu][nu][ix];
  }

  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++){
      free(pimn[i][j]);
    }
    free(pimn[i]);
  }
  free(pimn);
}

/* performance improvement version 1: rearrange data structure of pi[x][mu][nu] */
inline void compute_pi(double *fwd_y, double * pi, int iflavor, unsigned VOLUME) 
{
  /* loop over position volume */
  #pragma omp parallel for
  for (int ix = 0; ix < VOLUME; ix++) {
    /* load 12 x 12 (x2 complex) d and u from fwd_y*/
    double u[12][24];
    double d[12][24];
    #pragma omp unroll
    for (int ia = 0; ia < 12; ia++)
    for (int ib = 0; ib < 24; ib++) {
      u[ia][ib] = fwd_y[prop_idx(iflavor, ia, ix, ib, VOLUME)];
    }
    #pragma omp unroll
    for (int ia = 0; ia < 12; ia++)
    for (int ib = 0; ib < 24; ib++){
      d[ia][ib] = fwd_y[prop_idx(1 - iflavor, ia, ix, ib, VOLUME)];
    }

    /* loop over mu and nu */
    for (int mu=0; mu<4; mu++)
    for (int nu=0; nu<4; nu++) {
      double gu[12][24];
      double dot_prod[12][24];

      /* apply gammas: gu = g_5 g_mu u */
      for (int ia=0; ia<12; ia++) {
        _fv_eq_gamma_ti_fv(gu[ia], mu, u[ia]);
        _fv_ti_eq_g5(gu[ia]);
      }

      /* compute <d, u>, dot_prod = <d, gmu> */
      for (int ia=0; ia<12; ia++ ) 
      for (int ib=0; ib<12; ib++ ) {
        complex w;
        _co_eq_fv_dag_ti_fv(&w, d[ib], gu[ia]);
        dot_prod[ia][2*ib] = w.re;
        dot_prod[ia][2*ib + 1] = w.im;
      }

      /* apply gammas: gu = g_nu g_5 dot_prod */
      for (int ia=0; ia<12; ia++) {
        _fv_ti_eq_g5(dot_prod[ia]);
        _fv_eq_gamma_ti_fv(gu[ia], nu, dot_prod[ia]);
      }

      /* take trace over ia: pi[x][mu][nu] = Tr[gu] */
      double trace = 0.0;
      for (int ia=0; ia<12; ia++) {trace += gu[ia][2 * ia]; } // only real part
      pi[ix*16+mu*4+nu] = trace;
    }
  }

  //allreduce(pi, 4*4*VOLUME);
}

/* Integration of Pi[mu][nu] over z */
inline void integrate_p1_0(double * pimn, double *P1, int iflavor, int const * gsw, unsigned VOLUME, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) 
{
  const int Lmax = get_Lmax(T_global, LX_global, LY_global, LZ_global);
  const int n_P1 = 4 * 4 * 4 * Lmax;
  //double **** local_P1 = init_4level_dtable ( 4, 4, 4, Lmax );
  double **** local_P1 = (double ****)malloc(sizeof(double ***) *4);
  for (int i=0; i<4; i++){
    local_P1[i] = (double ***)malloc(sizeof(double **) *4);
    for (int j=0; j<4; j++){
      local_P1[i][j] = (double **)malloc(sizeof(double *) *4);
      for (int k=0; k<4; k++){
        local_P1[i][j][k] = (double *)calloc(Lmax, sizeof(double));
      }
    }
  }
  if ( local_P1 == NULL )
  {
    fprintf ( stderr, "Error alloc local_P1\n" );
    exit ( 57 );
  }
  //memset((void*)local_P1[0][0][0], 0, sizeof(double)*n_P1);

  for ( int sigma = 0; sigma < 4; sigma++ )
  {
    for ( int nu = 0; nu < 4; nu++ )
    {
      // TODO: Parallelize over non-summed coordinate?
      for ( unsigned int iz = 0; iz < VOLUME; iz++ )
      {
        /* int const z[4] = {
          ( g_lexic2coords[iz][0] + g_proc_coords[0] * T  - gsw[0] + T_global  ) % T_global,
          ( g_lexic2coords[iz][1] + g_proc_coords[1] * LX - gsw[1] + LX_global ) % LX_global,
          ( g_lexic2coords[iz][2] + g_proc_coords[2] * LY - gsw[2] + LY_global ) % LY_global,
          ( g_lexic2coords[iz][3] + g_proc_coords[3] * LZ - gsw[3] + LZ_global ) % LZ_global }; 
        */
        
          /* find global z[4] */
        const int z[4] = {(iz / (LX * LY * LZ) + g_proc_coords[0] * T - gsw[0] + T_global) % T_global,
          (iz / (LY * LZ) % LX + g_proc_coords[1] * LX - gsw[1] + LX_global) % LX_global,
          ((iz / LZ) % LY + g_proc_coords[2] * LY - gsw[2] + LY_global) % LY_global,
          (iz % LZ + g_proc_coords[3] * LZ - gsw[3] + LZ_global) % LZ_global};

        for ( int rho = 0; rho < 4; rho++ )
        {
          local_P1[rho][sigma][nu][z[rho]] += pimn[sigma*4*VOLUME + nu*VOLUME + iz];
        }
      }
    }
  }

  for (int rho=0; rho<4; rho++)
  for (int sigma=0; sigma<4; sigma++)
  for (int nu=0; nu<4; nu++)
  for (int i=0; i<Lmax; i++) {
    P1[rho*16*Lmax + sigma*4*Lmax + nu*Lmax + i] = local_P1[rho][sigma][nu][i];
  }

  for (int rho=0; rho<4; rho++){
    for (int sigma=0; sigma<4; sigma++){
      for (int nu=0; nu<4; nu++){
        free(local_P1[rho][sigma][nu]);
      }
      free(local_P1[rho][sigma]);
    }
    free(local_P1[rho]);
  }
}

/* rerarrange the summation order to z, rho, sigma, nu
   note that input pi[x][mu][nu] is different P1[rho][sigma][nu][z] is unchanged */
inline void integrate_p1(double const *Pi, double *P1, int iflavor,  int const * gsw, unsigned VOLUME, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) 
{
  const int local_dim[4] = {static_cast<int>(T), static_cast<int>(LX), static_cast<int>(LY), static_cast<int>(LZ)};
  const int global_dim[4] = {static_cast<int>(T_global), static_cast<int>(LX_global), static_cast<int>(LY_global), static_cast<int>(LZ_global)};

  const int Lmax = T_global; // T will be the largest dimension
  const int n_P1 = 4 * 4 * 4 * Lmax;
  /* P1 set zero */
  #pragma omp parallel for
  for (int i=0; i<n_P1; i++) {
    P1[i] = 0.0;
  }

  /* for (int iz = 0; iz < VOLUME; iz++ ) {
    const int z[4] = {(iz / (LX * LY * LZ) + g_proc_coords[0] * T - gsw[0] + T_global) % T_global,
      (iz / (LY * LZ) % LX + g_proc_coords[1] * LX - gsw[1] + LX_global) % LX_global,
      ((iz / LZ) % LY + g_proc_coords[2] * LY - gsw[2] + LY_global) % LY_global,
      (iz % LZ + g_proc_coords[3] * LZ - gsw[3] + LZ_global) % LZ_global};
      
      for (int rho=0; rho<4; rho++)
      for (int sigma=0; sigma<4; sigma++)
      for (int nu=0; nu<4; nu++) {
          P1[rho*Lmax*16 + sigma*Lmax*4 + nu*Lmax + z[rho]] += Pi[iz*16 + sigma*4 + nu]; 
      }
  } */

  #pragma omp parallel for collapse(3)
  for (int rho=0; rho<4; rho++)
  for (int sigma=0; sigma<4; sigma++)
  for (int nu=0; nu<4; nu++)
  for (int zr=0; zr<local_dim[rho]; zr++) {
    double sum = 0.0;
    // the index of the three non-rho directions
    int dir[3];
    int cnt = 0;
    for (int d=0; d<4; d++){
        if (d!=rho){
            dir[cnt] = d;
            cnt++;
        }
    }
    // construct local z[4], rho direction fixed
    int z[4];
    z[rho]=zr;
    // loop over the rest VOLUME/local_dim[rho] points in 3D
    for (int iz = 0; iz < VOLUME/local_dim[rho]; iz++) {
        // the 3 other local z directions
        z[dir[0]] = iz / (local_dim[dir[1]] * local_dim[dir[2]]);
        z[dir[1]] = iz / local_dim[dir[2]] % local_dim[dir[1]];
        z[dir[2]] = iz % local_dim[dir[2]];

        // now machine address z_lex
        const int z_lex = z[0] * LX * LY * LZ + z[1] * LY * LZ + z[2] * LZ + z[3];

        // accumulate sum
        sum += Pi[z_lex*16 + sigma*4 + nu];
    }

    // write to P1
    // global z[rho] - w[rho]
    const unsigned z_w = (zr + local_dim[rho] * g_proc_coords[rho]  + global_dim[rho] - gsw[rho]) % global_dim[rho];
    
    #pragma omp atomic
    P1[rho*16*Lmax + sigma*4*Lmax + nu*Lmax + z_w] += sum;
  }
}

/* Computation of P2 and P3 */
/***********************************************************
 * P2_{rsn}(y)
 *   = sum_x (L_[r,s];mnl(x,y) + L_[r,s];nml(y,x)) Pi_{ml}(x)
 * P3_{rsn}(y)
 *   = sum_x (L_[r,s];mln(x+y,y) Pi_{ml}(x)
 ***********************************************************/
template<typename QED_kernel_temps>
inline void compute_p23_0(double *pimn, double (*P23)[kernel_n*kernel_n_geom][4][4][4], const int*gsw, int n_y, const int *gycoords, const double xunit[2],
QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global){
  for ( int yi = 0; yi < n_y; yi++ )
  {
    double kerv1[6][4][4][4] KQED_ALIGN ;
    double kerv2[6][4][4][4] KQED_ALIGN ;
    double kerv3[6][4][4][4] KQED_ALIGN ;
    double kerv4[6][4][4][4] KQED_ALIGN ;
    
    // For P2: y = (gsy - gsw)
    // For P3: y' = (gsw - gsy)
    // We define y = (gsy - gsw) and use -y as input for P3.
    int const * gsy = &gycoords[4*yi];
    int const y[4] = {
      ( gsy[0] - gsw[0] + T_global ) % T_global,
      ( gsy[1] - gsw[1] + LX_global ) % LX_global,
      ( gsy[2] - gsw[2] + LY_global ) % LY_global,
      ( gsy[3] - gsw[3] + LZ_global ) % LZ_global
    };
    int yv[4];
    site_map_zerohalf ( yv, y, T_global, LX_global, LY_global, LZ_global);
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
      /* int const x[4] = {
        ( g_lexic2coords[ix][0] + g_proc_coords[0] * T  - gsw[0] + T_global  ) % T_global,
        ( g_lexic2coords[ix][1] + g_proc_coords[1] * LX - gsw[1] + LX_global ) % LX_global,
        ( g_lexic2coords[ix][2] + g_proc_coords[2] * LY - gsw[2] + LY_global ) % LY_global,
        ( g_lexic2coords[ix][3] + g_proc_coords[3] * LZ - gsw[3] + LZ_global ) % LZ_global }; */
      int const x[4] = {(ix / (LX * LY * LZ) + g_proc_coords[0] * T- gsw[0] + T_global) % T_global,
      (ix / (LY * LZ) % LX + g_proc_coords[1] * LX - gsw[1] + LX_global) % LX_global,
      ((ix / LZ) % LY + g_proc_coords[2] * LY - gsw[2] + LY_global) % LY_global,
      (ix % LZ  + g_proc_coords[3] * LZ - gsw[3] + LZ_global) % LZ_global};

      int xv[4];
      site_map_zerohalf ( xv, x, T_global, LX_global, LY_global, LZ_global);

      double const xm[4] = {
        xv[0] * xunit[0],
        xv[1] * xunit[0],
        xv[2] * xunit[0],
        xv[3] * xunit[0] };

      double const xm_minus[4] = {
        -xv[0] * xunit[0],
        -xv[1] * xunit[0],
        -xv[2] * xunit[0],
        -xv[3] * xunit[0] };

      double const ym[4] = {
        yv[0] * xunit[0],
        yv[1] * xunit[0],
        yv[2] * xunit[0],
        yv[3] * xunit[0] };

      double const ym_minus[4] = {
        -yv[0] * xunit[0],
        -yv[1] * xunit[0],
        -yv[2] * xunit[0],
        -yv[3] * xunit[0] };

      double const xm_mi_ym[4] = {
        xm[0] - ym[0],
        xm[1] - ym[1],
        xm[2] - ym[2],
        xm[3] - ym[3] };
      double const ym_mi_xm[4] = {
        ym[0] - xm[0],
        ym[1] - xm[1],
        ym[2] - xm[2],
        ym[3] - xm[3] };

      for ( int ikernel = 0; ikernel < kernel_n; ikernel++ )
      {
        KQED_LX[ikernel]( xm, ym,             kqed_t, kerv1 );
        KQED_LX[ikernel]( ym, xm,             kqed_t, kerv2 );
        KQED_LX[ikernel]( xm_mi_ym, ym_minus, kqed_t, kerv3 );
        //KQED_LX[ikernel]( ym_mi_xm, xm_minus, kqed_t, kerv4 );
        for( int k = 0; k < 6; k++ )
        {
          int const rho   = idx_comb[k][0];
          int const sigma = idx_comb[k][1];
          for ( int nu = 0; nu < 4; nu++ )
          {
            /* #if kernel_n_geom != 5
            #error "Number of QED kernel geometries does not match implementation"
            #endif */
            for ( int mu = 0; mu < 4; mu++ )
            {
              for ( int lambda = 0; lambda < 4; lambda++ )
              {
                // P2_0
                P23[yi][ikernel*kernel_n_geom + 0][rho][sigma][nu] +=
                    kerv1[k][mu][nu][lambda] * pimn[mu*VOLUME*4 + lambda*VOLUME + ix];
                // P2_1
                P23[yi][ikernel*kernel_n_geom + 1][rho][sigma][nu] +=
                    kerv2[k][nu][mu][lambda] * pimn[mu*VOLUME*4 + lambda*VOLUME + ix];
                // P3
                P23[yi][ikernel*kernel_n_geom + 2][rho][sigma][nu] +=
                    kerv3[k][mu][lambda][nu] * pimn[mu*VOLUME*4 + lambda*VOLUME + ix];
                // P4_0
                /* P23[yi][ikernel*kernel_n_geom + 3][rho][sigma][nu] +=
                    kerv4[k][nu][lambda][mu] * pimn[mu*T_global*4 + lambda*T_global + ix]; */
              }
            }
            // P4_1
            /* P23[yi][ikernel*kernel_n_geom + 4][rho][sigma][nu] =
                (yv[rho]-xv[rho]) * P23[yi][ikernel*kernel_n_geom + 3][rho][sigma][nu];
            P23[yi][ikernel*kernel_n_geom + 4][sigma][rho][nu] =
                (yv[sigma]-xv[sigma]) * (-P23[yi][ikernel*kernel_n_geom + 3][rho][sigma][nu]); */
          }
        }
      }
    }
  }
  //allreduce(&P23[0][0][0][0][0], n_y*kernel_n*kernel_n_geom*64);
}

/* optimised compute_p23: loop rearrangement */
inline void compute_p23(double const *pi, double *P23, const int *gsw, int n_y, const int *gycoords, const double xunit[2],
QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global){
  /* #if kernel_n_geom != 5
  #error "Number of QED kernel geometries does not match implementation"
  #endif */
  /* clear P23 */
  int const n_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
  #pragma omp parallel for
  for (int i=0; i<n_p23; i++){
    P23[i] = 0.;
  }

  #pragma omp parallel for
  for ( int yi = 0; yi < n_y; yi++ ){
    double kerv1[6][4][4][4] KQED_ALIGN ;
    double kerv2[6][4][4][4] KQED_ALIGN ;
    double kerv3[6][4][4][4] KQED_ALIGN ;
    //double kerv4[6][4][4][4] KQED_ALIGN ;
    // For P2: y = (gsy - gsw)
    // For P3: y' = (gsw - gsy)
    // We define y = (gsy - gsw) and use -y as input for P3.
    int const * gsy = &gycoords[4*yi];
    int const y[4] = {
      ( gsy[0] - gsw[0] + T_global ) % T_global,
      ( gsy[1] - gsw[1] + LX_global ) % LX_global,
      ( gsy[2] - gsw[2] + LY_global ) % LY_global,
      ( gsy[3] - gsw[3] + LZ_global ) % LZ_global
    };
    int yv[4];
    site_map_zerohalf ( yv, y, T_global, LX_global, LY_global, LZ_global );

    double const ym[4] = {
      yv[0] * xunit[0],
      yv[1] * xunit[0],
      yv[2] * xunit[0],
      yv[3] * xunit[0] };

    double const ym_minus[4] = {
      -yv[0] * xunit[0],
      -yv[1] * xunit[0],
      -yv[2] * xunit[0],
      -yv[3] * xunit[0] };

    for ( int ikernel = 0; ikernel < kernel_n; ikernel++ ){
      /* a different local copy of P2/3 for each kernel */
      double local_p2_0[64]={0};
      double local_p2_1[64]={0};
      double local_p3[64]={0};
      for ( unsigned int ix = 0; ix < VOLUME; ix++ ){
        /* int const x[4] = {
          ( g_lexic2coords[ix][0] + g_proc_coords[0] * T  - gsw[0] + T_global  ) % T_global,
          ( g_lexic2coords[ix][1] + g_proc_coords[1] * LX - gsw[1] + LX_global ) % LX_global,
          ( g_lexic2coords[ix][2] + g_proc_coords[2] * LY - gsw[2] + LY_global ) % LY_global,
          ( g_lexic2coords[ix][3] + g_proc_coords[3] * LZ - gsw[3] + LZ_global ) % LZ_global }; */
        int const x[4] = {(ix / (LX * LY * LZ)  + g_proc_coords[0] * T - gsw[0] + T_global) % T_global,
        (ix / (LY * LZ) % LX + g_proc_coords[1] * LX - gsw[1] + LX_global) % LX_global,
        ((ix / LZ) % LY + g_proc_coords[2] * LY - gsw[2] + LY_global) % LY_global,
        (ix % LZ + g_proc_coords[3] * LZ - gsw[3] + LZ_global) % LZ_global};

        int xv[4];
        site_map_zerohalf ( xv, x, T_global, LX_global, LY_global, LZ_global);

        const double pix[16] = {pi[ix*16 +0], pi[ix*16 +1], pi[ix*16 +2], pi[ix*16 +3],
                          pi[ix*16 +4], pi[ix*16 +5], pi[ix*16 +6], pi[ix*16 +7],
                          pi[ix*16 +8], pi[ix*16 +9], pi[ix*16 +10],pi[ix*16 +11],
                          pi[ix*16 +12],pi[ix*16 +13],pi[ix*16 +14],pi[ix*16 +15]};
        /* double *pix = pi + ix*16; */
        double const xm[4] = {
          xv[0] * xunit[0],
          xv[1] * xunit[0],
          xv[2] * xunit[0],
          xv[3] * xunit[0] };

        double const xm_mi_ym[4] = {
          xm[0] - ym[0],
          xm[1] - ym[1],
          xm[2] - ym[2],
          xm[3] - ym[3] };

      
        KQED_LX[ikernel]( xm, ym,             kqed_t, kerv1 );
        KQED_LX[ikernel]( ym, xm,             kqed_t, kerv2 );
        KQED_LX[ikernel]( xm_mi_ym, ym_minus, kqed_t, kerv3 );
        
        /* P2_0 unroll k (too much register pressure)*/
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++)
        for (int lambda=0; lambda<4; lambda++){
          // k=0: {0,1}
          local_p2_0[0*16 + 1*4 + nu] += kerv1[0][mu][nu][lambda] * pix[mu*4 +lambda];
          local_p2_1[0*16 + 1*4 + nu] += kerv2[0][nu][mu][lambda] * pix[mu*4+lambda];
          local_p3[0*16 + 1*4 + nu] += kerv3[0][mu][lambda][nu] * pix[mu*4+lambda];
        }
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++)
        for (int lambda=0; lambda<4; lambda++){
          // k=1: {0,2}
          local_p2_0[0*16 + 2*4 + nu] += kerv1[1][mu][nu][lambda] * pix[mu*4 +lambda];
          local_p2_1[0*16 + 2*4 + nu] += kerv2[1][nu][mu][lambda] * pix[mu*4+lambda];
          local_p3[0*16 + 2*4 + nu] += kerv3[1][mu][lambda][nu] * pix[mu*4+lambda];
        }
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++)
        for (int lambda=0; lambda<4; lambda++){
          // k=2: {0,3}
          local_p2_0[0*16 + 3*4 + nu] += kerv1[2][mu][nu][lambda] * pix[mu*4 +lambda];
          local_p2_1[0*16 + 3*4 + nu] += kerv2[2][nu][mu][lambda] * pix[mu*4+lambda];
          local_p3[0*16 + 3*4 + nu] += kerv3[2][mu][lambda][nu] * pix[mu*4+lambda];
        }
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++)
        for (int lambda=0; lambda<4; lambda++){
          // k=3: {1,2}
          local_p2_0[1*16 + 2*4 + nu] += kerv1[3][mu][nu][lambda] * pix[mu*4 +lambda];
          local_p2_1[1*16 + 2*4 + nu] += kerv2[3][nu][mu][lambda] * pix[mu*4+lambda];
          local_p3[1*16 + 2*4 + nu] += kerv3[3][mu][lambda][nu] * pix[mu*4+lambda];
        }
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++)
        for (int lambda=0; lambda<4; lambda++){
          // k=4: {1,3}
          local_p2_0[1*16 + 3*4 + nu] += kerv1[4][mu][nu][lambda] * pix[mu*4 +lambda];
          local_p2_1[1*16 + 3*4 + nu] += kerv2[4][nu][mu][lambda] * pix[mu*4+lambda];
          local_p3[1*16 + 3*4 + nu] += kerv3[4][mu][lambda][nu] * pix[mu*4+lambda];
        }
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++)
        for (int lambda=0; lambda<4; lambda++){
          // k=5: {2,3}
          local_p2_0[2*16 + 3*4 + nu] += kerv1[5][mu][nu][lambda] * pix[mu*4 +lambda];
          local_p2_1[2*16 + 3*4 + nu] += kerv2[5][nu][mu][lambda] * pix[mu*4+lambda];
          local_p3[2*16 + 3*4 + nu] += kerv3[5][mu][lambda][nu] * pix[mu*4+lambda];
        }
      }
          
      /* Accumulate to global P23 */
      double *P23y = P23 + yi * (kernel_n * 64) + ikernel * 64;
      int const offset = n_y * kernel_n * 64;
      #pragma unroll
      for (int i=0; i<64; i++) P23y[i] = local_p2_0[i];
      #pragma unroll
      for (int i=0; i<64; i++) P23y[offset + i] = local_p2_1[i];
      #pragma unroll
      for (int i=0; i<64; i++) P23y[2 * offset + i] = local_p3[i];
    }
  }
}

void compute_2p2_cpu(double* fwd_y, double *P1, double *P23, int const* gsw, int iflavor, int n_y, const int * gycoords,
    const double xunit[2], QED_kernel_temps kqed_t, unsigned VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid, 
    unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
  double * pi = (double *)calloc(16 * VOLUME, sizeof(double));
  int const n_p1 = 4 * 4 * 4 * get_Lmax(T_global, LX_global, LY_global, LZ_global);
  int const n_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;

  compute_pi(fwd_y, pi, iflavor, VOLUME);

  MPI_Request reqs[2];

  integrate_p1(pi, P1, iflavor, gsw, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
  MPI_Iallreduce(MPI_IN_PLACE, P1, n_p1, MPI_DOUBLE, MPI_SUM, g_cart_grid, &reqs[0]);

  compute_p23(pi, P23, gsw, n_y, gycoords, xunit, kqed_t, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
  MPI_Iallreduce(MPI_IN_PLACE, P23, n_p23, MPI_DOUBLE, MPI_SUM, g_cart_grid, &reqs[1]);

  free(pi);

  MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
}

/* check correctness openmp and mpi*/
void check_Pi(size_t vol) {
  double *p1_0 = (double *)calloc(16 * vol, sizeof(double));
  double *p1_1 = (double *)calloc(16 * vol, sizeof(double));
  //double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
  double **spinor_work = (double **) malloc(sizeof(double *) * 2);
  for (int i=0; i<2; i++) spinor_work[i] = (double *)malloc(sizeof(double)*24 * vol);
  //double *** fwd_y = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );
  double *fwd_y = (double *) malloc(sizeof(double *)*2 * 12 * 24 * vol);

  /* fill fwd_y with test data ... */
  for (int iflavor = 0; iflavor < 2; iflavor++) {
    srand(iflavor + 1234);
    for (int ia = 0; ia < 12; ia++) {
      for (unsigned int ix = 0; ix < vol; ix++) {
        for (int comp = 0; comp < 24; comp++) {
          fwd_y[prop_idx(iflavor, ia, ix, comp, vol)] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
        }
      }
    }
  } 

  compute_pi_0(fwd_y, p1_0, 0, spinor_work, vol);

  compute_pi(fwd_y, p1_1, 0, vol);

  int flag = 0;
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int x=0; x<vol; x++){
    const double p0 = p1_0[mu*4*vol + nu*vol + x];
    const double p1 = p1_1[x*16 + mu*4 +nu];
    const double diff = p0 - p1;
    if (diff * diff > 1e-26) {
      flag=1;
      printf("Pi difference at [%d][%d][%d], %f VS %f\n.", mu, nu, x, p0, p1);
    }
  }
  if (flag) printf("Pi correctnenss FAILED.\n");
  else printf("Pi correctness PASSED.\n");

  free(p1_0);
  free(p1_1);
  for (int i=0; i<2; i++) free(spinor_work[i]);
  free(spinor_work);

  free(fwd_y);
}
void check_integral(size_t vol, int w0, int w1, int w2, int w3, int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
  double *pi = (double *)calloc(16 * vol, sizeof(double));
  double *pi_rearrange = (double *)calloc(16 * vol, sizeof(double));
  double *P1 = (double *)calloc(4 * 4 * 4 * get_Lmax(T_global, LX_global, LY_global, LZ_global), sizeof(double));
  for (int i=0; i<16*vol; i++){
    pi[i] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
  }
  const int gsw[4] = {w0,w1,w2,w3};
  integrate_p1_0(pi, P1, 0, gsw, vol, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
  double *P1_check = (double *)calloc(4 * 4 * 4 * get_Lmax(T_global, LX_global, LY_global, LZ_global), sizeof(double));
  /* Rearrange pi for new format i.e. pi[mu][nu][x] to pi[x][mu][nu] */
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int x=0; x<vol; x++){
    pi_rearrange[x*16 + mu*4 + nu] = pi[mu*4*vol + nu*vol + x];
  }

  integrate_p1(pi_rearrange, P1_check, 0, gsw, vol, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
  int flag = 0;
  const int n_P1 = 4 * 4 * 4 * get_Lmax(T_global, LX_global, LY_global, LZ_global);
  for (int i=0; i<n_P1; i++){
    const double diff = P1[i] - P1_check[i];
    if (diff * diff > 1e-26) {
      flag=1;
      printf("Integral difference at %d: %f VS %f\n.", i, P1[i], P1_check[i]);
    }
    //printf("Integral difference at %d: %f VS %f\n.", i, P1[i], P1_check[i]);
  }
  if (flag) printf("Integral correctness FAILED.\n");
  else printf("Integral correctness PASSED.\n");
  free(pi);
  free(pi_rearrange);
  free(P1);
  free(P1_check);
}

void check_p23(unsigned vol, const int* gsw, int n_y, const int *gycoords, const double xunit[2], int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
  double *pi = (double *)calloc(16 * vol, sizeof(double));
  double *pi_rearrange = (double *)calloc(16 * vol, sizeof(double));
  int const n_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
  double (*P23)[kernel_n*kernel_n_geom][4][4][4] = (double (*)[kernel_n*kernel_n_geom][4][4][4]) calloc(n_y, sizeof(*P23));
  double *P23_new = (double *) malloc(sizeof(*P23_new) * n_p23);
  for (int i=0; i<16*vol; i++){
    pi[i] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
  }
  struct QED_kernel_temps kqed_t ;
  initialise(&kqed_t);
  compute_p23_0(pi, P23, gsw, n_y, gycoords, xunit, kqed_t, vol, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
  /* Rearrange pi for new format i.e. pi[mu][nu][x] to pi[x][mu][nu] */
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int x=0; x<vol; x++){
    pi_rearrange[x*16 + mu*4 + nu] = pi[mu*4*vol + nu*vol + x];
  }
  struct QED_kernel_temps kqed_t_new ;
  initialise(&kqed_t_new);
  compute_p23(pi_rearrange, P23_new, gsw, n_y, gycoords, xunit, kqed_t_new, vol, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);

  // Add correctness check here if needed
  int flag = 0;
  for (int x=0; x<n_y; x++)
  for (int ikernel=0; ikernel<kernel_n; ikernel++)
  for (int g=0; g<3; g++)
  for (int rho=0; rho<4; rho++)
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++){
    const double diff = P23[x][ikernel * kernel_n_geom + g][rho][mu][nu] - P23_new[x * (kernel_n * kernel_n_geom * 4 *4 *4) + ikernel * (kernel_n_geom *4 *4 *4) + g * (4*4*4) + rho * (4*4) + mu *4 + nu];
    if (diff * diff > 1e-26) {
      flag=1;
      printf("P23 difference at [%d][%d][%d][%d][%d][%d]: %f VS %f diff=%e\n.",
         x, ikernel, g, rho, mu, nu, P23[x][ikernel * kernel_n_geom + g][rho][mu][nu], P23_new[x * (kernel_n * kernel_n_geom * 4 *4 *4) + ikernel * (kernel_n_geom *4 *4 *4) + g * (4*4*4) + rho * (4*4) + mu *4 + nu], diff);
    }
    /* printf("Integral difference at [%d][%d][%d][%d][%d][%d]: %f VS %f diff=%e\n.",
         x, ikernel, g, rho, mu, nu, P23[x][ikernel * kernel_n_geom + g][rho][mu][nu], P23_new[x][ikernel * kernel_n_geom + g][rho][mu][nu], diff);
 */  }
  if (flag) printf("P23 correctness FAILED.\n");
  else printf("P23 correctness PASSED.\n");

  free(pi);
  free(pi_rearrange);
  free(P23);
  free(P23_new);
}


/* check correctness cuda */
void check_Pi_cuda(size_t const vol) {
  double *pi = (double *)calloc(16 * vol, sizeof(double));

  double *fwd_y = (double *) malloc(sizeof(double) * 2 * 12 * 24 * vol);

  /* fill fwd_y with test data ... */
  srand(1234);
  for (int iflavor = 0; iflavor < 2; iflavor++) {
    for (int ia = 0; ia < 12; ia++) {
      for (unsigned int ix = 0; ix < vol; ix++) {
        for (int comp = 0; comp < 24; comp++) {
          fwd_y[prop_idx(iflavor, ia, ix, comp, vol)] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
        }
      }
    }
  } 

  compute_pi(fwd_y, pi, 0, vol);

  // read from pi_cuda, data separated by "\n"
  double *pi_cuda = (double *)malloc(16 * vol * sizeof(double));
  FILE *fp = fopen("pi_cuda.dat", "r");
  for (int i=0; i<16*vol; i++) {
    fscanf(fp, "%lf", &pi_cuda[i]);
  }
  fclose(fp);

  // compare pi and pi_cuda
  int flag = 0;
  for (int i=0; i<16*vol; i++){
    const double p0 = pi[i];;
    const double p1 = pi_cuda[i];
    const double diff = p0 - p1;
    if (diff * diff > 1e-18) {
      flag=1;
      printf("Pi difference at [%d], %.10f VS %.10f\n.", i, p0, p1);
      break;
    }
  }
  if (flag) printf("Pi correctnenss FAILED.\n");
  else printf("Pi correctness PASSED.\n");
  free(pi);
  free(pi_cuda);
  free(fwd_y);
}

void check_P1_cuda(int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
  int const vol = LX * LY * LZ * T;
  double*p1 = (double *)malloc(64 * T_global * sizeof(double));
  double *Pi = (double *) malloc(sizeof(double) * 16 * vol);
  srand(1234);
  for (int i=0; i< 16 * vol; i++) Pi[i] = rand()*2./RAND_MAX - 1.;

  const int gsw[4] = {1,1,1,1};
  integrate_p1(Pi, p1, 0 , gsw, vol, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);

  // read from pi_cuda, data separated by "\n"
  double *p1_cuda = (double *)malloc(64 * T_global * sizeof(double));
  FILE *fp = fopen("p1_cuda.dat", "r");
  for (int i=0; i < 64 * T_global; i++) {
    fscanf(fp, "%lf", &p1_cuda[i]);
  }
  fclose(fp);

  // compare p1 and p1_cuda
  int flag = 0;
  for (int i=0; i< 64 * T_global; i++){
    const double p0 = p1[i];;
    const double p1 = p1_cuda[i];
    const double diff = p0 - p1;
    if (diff * diff > 1e-18) {
      flag=1;
      printf("P1 difference at [%d], %.10f VS %.10f\n", i, p0, p1);
    }
  }
  if (flag) printf("P1 correctnenss FAILED.\n");
  else printf("P1 correctness PASSED.\n");
  free(Pi);
  free(p1);
  free(p1_cuda);
}

void check_P23_cuda(int const g_proc_coords[4], unsigned T, unsigned LX, unsigned LY, unsigned LZ,
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
  int const vol = LX_global * LY_global * LZ_global * T_global;
  const int n_y = 2;
  const int gsw[4] = {1,1,1,1};
  int *gycoords = (int *)malloc(sizeof(int) * 4 * n_y);
  for (int i=0; i<n_y; i++){
    gycoords[4*i +0] = (i+2)%T_global;
    gycoords[4*i +1] = (i+3)%LX_global;
    gycoords[4*i +2] = (i+4)%LY_global;
    gycoords[4*i +3] = (i+5)%LZ_global;
  }
  double xunit[2] = {0.1,0.2};

  double *Pi = (double *) malloc(sizeof(double) * 16 * vol);
  srand(1234);
  for (int i=0; i< 16 * vol; i++) Pi[i] = rand()*2./RAND_MAX - 1.;

  int const n_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
  double *P23 = (double *) malloc(sizeof(double) * n_p23);

  struct QED_kernel_temps kqed_t_new ;
  initialise(&kqed_t_new);

  compute_p23(Pi, P23, gsw, n_y, gycoords, xunit, kqed_t_new, vol, g_proc_coords, T_global, LX_global, LY_global, LZ_global, T_global, LX_global, LY_global, LZ_global);

  // read from p23_cuda.dat
  double *P23_cuda = (double *) malloc(sizeof(double) * n_p23);
  FILE *fp = fopen("p23_cuda.dat", "r");
  for (int i=0; i<n_p23; i++) {
    fscanf(fp, "%lf", &P23_cuda[i]);
  }
  fclose(fp);

  // compare P23 and P23_cuda
  int flag = 0;
  for (int i=0; i<n_p23; i++) {
    const double p0 = P23[i];
    const double p1 = P23_cuda[i];
    const double diff = p0 - p1;
    if (diff * diff > 1e-18) {
      flag=1;
      printf("P23 difference at [%d], %.10f VS %.10f\n.", i, p0, p1);
    }
  }
  if (flag) printf("P23 correctnenss FAILED.\n");
  else printf("P23 correctness PASSED.\n");
  free(Pi);
  free(P23);
  free(P23_cuda);
} 
