#include <stdint.h> 
#include <iostream>
#include <alpaka/alpaka.hpp>
#define Real_t double
#define Real_tp double*
namespace lulesh_port_kernels{


using Index_t= std::int32_t;
using Int_t=std::int32_t;
//this needs adjustment when using float
ALPAKA_FN_ACC inline auto FMAX(Real_t  arg1,Real_t  arg2)-> Real_t {return FMAX(arg1,arg2) ; }
ALPAKA_FN_ACC auto AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3) -> Real_t
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1); 
   Real_t temp = (fx * gx + fy * gy + fz * gz);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      temp * temp;
   return area ;
}

ALPAKA_FN_ACC auto CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d ) -> void
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];
 
  Real_t tmp1 = (xvel[0]-xvel[6]);
  Real_t tmp2 = (xvel[1]-xvel[7]);
  Real_t tmp3 = (xvel[2]-xvel[4]);
  Real_t tmp4 = (xvel[3]-xvel[5]);


  d[0] = inv_detJ *  ( pfx[0] * tmp1
                     + pfx[1] * tmp2
                     + pfx[2] * tmp3
                     + pfx[3] * tmp4);

  dxddy  = inv_detJ * ( pfy[0] * tmp1
                      + pfy[1] * tmp2
                      + pfy[2] * tmp3
                      + pfy[3] * tmp4);

  dxddz  = inv_detJ * ( pfz[0] * tmp1
                      + pfz[1] * tmp2
                      + pfz[2] * tmp3
                      + pfz[3] * tmp4);

  tmp1 = (yvel[0]-yvel[6]);
  tmp2 = (yvel[1]-yvel[7]);
  tmp3 = (yvel[2]-yvel[4]);
  tmp4 = (yvel[3]-yvel[5]);

  d[1] = inv_detJ *  ( pfy[0] * tmp1
                     + pfy[1] * tmp2
                     + pfy[2] * tmp3
                     + pfy[3] * tmp4);

  dyddx  = inv_detJ * ( pfx[0] * tmp1 
                      + pfx[1] * tmp2
                      + pfx[2] * tmp3
                      + pfx[3] * tmp4);

  dyddz  = inv_detJ * ( pfz[0] * tmp1
                      + pfz[1] * tmp2
                      + pfz[2] * tmp3
                      + pfz[3] * tmp4);

  tmp1 = (zvel[0]-zvel[6]);
  tmp2 = (zvel[1]-zvel[7]);
  tmp3 = (zvel[2]-zvel[4]);
  tmp4 = (zvel[3]-zvel[5]);

  d[2] = inv_detJ * ( pfz[0] * tmp1
                     + pfz[1] * tmp2
                     + pfz[2] * tmp3
                     + pfz[3] * tmp4);

  dzddx  = inv_detJ * ( pfx[0] * tmp1 
                      + pfx[1] * tmp2
                      + pfx[2] * tmp3
                      + pfx[3] * tmp4);

  dzddy  = inv_detJ * ( pfy[0] * tmp1
                      + pfy[1] * tmp2
                      + pfy[2] * tmp3
                      + pfy[3] * tmp4);

  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}
ALPAKA_FN_ACC auto CalcMonoGradient(Real_t *x, Real_t *y, Real_t *z,
                      Real_t *xv, Real_t *yv, Real_t *zv,
                      Real_t vol, 
                      Real_t *delx_zeta, 
                      Real_t *delv_zeta,
                      Real_t *delx_xi,
                      Real_t *delv_xi,
                      Real_t *delx_eta,
                      Real_t *delv_eta) -> void
{

   #define SUM4(a,b,c,d) (a + b + c + d)
   const Real_t ptiny = Real_t(1.e-36) ;
   Real_t ax,ay,az ;
   Real_t dxv,dyv,dzv ;

   Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

   Real_t dxj = Real_t(-0.25)*(SUM4(x[0],x[1],x[5],x[4]) - SUM4(x[3],x[2],x[6],x[7])) ;
   Real_t dyj = Real_t(-0.25)*(SUM4(y[0],y[1],y[5],y[4]) - SUM4(y[3],y[2],y[6],y[7])) ;
   Real_t dzj = Real_t(-0.25)*(SUM4(z[0],z[1],z[5],z[4]) - SUM4(z[3],z[2],z[6],z[7])) ;

   Real_t dxi = Real_t( 0.25)*(SUM4(x[1],x[2],x[6],x[5]) - SUM4(x[0],x[3],x[7],x[4])) ;
   Real_t dyi = Real_t( 0.25)*(SUM4(y[1],y[2],y[6],y[5]) - SUM4(y[0],y[3],y[7],y[4])) ;
   Real_t dzi = Real_t( 0.25)*(SUM4(z[1],z[2],z[6],z[5]) - SUM4(z[0],z[3],z[7],z[4])) ;

   Real_t dxk = Real_t( 0.25)*(SUM4(x[4],x[5],x[6],x[7]) - SUM4(x[0],x[1],x[2],x[3])) ;
   Real_t dyk = Real_t( 0.25)*(SUM4(y[4],y[5],y[6],y[7]) - SUM4(y[0],y[1],y[2],y[3])) ;
   Real_t dzk = Real_t( 0.25)*(SUM4(z[4],z[5],z[6],z[7]) - SUM4(z[0],z[1],z[2],z[3])) ;

   /* find delvk and delxk ( i cross j ) */
   ax = dyi*dzj - dzi*dyj ;
   ay = dzi*dxj - dxi*dzj ;
   az = dxi*dyj - dyi*dxj ;

   *delx_zeta = vol / sqrt(ax*ax + ay*ay + az*az + ptiny) ; 

   ax *= norm ;
   ay *= norm ;
   az *= norm ; 

   dxv = Real_t(0.25)*(SUM4(xv[4],xv[5],xv[6],xv[7]) - SUM4(xv[0],xv[1],xv[2],xv[3])) ;
   dyv = Real_t(0.25)*(SUM4(yv[4],yv[5],yv[6],yv[7]) - SUM4(yv[0],yv[1],yv[2],yv[3])) ;
   dzv = Real_t(0.25)*(SUM4(zv[4],zv[5],zv[6],zv[7]) - SUM4(zv[0],zv[1],zv[2],zv[3])) ; 

   *delv_zeta = ax*dxv + ay*dyv + az*dzv ;

   /* find delxi and delvi ( j cross k ) */

   ax = dyj*dzk - dzj*dyk ;
   ay = dzj*dxk - dxj*dzk ;
   az = dxj*dyk - dyj*dxk ;

   *delx_xi = vol / sqrt(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = Real_t(0.25)*(SUM4(xv[1],xv[2],xv[6],xv[5]) - SUM4(xv[0],xv[3],xv[7],xv[4])) ;
   dyv = Real_t(0.25)*(SUM4(yv[1],yv[2],yv[6],yv[5]) - SUM4(yv[0],yv[3],yv[7],yv[4])) ;
   dzv = Real_t(0.25)*(SUM4(zv[1],zv[2],zv[6],zv[5]) - SUM4(zv[0],zv[3],zv[7],zv[4])) ;

   *delv_xi = ax*dxv + ay*dyv + az*dzv ;

   /* find delxj and delvj ( k cross i ) */

   ax = dyk*dzi - dzk*dyi ;
   ay = dzk*dxi - dxk*dzi ;
   az = dxk*dyi - dyk*dxi ;

   *delx_eta = vol / sqrt(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = Real_t(-0.25)*(SUM4(xv[0],xv[1],xv[5],xv[4]) - SUM4(xv[3],xv[2],xv[6],xv[7])) ;
   dyv = Real_t(-0.25)*(SUM4(yv[0],yv[1],yv[5],yv[4]) - SUM4(yv[3],yv[2],yv[6],yv[7])) ;
   dzv = Real_t(-0.25)*(SUM4(zv[0],zv[1],zv[5],zv[4]) - SUM4(zv[3],zv[2],zv[6],zv[7])) ;

   *delv_eta = ax*dxv + ay*dyv + az*dzv ;
#undef SUM4
}

ALPAKA_FN_ACC auto CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)-> Real_t
{
   Real_t a, charLength = Real_t(0.0);

   a = lulesh_port_kernels::AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ; // 38
   charLength = lulesh_port_kernels::FMAX(a,charLength) ;

   a = lulesh_port_kernels::AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = lulesh_port_kernels::FMAX(a,charLength) ;

   a = lulesh_port_kernels::AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = lulesh_port_kernels::FMAX(a,charLength) ;

   a = lulesh_port_kernels::AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = lulesh_port_kernels::FMAX(a,charLength) ;

   a =lulesh_port_kernels::AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = lulesh_port_kernels::FMAX(a,charLength) ;

   a = lulesh_port_kernels::AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = lulesh_port_kernels::FMAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / sqrt(charLength);

   return charLength;
}
ALPAKA_FN_ACC auto CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 ) -> Real_t
{
   printf("h\n");
   Real_t twelveth = Real_t(1.0)/Real_t(12.0);
  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))


  // 11 + 3*14
  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;
  printf("%f, ",volume);

  return volume ;
}
ALPAKA_FN_ACC auto static inline SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3) -> void
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}
ALPAKA_FN_ACC static inline auto VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz) -> void
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);

   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}
ALPAKA_FN_ACC static inline auto CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
                              Real_t *hourgam1, Real_t *hourgam2, Real_t *hourgam3,
                              Real_t *hourgam4, Real_t *hourgam5, Real_t *hourgam6,
                              Real_t *hourgam7, Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz ) -> void
{
   Index_t i00=0;
   Index_t i01=1;
   Index_t i02=2;
   Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}
ALPAKA_FN_ACC static inline
auto CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8]) -> void 
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   lulesh_port_kernels::SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   lulesh_port_kernels::SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   lulesh_port_kernels::SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   lulesh_port_kernels::SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   lulesh_port_kernels::SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   lulesh_port_kernels::SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}
ALPAKA_FN_ACC static auto inline CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume ) -> void
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  /*

  b[0][4] = - cjxxi  -  cjxet  +  cjxze;
  b[0][5] = + cjxxi  -  cjxet  +  cjxze;
  b[0][6] = + cjxxi  +  cjxet  +  cjxze;
  b[0][7] = - cjxxi  +  cjxet  +  cjxze;

  */

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}
ALPAKA_FN_ACC auto inline CalcHourglassModes(const Real_t xn[8], const Real_t yn[8], const Real_t zn[8],
                        const Real_t dvdxn[8], const Real_t dvdyn[8], const Real_t dvdzn[8],
                        Real_t hourgam[8][4], Real_t volinv) -> void
{
    Real_t hourmodx, hourmody, hourmodz;

    hourmodx = xn[0] + xn[1] - xn[2] - xn[3] - xn[4] - xn[5] + xn[6] + xn[7];
    hourmody = yn[0] + yn[1] - yn[2] - yn[3] - yn[4] - yn[5] + yn[6] + yn[7];
    hourmodz = zn[0] + zn[1] - zn[2] - zn[3] - zn[4] - zn[5] + zn[6] + zn[7]; // 21
    hourgam[0][0] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][0] =  1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][0] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][0] = -1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][0] = -1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][0] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][0] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][0] =  1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz); // 60

    hourmodx = xn[0] - xn[1] - xn[2] + xn[3] - xn[4] + xn[5] + xn[6] - xn[7];
    hourmody = yn[0] - yn[1] - yn[2] + yn[3] - yn[4] + yn[5] + yn[6] - yn[7];
    hourmodz = zn[0] - zn[1] - zn[2] + zn[3] - zn[4] + zn[5] + zn[6] - zn[7];
    hourgam[0][1] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][1] = -1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][1] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][1] =  1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][1] = -1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][1] =  1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][1] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][1] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

    hourmodx = xn[0] - xn[1] + xn[2] - xn[3] + xn[4] - xn[5] + xn[6] - xn[7];
    hourmody = yn[0] - yn[1] + yn[2] - yn[3] + yn[4] - yn[5] + yn[6] - yn[7];
    hourmodz = zn[0] - zn[1] + zn[2] - zn[3] + zn[4] - zn[5] + zn[6] - zn[7];
    hourgam[0][2] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][2] = -1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][2] =  1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][2] = -1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][2] =  1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][2] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][2] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][2] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

    hourmodx = -xn[0] + xn[1] - xn[2] + xn[3] + xn[4] - xn[5] + xn[6] - xn[7];
    hourmody = -yn[0] + yn[1] - yn[2] + yn[3] + yn[4] - yn[5] + yn[6] - yn[7];
    hourmodz = -zn[0] + zn[1] - zn[2] + zn[3] + zn[4] - zn[5] + zn[6] - zn[7];
    hourgam[0][3] = -1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][3] =  1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][3] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][3] =  1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][3] =  1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][3] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][3] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][3] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

}
ALPAKA_FN_ACC auto CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8]) -> void
{
   lulesh_port_kernels::VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   lulesh_port_kernels::VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   lulesh_port_kernels::VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   lulesh_port_kernels::VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   lulesh_port_kernels::VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   lulesh_port_kernels::VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   lulesh_port_kernels::VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   lulesh_port_kernels::VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

class CalcPositionAndVelocityForNodes_kernel_class{


    int numNode;
    Real_t deltatime; 
    Real_t u_cut;
    Real_t* __restrict__ x;
    Real_t* __restrict__ y;
    Real_t* __restrict__ z;
    Real_t* __restrict__ xd;
    Real_t* __restrict__ yd;
    Real_t* __restrict__ zd;
    const Real_t* __restrict__ xdd;
    const Real_t* __restrict__ ydd;
    const Real_t* __restrict__ zdd;
    
    public:
    CalcPositionAndVelocityForNodes_kernel_class(
        int numNode, 
        const Real_t deltatime, 
        const Real_t u_cut,
        Real_t* __restrict__ x,  
        Real_t* __restrict__ y,  
        Real_t* __restrict__ z,
        Real_t* __restrict__ xd, 
        Real_t* __restrict__ yd, 
        Real_t* __restrict__ zd,
        const Real_t* __restrict__ xdd,
        const Real_t* __restrict__ ydd, 
        const Real_t* __restrict__ zdd
    ){
        this->numNode=numNode; 
        this->deltatime=deltatime; 
        this->u_cut=u_cut;
        this->x=x;  
        this->y=y;  
        this->z=z;
        this->xd=xd; 
        this->yd=yd; 
        this->zd=zd;
        this->xdd=xdd;
        this->ydd=ydd; 
        this->zdd=zdd;
    };
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;
        
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        Vec1 const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);
        int i=static_cast<int>(linearizedGlobalThreadIdx[0u]);
        
        if (i < numNode)
        {
            Real_t xdtmp, ydtmp, zdtmp, dt;
            dt = deltatime;
      
            xdtmp = xd[i] + xdd[i] * dt ;
            ydtmp = yd[i] + ydd[i] * dt ;
            zdtmp = zd[i] + zdd[i] * dt ;

            if( fabs(xdtmp) < u_cut ) xdtmp = 0.0;
            if( fabs(ydtmp) < u_cut ) ydtmp = 0.0;
            if( fabs(zdtmp) < u_cut ) zdtmp = 0.0;
 
            x[i] += xdtmp * dt;
            y[i] += ydtmp * dt;
            z[i] += zdtmp * dt;
      
            xd[i] = xdtmp; 
            yd[i] = ydtmp; 
            zd[i] = zdtmp; 
        }
    };
};

class ApplyAccelerationBoundaryConditionsForNodes_kernel_class{

    int numNodeBC;
    Real_t *xyzdd; 
    Index_t *symm;

    public:
    ApplyAccelerationBoundaryConditionsForNodes_kernel_class(
    int numNodeBC, 
    Real_t *xyzdd, 
    Index_t *symm
    ){
        this->numNodeBC=numNodeBC;
        this->xyzdd=xyzdd;
        this->symm=symm;
    };
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;
        
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        Vec1 const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);
        int tid=static_cast<int>(linearizedGlobalThreadIdx[0u]);//(alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0u]);
        
        if (tid < numNodeBC)
        {
            xyzdd[symm[tid]] = Real_t(0.0) ;
        }
    };
};


class CalcAccelerationForNodes_kernel_class{


    int numNode;
    Real_t *xdd;
    Real_t *ydd;
    Real_t *zdd;
    Real_t *fx;
    Real_t *fy;
    Real_t *fz;
    Real_t *nodalMass;
    
    public:
    CalcAccelerationForNodes_kernel_class(
    int numNode,
    Real_t *xdd,
    Real_t *ydd,
    Real_t *zdd,
    Real_t *fx,
    Real_t *fy,
    Real_t *fz,
    Real_t *nodalMass
    ){
         this->numNode=numNode;
         this->xdd=xdd;
         this->ydd=ydd;
         this->zdd=zdd;
         this->fx=fx;
         this->fy=fy;
         this->fz=fz;
         this->nodalMass=nodalMass;
    
    };
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        std::printf("[DEVICE] CalcAccelerationForNodes_kernel_class\n");
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;
        
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        int tid=static_cast<int>(alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0u]);
        std::printf("device\n");
        if (tid < numNode)
        {
            Real_t one_over_nMass = Real_t(1.)/nodalMass[tid];
            xdd[tid]=fx[tid]*one_over_nMass;
            ydd[tid]=fy[tid]*one_over_nMass;
            zdd[tid]=fz[tid]*one_over_nMass;
        }
        
    
    };



};

class AddNodeForcesFromElems_kernel_class{

    Index_t numNode;
    Index_t padded_numNode;
    const Int_t* nodeElemCount; 
    const Int_t* nodeElemStart; 
    const Index_t* nodeElemCornerList;
    const Real_t* fx_elem;
    const Real_t* fy_elem; 
    const Real_t* fz_elem;
    Real_t* fx_node;
    Real_t* fy_node; 
    Real_t* fz_node;
    Int_t num_threads;

    public:
    AddNodeForcesFromElems_kernel_class(
    Index_t numNode,
    Index_t padded_numNode,
    const Int_t* nodeElemCount, 
    const Int_t* nodeElemStart, 
    const Index_t* nodeElemCornerList,
    const Real_t* fx_elem, 
    const Real_t* fy_elem, 
    const Real_t* fz_elem,
    Real_t* fx_node, 
    Real_t* fy_node, 
    Real_t* fz_node,
    const Int_t num_threads
    ){
    this->numNode=numNode;
    this->padded_numNode=padded_numNode;
    this->nodeElemCount=nodeElemCount; 
    this->nodeElemStart=nodeElemStart; 
    this->nodeElemCornerList=nodeElemCornerList;
    this->fx_elem=fx_elem; 
    this->fy_elem=fy_elem; 
    this->fz_elem=fz_elem;
    this->fx_node=fx_node; 
    this->fy_node=fy_node; 
    this->fz_node=fz_node;
    this->num_threads=num_threads;
    };
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        //std::printf("[DEVICE] AddNodeForcesFromElems_kernel_class\n");
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;
        
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        Int_t tid=static_cast<unsigned int>(alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0u]);
    
        if (tid < num_threads)
        {
            Index_t g_i = tid;
            Int_t count=nodeElemCount[g_i];
            Int_t start=nodeElemStart[g_i];
            Real_t fx,fy,fz;
            fx=fy=fz=Real_t(0.0);

            for (int j=0;j<count;j++) 
            {
                Index_t pos=nodeElemCornerList[start+j]; // Uncoalesced access here
                fx += fx_elem[pos]; 
                fy += fy_elem[pos]; 
                fz += fz_elem[pos];
            }


            fx_node[g_i]=fx; 
            fy_node[g_i]=fy; 
            fz_node[g_i]=fz;
        }
    };    


};

class CalcVolumeForceForElems_kernel_class{

    Real_t* __restrict__ volo,*__restrict__ v,*__restrict__ p,*__restrict__ q;
    Real_t hourg;
    Index_t numElem; 
    Index_t padded_numElem; 
    Index_t * nodelist;
    Real_t * elemMass;
    Real_t *__restrict__ ss,*__restrict__ x,*__restrict__ y,*__restrict__ z,*__restrict__ xd,*__restrict__ yd,*__restrict__ zd,*__restrict__ fx_elem,*__restrict__ fy_elem,*__restrict__ fz_elem;

    Real_t coefficient;
    Index_t* __restrict__ bad_vol;
    Index_t num_threads;
    bool hour_gt_zero;
    public:
    CalcVolumeForceForElems_kernel_class(

    Real_t * __restrict__ volo, 
    Real_t * __restrict__ v,
    Real_t * __restrict__ p, 
    Real_t * __restrict__ q,
    Real_t hourg,
    Index_t numElem, 
    Index_t padded_numElem, 
    Index_t * __restrict__ nodelist,
    Real_t *   __restrict__ ss, 
    Real_t * __restrict__ elemMass,
    Real_t * __restrict__x,   Real_t * __restrict__y,  Real_t * __restrict__ z,
    Real_t * __restrict__xd,  Real_t * __restrict__yd,  Real_t * __restrict__zd,
    //TextureObj<Real_t> x,  TextureObj<Real_t> y,  TextureObj<Real_t> z,
    //TextureObj<Real_t> xd,  TextureObj<Real_t> yd,  TextureObj<Real_t> zd,
    //TextureObj<Real_t>* x,  TextureObj<Real_t>* y,  TextureObj<Real_t>* z,
    //TextureObj<Real_t>* xd,  TextureObj<Real_t>* yd,  TextureObj<Real_t>* zd,
    Real_t* fx_elem, 
    Real_t*  fy_elem, 
    Real_t* fz_elem,
    Index_t*  bad_vol,
    const Index_t num_threads,
    bool hour_gt_zero):volo(volo),ss(ss),x(x),y(y),z(z),xd(xd),yd(yd),zd(zd),
    fx_elem(fx_elem),fy_elem(fy_elem),fz_elem(fz_elem),coefficient(coefficient),bad_vol(bad_vol),num_threads(num_threads),hour_gt_zero(hour_gt_zero){
        this->elemMass=elemMass;


    };
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {

        Real_t xn[8],yn[8],zn[8];
        Real_t xdn[8],ydn[8],zdn[8];
        Real_t dvdxn[8],dvdyn[8],dvdzn[8];
        Real_t hgfx[8],hgfy[8],hgfz[8];
        Real_t hourgam[8][4];
  /*************************************************
  *     FUNCTION: Calculates the volume forces
  *************************************************/
      using Dim = alpaka::Dim<TAcc>;
      using Idx = alpaka::Idx<TAcc>;
      using Vec = alpaka::Vec<Dim, Idx>;
      using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;

      // In the most cases the parallel work distibution depends
      // on the current index of a thread and how many threads
      // exist overall. These information can be obtained by
      // getIdx() and getWorkDiv(). In this example these
      // values are obtained for a global scope.
      Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
      Index_t elem=static_cast<unsigned>(alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0u]);
      if (!elem < num_threads)return;
      Real_t volume = v[elem];
      Real_t det = volo[elem] * volume;

      // Check for bad volume
      if (volume < 0.) {
        *bad_vol = elem; 
      }

      Real_t ss1 = ss[elem];
      Real_t mass1 = elemMass[elem];
      Real_t sigxx = -p[elem] - q[elem];

      Index_t n[8];
      for (int i=0;i<8;i++) {
        n[i] = nodelist[elem+i*padded_numElem];
      }

      Real_t volinv = Real_t(1.0) / det;
      //#pragma unroll 
      //for (int i=0;i<8;i++) {
      //  xn[i] =x[n[i]];
      //  yn[i] =y[n[i]];
      //  zn[i] =z[n[i]];
      //}
        //vectorize later
      for (int i=0;i<8;i++)xn[i] =x[n[i]];

      for (int i=0;i<8;i++)yn[i] =y[n[i]];

      for (int i=0;i<8;i++)zn[i] =z[n[i]];


      Real_t volume13 = cbrt(det);
      Real_t coefficient2 = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      /*************************************************/
      /*    compute the volume derivatives             */
      /*************************************************/
      lulesh_port_kernels::CalcElemVolumeDerivative(dvdxn, dvdyn, dvdzn, xn, yn, zn); 

      /*************************************************/
      /*    compute the hourglass modes                */
      /*************************************************/
      lulesh_port_kernels::CalcHourglassModes(xn,yn,zn,dvdxn,dvdyn,dvdzn,hourgam,volinv);

      /*************************************************/
      /*    CalcStressForElems                         */
      /*************************************************/
      Real_t B[3][8];

      lulesh_port_kernels::CalcElemShapeFunctionDerivatives(xn, yn, zn, B, &det); 

      lulesh_port_kernels::CalcElemNodeNormals( B[0] , B[1], B[2], xn, yn, zn); 

      // Check for bad volume
      if (det < 0.) {
        *bad_vol = elem; 
      }

      for (int i=0;i<8;i++)
      {
        hgfx[i] = -( sigxx*B[0][i] );
        hgfy[i] = -( sigxx*B[1][i] );
        hgfz[i] = -( sigxx*B[2][i] );
      }

      if (this->hour_gt_zero)
      {
        /*************************************************/
        /*    CalcFBHourglassForceForElems               */
        /*************************************************/

  //      #pragma unroll 
  //      for (int i=0;i<8;i++) {
  //        xdn[i] =xd[n[i]];
  //        ydn[i] =yd[n[i]];
  //        zdn[i] =zd[n[i]];
  //      }

        for (int i=0;i<8;i++)
          xdn[i] =xd[n[i]];

        for (int i=0;i<8;i++)
          ydn[i] =yd[n[i]];
        for (int i=0;i<8;i++)
          zdn[i] =zd[n[i]];




        lulesh_port_kernels::CalcElemFBHourglassForce
        ( &xdn[0],&ydn[0],&zdn[0],
          hourgam[0],hourgam[1],hourgam[2],hourgam[3],
          hourgam[4],hourgam[5],hourgam[6],hourgam[7],
          coefficient2,
          &hgfx[0],&hgfy[0],&hgfz[0]
        );

      }
      for (int node=0;node<8;node++)
      {
        Index_t store_loc = elem+padded_numElem*node;
        fx_elem[store_loc]=hgfx[node]; 
        fy_elem[store_loc]=hgfy[node]; 
        fz_elem[store_loc]=hgfz[node];
      }
      // If elem < numElem
};//end alpaka function
};//end class
class CalcKinematicsAndMonotonicQGradient_kernel_class{
   
   Index_t numElem, padded_numElem;
   const Real_t dt;
   const Index_t* __restrict__ nodelist;
   const Real_t* __restrict__ volo, * __restrict__ v;

    const Real_t* __restrict__ x;
    const Real_t* __restrict__ y; 
    const Real_t* __restrict__ z;
    const Real_t* __restrict__ xd; 
    const Real_t* __restrict__ yd; 
    const Real_t* __restrict__ zd;
    Real_t* __restrict__ vnew;
    Real_t* __restrict__ delv;
    Real_t* __restrict__ arealg;
    Real_t* __restrict__ dxx;
    Real_t* __restrict__ dyy;
    Real_t* __restrict__ dzz;
    Real_t* __restrict__ vdov;
    Real_t* __restrict__ delx_zeta; 
    Real_t* __restrict__ delv_zeta;
    Real_t* __restrict__ delx_xi; 
    Real_t* __restrict__ delv_xi; 
    Real_t* __restrict__ delx_eta;
    Real_t* __restrict__ delv_eta;
    Index_t* __restrict__ bad_vol;
    const Index_t num_threads;
   public:
   CalcKinematicsAndMonotonicQGradient_kernel_class(
   Index_t numElem, Index_t padded_numElem, const Real_t dt,
    const Index_t* __restrict__ nodelist, const Real_t* __restrict__ volo, const Real_t* __restrict__ v,

    const Real_t* __restrict__ x, 
    const Real_t* __restrict__ y, 
    const Real_t* __restrict__ z,
    const Real_t* __restrict__ xd, 
    const Real_t* __restrict__ yd, 
    const Real_t* __restrict__ zd,
    Real_t* __restrict__ vnew,
    Real_t* __restrict__ delv,
    Real_t* __restrict__ arealg,
    Real_t* __restrict__ dxx,
    Real_t* __restrict__ dyy,
    Real_t* __restrict__ dzz,
    Real_t* __restrict__ vdov,
    Real_t* __restrict__ delx_zeta, 
    Real_t* __restrict__ delv_zeta,
    Real_t* __restrict__ delx_xi, 
    Real_t* __restrict__ delv_xi, 
    Real_t* __restrict__ delx_eta,
    Real_t* __restrict__ delv_eta,
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads
    ):numElem(numElem),padded_numElem(padded_numElem),dt(dt),
      nodelist(nodelist),volo(volo),v(v),x(x),y(y),z(z),xd(xd),yd(yd),zd(zd),vnew(vnew),delv(delv),arealg(arealg),dxx(dxx),dyy(dyy),vdov(vdov),delx_zeta(delx_zeta),
      delv_zeta(delv_zeta),delx_xi(delx_xi),delv_xi(delv_xi),delx_eta(delx_eta),delv_eta(delv_eta),bad_vol(bad_vol),num_threads(num_threads){};
   template<typename TAcc>
   ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
      Real_t B[3][8] ; /** shape function derivatives */
   Index_t nodes[8] ;
   Real_t x_local[8] ;
   Real_t y_local[8] ;
   Real_t z_local[8] ;
   Real_t xd_local[8] ;
   Real_t yd_local[8] ;
   Real_t zd_local[8] ;
   Real_t D[6];
   using Dim = alpaka::Dim<TAcc>;
   using Idx = alpaka::Idx<TAcc>;
   using Vec = alpaka::Vec<Dim, Idx>;
   using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;

   Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
   Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
   Index_t k=static_cast<unsigned>(alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0u]);

  if ( k < num_threads) {
    Real_t volume ;
    Real_t relativeVolume ;


    //#pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*padded_numElem];
      nodes[lnode] = gnode;
    }

    //#pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      x_local[lnode] = x[nodes[lnode]];

    //#pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      y_local[lnode] = y[nodes[lnode]];

    //#pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      z_local[lnode] = z[nodes[lnode]];



    // volume calculations
    //printf("1,");
    volume = lulesh_port_kernels::CalcElemVolume(x_local[0], x_local[1], x_local[2], x_local[3], x_local[4], x_local[5], x_local[6], x_local[7], 
    y_local[0], y_local[1], y_local[2], y_local[3], y_local[4], y_local[5], y_local[6], y_local[7], 
    z_local[0], z_local[1], z_local[2], z_local[3], z_local[4], z_local[5], z_local[6], z_local[7]); 
    printf("2,");
    relativeVolume = volume / volo[k] ; 
    vnew[k] = relativeVolume ;

    delv[k] = relativeVolume - v[k] ;
    // set characteristic length
    arealg[k] = lulesh_port_kernels::CalcElemCharacteristicLength(x_local,y_local,z_local,volume);

    // get nodal velocities from global array and copy into local arrays.
    //#pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodes[lnode];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
    }

    Real_t dt2 = Real_t(0.5) * dt;

    //#pragma unroll
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j]; 
    }

    Real_t detJ;

    lulesh_port_kernels::CalcElemShapeFunctionDerivatives(x_local,y_local,z_local,B,&detJ );

    lulesh_port_kernels::CalcElemVelocityGradient(xd_local,yd_local,zd_local,B,detJ,D);

    // ------------------------
    // CALC LAGRANGE ELEM 2
    // ------------------------

    // calc strain rate and apply as constraint (only done in FB element)
    Real_t vdovNew = D[0] + D[1] + D[2];
    Real_t vdovthird = vdovNew/Real_t(3.0) ;
    
    // make the rate of deformation tensor deviatoric
    vdov[k] = vdovNew ;
    dxx[k] = D[0] - vdovthird ;
    dyy[k] = D[1] - vdovthird ;
    dzz[k] = D[2] - vdovthird ; 

    // ------------------------
    // CALC MONOTONIC Q GRADIENT
    // ------------------------
    Real_t vol = volo[k]*vnew[k];

   // Undo x_local update
    //#pragma unroll
    for ( Index_t j=0 ; j<8 ; ++j ) {
       x_local[j] += dt2 * xd_local[j];
       y_local[j] += dt2 * yd_local[j];
       z_local[j] += dt2 * zd_local[j]; 
    }

   lulesh_port_kernels::CalcMonoGradient(x_local,y_local,z_local,xd_local,yd_local,zd_local,
                          vol, 
                          &delx_zeta[k],&delv_zeta[k],&delx_xi[k],
                          &delv_xi[k], &delx_eta[k], &delv_eta[k]);

  //Check for bad volume 
  if (relativeVolume < 0)
    *bad_vol = k;
  }






    }//end function
};//end class
}//end namespace
