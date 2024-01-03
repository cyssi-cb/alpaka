#include <stdint.h>
#include <alpaka/alpaka.hpp>
namespace lulesh_port_kernels{

using Real_t = double;
using Index_t= std::uint32_t;
auto static inline SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
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
static inline auto VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
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
static inline auto CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
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
static inline
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
static auto inline CalcElemShapeFunctionDerivatives( const Real_t* const x,
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
auto inline CalcHourglassModes(const Real_t xn[8], const Real_t yn[8], const Real_t zn[8],
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
auto CalcElemVolumeDerivative(Real_t dvdx[8],
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
template< bool hourg_gt_zero,typename Index_t> 
class CalcVolumeForceForElems_kernel{

    const Real_t* __restrict__ volo;
    const Real_t* __restrict__ v;
    const Real_t* __restrict__ p; 
    const Real_t* __restrict__ q;
    Real_t hourg;
    Index_t numElem; 
    Index_t padded_numElem; 
    const Index_t* __restrict__ nodelist;
    const Real_t* __restrict__ ss;
    const Real_t* __restrict__ elemMass;
    const Real_t* __restrict__ x;   const Real_t* __restrict__ y,  const Real_t* __restrict__  z;
    const Real_t* __restrict__ xd;  const Real_t* __restrict__ yd,  const Real_t* __restrict__  zd;
    Real_t* __restrict__ fx_elem; 
    Real_t* __restrict__ fy_elem; 
    Real_t* __restrict__ fz_elem;

    Real_t coefficient;
    Index_t* __restrict__ bad_vol;
    const Index_t num_threads;


    Real_t xn[8],yn[8],zn[8];
    Real_t xdn[8],ydn[8],zdn[8];
    Real_t dvdxn[8],dvdyn[8],dvdzn[8];
    Real_t hgfx[8],hgfy[8],hgfz[8];
    Real_t hourgam[8][4];
    CalcVolumeForceForElems_kernel(

    const Real_t* __restrict__ volo, 
    const Real_t* __restrict__ v,
    const Real_t* __restrict__ p, 
    const Real_t* __restrict__ q,
    Real_t hourg,
    Index_t numElem, 
    Index_t padded_numElem, 
    const Index_t* __restrict__ nodelist,
    const Real_t* __restrict__ ss, 
    const Real_t* __restrict__ elemMass,
    const Real_t* __restrict__ x,   const Real_t* __restrict__ y,  const Real_t* __restrict__  z,
    const Real_t* __restrict__ xd,  const Real_t* __restrict__ yd,  const Real_t* __restrict__  zd,
    //TextureObj<Real_t> x,  TextureObj<Real_t> y,  TextureObj<Real_t> z,
    //TextureObj<Real_t> xd,  TextureObj<Real_t> yd,  TextureObj<Real_t> zd,
    //TextureObj<Real_t>* x,  TextureObj<Real_t>* y,  TextureObj<Real_t>* z,
    //TextureObj<Real_t>* xd,  TextureObj<Real_t>* yd,  TextureObj<Real_t>* zd,
    Real_t* __restrict__ fx_elem, 
    Real_t* __restrict__ fy_elem, 
    Real_t* __restrict__ fz_elem,
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads):volo(volo),ss(ss),elemMass(elemMass),x(x),y(y),z(z),xd(xd),yd(yd),zd(zd),
    fx_elem(fx_elem),fy_elem(fy_elem),fz_elem(fz_elem),coefficient(coefficient),bad_vol(bad_vol),num_threads(num_threads);
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
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
      ALPAKA_VECTORIZE_HINT
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

      ALPAKA_VECTORIZE_HINT
      for (int i=0;i<8;i++)
        xn[i] =x[n[i]];

      ALPAKA_VECTORIZE_HINT
      for (int i=0;i<8;i++)
        yn[i] =y[n[i]];

      ALPAKA_VECTORIZE_HINT
      for (int i=0;i<8;i++)
        zn[i] =z[n[i]];


      Real_t volume13 = cbrt(det);
      coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

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

      ALPAKA_VECTORIZE_HINT
      for (int i=0;i<8;i++)
      {
        hgfx[i] = -( sigxx*B[0][i] );
        hgfy[i] = -( sigxx*B[1][i] );
        hgfz[i] = -( sigxx*B[2][i] );
      }

      if (hourg_gt_zero)
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

        ALPAKA_VECTORIZE_HINT
        for (int i=0;i<8;i++)
          xdn[i] =xd[n[i]];

        ALPAKA_VECTORIZE_HINT
        for (int i=0;i<8;i++)
          ydn[i] =yd[n[i]];

        ALPAKA_VECTORIZE_HINT
        for (int i=0;i<8;i++)
          zdn[i] =zd[n[i]];




        CalcElemFBHourglassForce
        ( &xdn[0],&ydn[0],&zdn[0],
          hourgam[0],hourgam[1],hourgam[2],hourgam[3],
          hourgam[4],hourgam[5],hourgam[6],hourgam[7],
          coefficient,
          &hgfx[0],&hgfy[0],&hgfz[0]
        );

      }
      ALPAKA_VECTORIZE_HINT
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
}//end namespace
