/*

                 Copyright (c) 2010.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 1.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include "../test/alpaka_vector_test.h"
#include "allocator.h"
#include "alpaka_utils.h"
#include "cuda_profiler_api.h"
#include "lulesh_kernels.h"
#include "sm_utils.inl"
#include "util.h"

#include <alpaka/alpaka.hpp>

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

/*
#ifdef USE_MPI
#include <mpi.h>
#endif
*/
#include "lulesh.h"

#include <sys/time.h>
#include <unistd.h>
#define TEST
#define ALPAKA

/****************************************************/
/* Allow flexibility for arithmetic representations */
/****************************************************/

inline __device__ __host__ real4 FABS(real4 arg)
{
    return fabsf(arg);
}

inline __device__ __host__ real8 FABS(real8 arg)
{
    return fabs(arg);
}

template<typename T>
T SQRT(T x)
{
    return alpaka::math::sqrt(x);
};

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

template<typename T>
T FMAX(T x, T y)
{
    return MAX(x, y);
};

/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M 0x0'0007
#define XI_M_SYMM 0x0'0001
#define XI_M_FREE 0x0'0002
#define XI_M_COMM 0x0'0004

#define XI_P 0x0'0038
#define XI_P_SYMM 0x0'0008
#define XI_P_FREE 0x0'0010
#define XI_P_COMM 0x0'0020

#define ETA_M 0x0'01c0
#define ETA_M_SYMM 0x0'0040
#define ETA_M_FREE 0x0'0080
#define ETA_M_COMM 0x0'0100

#define ETA_P 0x0'0e00
#define ETA_P_SYMM 0x0'0200
#define ETA_P_FREE 0x0'0400
#define ETA_P_COMM 0x0'0800

#define ZETA_M 0x0'7000
#define ZETA_M_SYMM 0x0'1000
#define ZETA_M_FREE 0x0'2000
#define ZETA_M_COMM 0x0'4000

#define ZETA_P 0x3'8000
#define ZETA_P_SYMM 0x0'8000
#define ZETA_P_FREE 0x1'0000
#define ZETA_P_COMM 0x2'0000

#define VOLUDER(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, dvdc)                                                 \
    {                                                                                                                 \
        const Real_t twelfth = Real_t(1.0) / Real_t(12.0);                                                            \
                                                                                                                      \
        dvdc = ((a1) + (a2)) * ((b0) + (b1)) - ((a0) + (a1)) * ((b1) + (b2)) + ((a0) + (a4)) * ((b3) + (b4))          \
               - ((a3) + (a4)) * ((b0) + (b4)) - ((a2) + (a5)) * ((b3) + (b5)) + ((a3) + (a5)) * ((b2) + (b5));       \
        dvdc *= twelfth;                                                                                              \
    }

/*
__device__

__forceinline__
void SumOverNodes(Real_t& val, volatile Real_t* smem, int cta_elem, int node) {

  int tid = (cta_elem << 3) + node;
  smem[tid] = val;
  if (node < 4)
  {
    smem[tid] += smem[tid+4];
    smem[tid] += smem[tid+2];
    smem[tid] += smem[tid+1];
  }
  val = smem[(cta_elem << 3)];
}
*/

__device__ __forceinline__ void SumOverNodesShfl(Real_t& val)
{
    val += utils::shfl_xor(val, 4, 8);
    val += utils::shfl_xor(val, 2, 8);
    val += utils::shfl_xor(val, 1, 8);
}

__host__ __device__ __forceinline__ Real_t CalcElemVolume(
    Real_t const x0,
    Real_t const x1,
    Real_t const x2,
    Real_t const x3,
    Real_t const x4,
    Real_t const x5,
    Real_t const x6,
    Real_t const x7,
    Real_t const y0,
    Real_t const y1,
    Real_t const y2,
    Real_t const y3,
    Real_t const y4,
    Real_t const y5,
    Real_t const y6,
    Real_t const y7,
    Real_t const z0,
    Real_t const z1,
    Real_t const z2,
    Real_t const z3,
    Real_t const z4,
    Real_t const z5,
    Real_t const z6,
    Real_t const z7)
{
    Real_t twelveth = Real_t(1.0) / Real_t(12.0);

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

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3)                                                            \
    ((x1) * ((y2) * (z3) - (z2) * (y3)) + (x2) * ((z1) * (y3) - (y1) * (z3)) + (x3) * ((y1) * (z2) - (z1) * (y2)))

    // 11 + 3*14
    Real_t volume = TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20, dy31 + dy72, dy63, dy20, dz31 + dz72, dz63, dz20)
                    + TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70, dy43 + dy57, dy64, dy70, dz43 + dz57, dz64, dz70)
                    + TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50, dy14 + dy25, dy61, dy50, dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

    volume *= twelveth;

    return volume;
}

__host__ __device__ __forceinline__ Real_t CalcElemVolume(Real_t const x[8], Real_t const y[8], Real_t const z[8])
{
    return CalcElemVolume(
        x[0],
        x[1],
        x[2],
        x[3],
        x[4],
        x[5],
        x[6],
        x[7],
        y[0],
        y[1],
        y[2],
        y[3],
        y[4],
        y[5],
        y[6],
        y[7],
        z[0],
        z[1],
        z[2],
        z[3],
        z[4],
        z[5],
        z[6],
        z[7]);
}

void AllocateNodalPersistent(Domain* domain, size_t domNodes)
{
    domain->x.resize(domNodes); /* coordinates */
    domain->y.resize(domNodes);
    domain->z.resize(domNodes);

    domain->xd.resize(domNodes); /* velocities */
    domain->yd.resize(domNodes);
    domain->zd.resize(domNodes);

    domain->xdd.resize(domNodes); /* accelerations */
    domain->ydd.resize(domNodes);
    domain->zdd.resize(domNodes);

    domain->fx.resize(domNodes); /* forces */
    domain->fy.resize(domNodes);
    domain->fz.resize(domNodes);

    domain->nodalMass.resize(domNodes); /* mass */
}

void AllocateElemPersistent(Domain* domain, size_t domElems, size_t padded_domElems)
{
    // domain->matElemlist.resize(domElems) ;  /* material indexset */
    domain->nodelist.resize(8 * padded_domElems); /* elemToNode connectivity */

    domain->lxim.resize(domElems); /* elem connectivity through face */
    domain->lxip.resize(domElems);
    domain->letam.resize(domElems);
    domain->letap.resize(domElems);
    domain->lzetam.resize(domElems);
    domain->lzetap.resize(domElems);

    domain->elemBC.resize(domElems); /* elem face symm/free-surf flag */

    domain->e.resize(domElems); /* energy */
    domain->p.resize(domElems); /* pressure */

    domain->q.resize(domElems); /* q */
    domain->ql.resize(domElems); /* linear term for q */
    domain->qq.resize(domElems); /* quadratic term for q */

    domain->v.resize(domElems); /* relative volume */

    domain->volo.resize(domElems); /* reference volume */
    domain->delv.resize(domElems); /* m_vnew - m_v */
    domain->vdov.resize(domElems); /* volume derivative over volume */

    domain->arealg.resize(domElems); /* elem characteristic length */

    domain->ss.resize(domElems); /* "sound speed" */

    domain->elemMass.resize(domElems); /* mass */
}

void AllocateSymmX(Domain* domain, size_t size)
{
    domain->symmX.resize(size);
}

void AllocateSymmY(Domain* domain, size_t size)
{
    domain->symmY.resize(size);
}

void AllocateSymmZ(Domain* domain, size_t size)
{
    domain->symmZ.resize(size);
}

void terminate_gracefully(void)
{
}

bool InitializeFields(Domain* domain)
{
/* Basic Field Initialization */
#ifdef ALPAKA
    domain->ss.fill(0.);
    domain->e.fill(0.);
    domain->p.fill(0.);
    domain->q.fill(0.);
    domain->v.fill(1.);
    domain->xd.fill(0.);
    domain->yd.fill(0.);
    domain->zd.fill(0.);
    domain->xdd.fill(0.);
    domain->ydd.fill(0.);
    domain->zdd.fill(0.);
    domain->nodalMass.fill(0.);
#else

    thrust::fill(domain->ss.begin(), domain->ss.end(), 0.);
    thrust::fill(domain->e.begin(), domain->e.end(), 0.);
    thrust::fill(domain->p.begin(), domain->p.end(), 0.);
    thrust::fill(domain->q.begin(), domain->q.end(), 0.);
    thrust::fill(domain->v.begin(), domain->v.end(), 1.);

    thrust::fill(domain->xd.begin(), domain->xd.end(), 0.);
    thrust::fill(domain->yd.begin(), domain->yd.end(), 0.);
    thrust::fill(domain->zd.begin(), domain->zd.end(), 0.);

    thrust::fill(domain->xdd.begin(), domain->xdd.end(), 0.);
    thrust::fill(domain->ydd.begin(), domain->ydd.end(), 0.);
    thrust::fill(domain->zdd.begin(), domain->zdd.end(), 0.);

    thrust::fill(domain->nodalMass.begin(), domain->nodalMass.end(), 0.);
#endif

    return true;
}

////////////////////////////////////////////////////////////////////////////////
void Domain::SetupCommBuffers(Int_t edgeNodes)
{
    // allocate a buffer large enough for nodal ghost data
    maxEdgeSize = MAX(this->sizeX, MAX(this->sizeY, this->sizeZ)) + 1;
    maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize * maxEdgeSize);
    maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize);

    // assume communication to 6 neighbors by default
    m_rowMin = (m_rowLoc == 0) ? 0 : 1;
    m_rowMax = (m_rowLoc == m_tp - 1) ? 0 : 1;
    m_colMin = (m_colLoc == 0) ? 0 : 1;
    m_colMax = (m_colLoc == m_tp - 1) ? 0 : 1;
    m_planeMin = (m_planeLoc == 0) ? 0 : 1;
    m_planeMax = (m_planeLoc == m_tp - 1) ? 0 : 1;

#if USE_MPI
    // account for face communication
    Index_t comBufSize = (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) * maxPlaneSize
                         * MAX_FIELDS_PER_MPI_COMM;

    // account for edge communication
    comBufSize
        += ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) + (m_rowMax & m_colMax)
            + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) + (m_rowMax & m_colMin) + (m_rowMin & m_planeMax)
            + (m_colMin & m_planeMax) + (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin))
           * maxPlaneSize * MAX_FIELDS_PER_MPI_COMM;

    // account for corner communication
    // factor of 16 is so each buffer has its own cache line
    comBufSize += ((m_rowMin & m_colMin & m_planeMin) + (m_rowMin & m_colMin & m_planeMax)
                   + (m_rowMin & m_colMax & m_planeMin) + (m_rowMin & m_colMax & m_planeMax)
                   + (m_rowMax & m_colMin & m_planeMin) + (m_rowMax & m_colMin & m_planeMax)
                   + (m_rowMax & m_colMax & m_planeMin) + (m_rowMax & m_colMax & m_planeMax))
                  * CACHE_COHERENCE_PAD_REAL;

    this->commDataSend = new Real_t[comBufSize];
    this->commDataRecv = new Real_t[comBufSize];

    // pin buffers
    cudaHostRegister(this->commDataSend, comBufSize * sizeof(Real_t), 0);
    cudaHostRegister(this->commDataRecv, comBufSize * sizeof(Real_t), 0);

    // prevent floating point exceptions
    memset(this->commDataSend, 0, comBufSize * sizeof(Real_t));
    memset(this->commDataRecv, 0, comBufSize * sizeof(Real_t));

    // allocate shadow GPU buffers
    cudaMalloc(&this->d_commDataSend, comBufSize * sizeof(Real_t));
    cudaMalloc(&this->d_commDataRecv, comBufSize * sizeof(Real_t));

    // prevent floating point exceptions
    cudaMemset(this->d_commDataSend, 0, comBufSize * sizeof(Real_t));
    cudaMemset(this->d_commDataRecv, 0, comBufSize * sizeof(Real_t));
#endif
}

void SetupConnectivityBC(Domain* domain, int edgeElems)
{
    int domElems = domain->numElem;

    Vector_h<Index_t> lxim_h(domElems);
    Vector_h<Index_t> lxip_h(domElems);
    Vector_h<Index_t> letam_h(domElems);
    Vector_h<Index_t> letap_h(domElems);
    Vector_h<Index_t> lzetam_h(domElems);
    Vector_h<Index_t> lzetap_h(domElems);

    /* set up elemement connectivity information */
    lxim_h[0] = 0;
    for(Index_t i = 1; i < domElems; ++i)
    {
        lxim_h[i] = i - 1;
        lxip_h[i - 1] = i;
    }
    lxip_h[domElems - 1] = domElems - 1;

    for(Index_t i = 0; i < edgeElems; ++i)
    {
        letam_h[i] = i;
        letap_h[domElems - edgeElems + i] = domElems - edgeElems + i;
    }
    for(Index_t i = edgeElems; i < domElems; ++i)
    {
        letam_h[i] = i - edgeElems;
        letap_h[i - edgeElems] = i;
    }

    for(Index_t i = 0; i < edgeElems * edgeElems; ++i)
    {
        lzetam_h[i] = i;
        lzetap_h[domElems - edgeElems * edgeElems + i] = domElems - edgeElems * edgeElems + i;
    }
    for(Index_t i = edgeElems * edgeElems; i < domElems; ++i)
    {
        lzetam_h[i] = i - edgeElems * edgeElems;
        lzetap_h[i - edgeElems * edgeElems] = i;
    }

    /* set up boundary condition information */
    Vector_h<Index_t> elemBC_h(domElems);
    for(Index_t i = 0; i < domElems; ++i)
    {
        elemBC_h[i] = 0; /* clear BCs by default */
    }

    Index_t ghostIdx[6]; // offsets to ghost locations

    for(Index_t i = 0; i < 6; ++i)
    {
        ghostIdx[i] = INT_MIN;
    }

    Int_t pidx = domElems;
    if(domain->m_planeMin != 0)
    {
        ghostIdx[0] = pidx;
        pidx += domain->sizeX * domain->sizeY;
    }

    if(domain->m_planeMax != 0)
    {
        ghostIdx[1] = pidx;
        pidx += domain->sizeX * domain->sizeY;
    }

    if(domain->m_rowMin != 0)
    {
        ghostIdx[2] = pidx;
        pidx += domain->sizeX * domain->sizeZ;
    }

    if(domain->m_rowMax != 0)
    {
        ghostIdx[3] = pidx;
        pidx += domain->sizeX * domain->sizeZ;
    }

    if(domain->m_colMin != 0)
    {
        ghostIdx[4] = pidx;
        pidx += domain->sizeY * domain->sizeZ;
    }

    if(domain->m_colMax != 0)
    {
        ghostIdx[5] = pidx;
    }

    /* symmetry plane or free surface BCs */
    for(Index_t i = 0; i < edgeElems; ++i)
    {
        Index_t planeInc = i * edgeElems * edgeElems;
        Index_t rowInc = i * edgeElems;
        for(Index_t j = 0; j < edgeElems; ++j)
        {
            if(domain->m_planeLoc == 0)
            {
                elemBC_h[rowInc + j] |= ZETA_M_SYMM;
            }
            else
            {
                elemBC_h[rowInc + j] |= ZETA_M_COMM;
                lzetam_h[rowInc + j] = ghostIdx[0] + rowInc + j;
            }

            if(domain->m_planeLoc == domain->m_tp - 1)
            {
                elemBC_h[rowInc + j + domElems - edgeElems * edgeElems] |= ZETA_P_FREE;
            }
            else
            {
                elemBC_h[rowInc + j + domElems - edgeElems * edgeElems] |= ZETA_P_COMM;
                lzetap_h[rowInc + j + domElems - edgeElems * edgeElems] = ghostIdx[1] + rowInc + j;
            }

            if(domain->m_rowLoc == 0)
            {
                elemBC_h[planeInc + j] |= ETA_M_SYMM;
            }
            else
            {
                elemBC_h[planeInc + j] |= ETA_M_COMM;
                letam_h[planeInc + j] = ghostIdx[2] + rowInc + j;
            }

            if(domain->m_rowLoc == domain->m_tp - 1)
            {
                elemBC_h[planeInc + j + edgeElems * edgeElems - edgeElems] |= ETA_P_FREE;
            }
            else
            {
                elemBC_h[planeInc + j + edgeElems * edgeElems - edgeElems] |= ETA_P_COMM;
                letap_h[planeInc + j + edgeElems * edgeElems - edgeElems] = ghostIdx[3] + rowInc + j;
            }

            if(domain->m_colLoc == 0)
            {
                elemBC_h[planeInc + j * edgeElems] |= XI_M_SYMM;
            }
            else
            {
                elemBC_h[planeInc + j * edgeElems] |= XI_M_COMM;
                lxim_h[planeInc + j * edgeElems] = ghostIdx[4] + rowInc + j;
            }

            if(domain->m_colLoc == domain->m_tp - 1)
            {
                elemBC_h[planeInc + j * edgeElems + edgeElems - 1] |= XI_P_FREE;
            }
            else
            {
                elemBC_h[planeInc + j * edgeElems + edgeElems - 1] |= XI_P_COMM;
                lxip_h[planeInc + j * edgeElems + edgeElems - 1] = ghostIdx[5] + rowInc + j;
            }
        }
    }

    domain->elemBC = elemBC_h;
    domain->lxim = lxim_h;
    domain->lxip = lxip_h;
    domain->letam = letam_h;
    domain->letap = letap_h;
    domain->lzetam = lzetam_h;
    domain->lzetap = lzetap_h;
}

void Domain::BuildMesh(
    Int_t nx,
    Int_t edgeNodes,
    Int_t edgeElems,
    Int_t domNodes,
    Int_t padded_domElems,
    Vector_h<Real_t>& x_h,
    Vector_h<Real_t>& y_h,
    Vector_h<Real_t>& z_h,
    Vector_h<Int_t>& nodelist_h)
{
    Index_t meshEdgeElems = m_tp * nx;

    x_h.resize(domNodes);
    y_h.resize(domNodes);
    z_h.resize(domNodes);

    // initialize nodal coordinates
    Index_t nidx = 0;
    Real_t tz = Real_t(1.125) * Real_t(m_planeLoc * nx) / Real_t(meshEdgeElems);
    for(Index_t plane = 0; plane < edgeNodes; ++plane)
    {
        Real_t ty = Real_t(1.125) * Real_t(m_rowLoc * nx) / Real_t(meshEdgeElems);
        for(Index_t row = 0; row < edgeNodes; ++row)
        {
            Real_t tx = Real_t(1.125) * Real_t(m_colLoc * nx) / Real_t(meshEdgeElems);
            for(Index_t col = 0; col < edgeNodes; ++col)
            {
                x_h[nidx] = tx;
                y_h[nidx] = ty;
                z_h[nidx] = tz;
                ++nidx;
                // tx += ds ; // may accumulate roundoff...
                tx = Real_t(1.125) * Real_t(m_colLoc * nx + col + 1) / Real_t(meshEdgeElems);
            }
            // ty += ds ;  // may accumulate roundoff...
            ty = Real_t(1.125) * Real_t(m_rowLoc * nx + row + 1) / Real_t(meshEdgeElems);
        }
        // tz += ds ;  // may accumulate roundoff...
        tz = Real_t(1.125) * Real_t(m_planeLoc * nx + plane + 1) / Real_t(meshEdgeElems);
    }

    x = x_h;
    y = y_h;
    z = z_h;

    nodelist_h.resize(padded_domElems * 8);

    // embed hexehedral elements in nodal point lattice
    Index_t zidx = 0;
    nidx = 0;
    for(Index_t plane = 0; plane < edgeElems; ++plane)
    {
        for(Index_t row = 0; row < edgeElems; ++row)
        {
            for(Index_t col = 0; col < edgeElems; ++col)
            {
                nodelist_h[0 * padded_domElems + zidx] = nidx;
                nodelist_h[1 * padded_domElems + zidx] = nidx + 1;
                nodelist_h[2 * padded_domElems + zidx] = nidx + edgeNodes + 1;
                nodelist_h[3 * padded_domElems + zidx] = nidx + edgeNodes;
                nodelist_h[4 * padded_domElems + zidx] = nidx + edgeNodes * edgeNodes;
                nodelist_h[5 * padded_domElems + zidx] = nidx + edgeNodes * edgeNodes + 1;
                nodelist_h[6 * padded_domElems + zidx] = nidx + edgeNodes * edgeNodes + edgeNodes + 1;
                nodelist_h[7 * padded_domElems + zidx] = nidx + edgeNodes * edgeNodes + edgeNodes;
                ++zidx;
                ++nidx;
            }
            ++nidx;
        }
        nidx += edgeNodes;
    }

    nodelist = nodelist_h; // copies host vector to device vector (throw thrust)
}

Domain* NewDomain(
    char* argv[],
    Int_t numRanks,
    Index_t colLoc,
    Index_t rowLoc,
    Index_t planeLoc,
    Index_t nx,
    int tp,
    bool structured,
    Int_t nr,
    Int_t balance,
    Int_t cost)
{
    Domain* domain = new Domain;
#ifndef ALPAKA
    domain->max_streams = 32;
    domain->streams.resize(domain->max_streams);

    for(Int_t i = 0; i < domain->max_streams; i++)
        cudaStreamCreate(&(domain->streams[i]));
#endif
    // TODO get Rid of cuda Event (use some Alpaka function instead)
    // cudaEventCreateWithFlags(&domain->time_constraint_computed,cudaEventDisableTiming);

    Index_t domElems;
    Index_t domNodes;
    Index_t padded_domElems;
    using std::cout;
    using std::endl;
    Vector_h<Index_t> nodelist_h;
    Vector_h<Real_t> x_h;
    Vector_h<Real_t> y_h;
    Vector_h<Real_t> z_h;
    if(structured)
    {
        domain->m_tp = tp;
        domain->m_numRanks = numRanks;

        domain->m_colLoc = colLoc;
        domain->m_rowLoc = rowLoc;
        domain->m_planeLoc = planeLoc;

        Index_t edgeElems = nx;
        Index_t edgeNodes = edgeElems + 1;
        domain->sizeX = edgeElems;
        domain->sizeY = edgeElems;
        domain->sizeZ = edgeElems;

        domain->numElem = domain->sizeX * domain->sizeY * domain->sizeZ;
        domain->padded_numElem = PAD(domain->numElem, 32);
        domain->numNode = (domain->sizeX + 1) * (domain->sizeY + 1) * (domain->sizeZ + 1);
        domain->padded_numNode = PAD(domain->numNode, 32);
        domElems = domain->numElem;
        domNodes = domain->numNode;
        padded_domElems = domain->padded_numElem;
        AllocateElemPersistent(domain, domElems, padded_domElems);
        AllocateNodalPersistent(domain, domNodes);
        domain->SetupCommBuffers(edgeNodes);

        if(!InitializeFields(domain))
            return NULL;
        domain->BuildMesh(nx, edgeNodes, edgeElems, domNodes, padded_domElems, x_h, y_h, z_h, nodelist_h);
        domain->numSymmX = domain->numSymmY = domain->numSymmZ = 0;

        if(domain->m_colLoc == 0)
            domain->numSymmX = (edgeElems + 1) * (edgeElems + 1);
        if(domain->m_rowLoc == 0)
            domain->numSymmY = (edgeElems + 1) * (edgeElems + 1);
        if(domain->m_planeLoc == 0)
            domain->numSymmZ = (edgeElems + 1) * (edgeElems + 1);
        AllocateSymmX(domain, edgeNodes * edgeNodes);
        AllocateSymmY(domain, edgeNodes * edgeNodes);
        AllocateSymmZ(domain, edgeNodes * edgeNodes);

        /* set up symmetry nodesets */

        Vector_h<Index_t> symmX_h(domain->symmX.size());
        Vector_h<Index_t> symmY_h(domain->symmY.size());
        Vector_h<Index_t> symmZ_h(domain->symmZ.size());

        Int_t nidx = 0;
        for(Index_t i = 0; i < edgeNodes; ++i)
        {
            Index_t planeInc = i * edgeNodes * edgeNodes;
            Index_t rowInc = i * edgeNodes;
            for(Index_t j = 0; j < edgeNodes; ++j)
            {
                if(domain->m_planeLoc == 0)
                {
                    symmZ_h[nidx] = rowInc + j;
                }
                if(domain->m_rowLoc == 0)
                {
                    symmY_h[nidx] = planeInc + j;
                }
                if(domain->m_colLoc == 0)
                {
                    symmX_h[nidx] = planeInc + j * edgeNodes;
                }
                ++nidx;
            }
        }

        if(domain->m_planeLoc == 0)
            domain->symmZ = symmZ_h;
        if(domain->m_rowLoc == 0)
            domain->symmY = symmY_h;
        if(domain->m_colLoc == 0)
            domain->symmX = symmX_h;

        SetupConnectivityBC(domain, edgeElems);
    }
    else
    {
        FILE* fp;
        int ee, en;

        if((fp = fopen(argv[2], "r")) == 0)
        {
            printf("could not open file %s\n", argv[2]);
            exit(LFileError);
        }

        bool fsuccess;
        fsuccess = fscanf(fp, "%d %d", &ee, &en);
        domain->numElem = Index_t(ee);
        domain->padded_numElem = PAD(domain->numElem, 32);

        domain->numNode = Index_t(en);
        domain->padded_numNode = PAD(domain->numNode, 32);

        domElems = domain->numElem;
        domNodes = domain->numNode;
        padded_domElems = domain->padded_numElem;

        AllocateElemPersistent(domain, domElems, padded_domElems);
        AllocateNodalPersistent(domain, domNodes);

        InitializeFields(domain);

        /* initialize nodal coordinates */
        x_h.resize(domNodes);
        y_h.resize(domNodes);
        z_h.resize(domNodes);

        for(Index_t i = 0; i < domNodes; ++i)
        {
            double px, py, pz;
            fsuccess = fscanf(fp, "%lf %lf %lf", &px, &py, &pz);
            x_h[i] = Real_t(px);
            y_h[i] = Real_t(py);
            z_h[i] = Real_t(pz);
        }
        domain->x = x_h;
        domain->y = y_h;
        domain->z = z_h;

        /* embed hexehedral elements in nodal point lattice */
        nodelist_h.resize(padded_domElems * 8);
        for(Index_t zidx = 0; zidx < domElems; ++zidx)
        {
            for(Index_t ni = 0; ni < Index_t(8); ++ni)
            {
                int n;
                fsuccess = fscanf(fp, "%d", &n);
                nodelist_h[ni * padded_domElems + zidx] = Index_t(n);
            }
        }
        domain->nodelist = nodelist_h;

        /* set up face-based element neighbors */
        Vector_h<Index_t> lxim_h(domElems);
        Vector_h<Index_t> lxip_h(domElems);
        Vector_h<Index_t> letam_h(domElems);
        Vector_h<Index_t> letap_h(domElems);
        Vector_h<Index_t> lzetam_h(domElems);
        Vector_h<Index_t> lzetap_h(domElems);

        for(Index_t i = 0; i < domElems; ++i)
        {
            int xi_m, xi_p, eta_m, eta_p, zeta_m, zeta_p;
            fsuccess = fscanf(fp, "%d %d %d %d %d %d", &xi_m, &xi_p, &eta_m, &eta_p, &zeta_m, &zeta_p);

            lxim_h[i] = Index_t(xi_m);
            lxip_h[i] = Index_t(xi_p);
            letam_h[i] = Index_t(eta_m);
            letap_h[i] = Index_t(eta_p);
            lzetam_h[i] = Index_t(zeta_m);
            lzetap_h[i] = Index_t(zeta_p);
        }

        domain->lxim = lxim_h;
        domain->lxip = lxip_h;
        domain->letam = letam_h;
        domain->letap = letap_h;
        domain->lzetam = lzetam_h;
        domain->lzetap = lzetap_h;

        /* set up X symmetry nodeset */

        fsuccess = fscanf(fp, "%d", &domain->numSymmX);
        Vector_h<Index_t> symmX_h(domain->numSymmX);
        for(Index_t i = 0; i < domain->numSymmX; ++i)
        {
            int n;
            fsuccess = fscanf(fp, "%d", &n);
            symmX_h[i] = Index_t(n);
        }
        domain->symmX = symmX_h;

        fsuccess = fscanf(fp, "%d", &domain->numSymmY);
        Vector_h<Index_t> symmY_h(domain->numSymmY);
        for(Index_t i = 0; i < domain->numSymmY; ++i)
        {
            int n;
            fsuccess = fscanf(fp, "%d", &n);
            symmY_h[i] = Index_t(n);
        }
        domain->symmY = symmY_h;

        fsuccess = fscanf(fp, "%d", &domain->numSymmZ);
        Vector_h<Index_t> symmZ_h(domain->numSymmZ);
        for(Index_t i = 0; i < domain->numSymmZ; ++i)
        {
            int n;
            fsuccess = fscanf(fp, "%d", &n);
            symmZ_h[i] = Index_t(n);
        }
        domain->symmZ = symmZ_h;

        /* set up free surface nodeset */
        Index_t numFreeSurf;
        fsuccess = fscanf(fp, "%d", &numFreeSurf);
        Vector_h<Index_t> freeSurf_h(numFreeSurf);
        for(Index_t i = 0; i < numFreeSurf; ++i)
        {
            int n;
            fsuccess = fscanf(fp, "%d", &n);
            freeSurf_h[i] = Index_t(n);
        }
        printf("%c\n", fsuccess); // nothing
        fclose(fp);

        /* set up boundary condition information */
        Vector_h<Index_t> elemBC_h(domElems);
        Vector_h<Index_t> surfaceNode_h(domNodes);

        for(Index_t i = 0; i < domain->numElem; ++i)
        {
            elemBC_h[i] = 0;
        }

        for(Index_t i = 0; i < domain->numNode; ++i)
        {
            surfaceNode_h[i] = 0;
        }

        for(Index_t i = 0; i < domain->numSymmX; ++i)
        {
            surfaceNode_h[symmX_h[i]] = 1;
        }

        for(Index_t i = 0; i < domain->numSymmY; ++i)
        {
            surfaceNode_h[symmY_h[i]] = 1;
        }

        for(Index_t i = 0; i < domain->numSymmZ; ++i)
        {
            surfaceNode_h[symmZ_h[i]] = 1;
        }

        for(Index_t zidx = 0; zidx < domain->numElem; ++zidx)
        {
            Int_t mask = 0;

            for(Index_t ni = 0; ni < 8; ++ni)
            {
                mask |= (surfaceNode_h[nodelist_h[ni * domain->padded_numElem + zidx]] << ni);
            }

            if((mask & 0x0f) == 0x0f)
                elemBC_h[zidx] |= ZETA_M_SYMM;
            if((mask & 0xf0) == 0xf0)
                elemBC_h[zidx] |= ZETA_P_SYMM;
            if((mask & 0x33) == 0x33)
                elemBC_h[zidx] |= ETA_M_SYMM;
            if((mask & 0xcc) == 0xcc)
                elemBC_h[zidx] |= ETA_P_SYMM;
            if((mask & 0x99) == 0x99)
                elemBC_h[zidx] |= XI_M_SYMM;
            if((mask & 0x66) == 0x66)
                elemBC_h[zidx] |= XI_P_SYMM;
        }

        for(Index_t zidx = 0; zidx < domain->numElem; ++zidx)
        {
            if(elemBC_h[zidx] == (XI_M_SYMM | ETA_M_SYMM | ZETA_M_SYMM))
            {
                domain->octantCorner = zidx;
                break;
            }
        }

        for(Index_t i = 0; i < domain->numNode; ++i)
        {
            surfaceNode_h[i] = 0;
        }

        for(Index_t i = 0; i < numFreeSurf; ++i)
        {
            surfaceNode_h[freeSurf_h[i]] = 1;
        }

        for(Index_t zidx = 0; zidx < domain->numElem; ++zidx)
        {
            Int_t mask = 0;

            for(Index_t ni = 0; ni < 8; ++ni)
            {
                mask |= (surfaceNode_h[nodelist_h[ni * domain->padded_numElem + zidx]] << ni);
            }

            if((mask & 0x0f) == 0x0f)
                elemBC_h[zidx] |= ZETA_M_SYMM;
            if((mask & 0xf0) == 0xf0)
                elemBC_h[zidx] |= ZETA_P_SYMM;
            if((mask & 0x33) == 0x33)
                elemBC_h[zidx] |= ETA_M_SYMM;
            if((mask & 0xcc) == 0xcc)
                elemBC_h[zidx] |= ETA_P_SYMM;
            if((mask & 0x99) == 0x99)
                elemBC_h[zidx] |= XI_M_SYMM;
            if((mask & 0x66) == 0x66)
                elemBC_h[zidx] |= XI_P_SYMM;
        }

        domain->elemBC = elemBC_h;

        /* deposit energy */
        Real_t arg[] = {3.948746e+7};
        domain->e.changeValue(domain->octantCorner, 1, &arg[0]);
    }
    /* set up node-centered indexing of elements */
    Vector_h<Index_t> nodeElemCount_h(domNodes);

    for(Index_t i = 0; i < domNodes; ++i)
    {
        nodeElemCount_h[i] = 0;
    }

    for(Index_t i = 0; i < domElems; ++i)
    {
        for(Index_t j = 0; j < 8; ++j)
        {
            ++(nodeElemCount_h[nodelist_h[j * padded_domElems + i]]);
        }
    }

    Vector_h<Index_t> nodeElemStart_h(domNodes);

    nodeElemStart_h[0] = 0;
    for(Index_t i = 1; i < domNodes; ++i)
    {
        nodeElemStart_h[i] = nodeElemStart_h[i - 1] + nodeElemCount_h[i - 1];
    }

    Vector_h<Index_t> nodeElemCornerList_h(nodeElemStart_h[domNodes - 1] + nodeElemCount_h[domNodes - 1]);

    for(Index_t i = 0; i < domNodes; ++i)
    {
        nodeElemCount_h[i] = 0;
    }

    for(Index_t j = 0; j < 8; ++j)
    {
        for(Index_t i = 0; i < domElems; ++i)
        {
            Index_t m = nodelist_h[padded_domElems * j + i];
            Index_t k = padded_domElems * j + i;
            Index_t offset = nodeElemStart_h[m] + nodeElemCount_h[m];
            nodeElemCornerList_h[offset] = k;
            ++(nodeElemCount_h[m]);
        }
    }

    Index_t clSize = nodeElemStart_h[domNodes - 1] + nodeElemCount_h[domNodes - 1];
    for(Index_t i = 0; i < clSize; ++i)
    {
        Index_t clv = nodeElemCornerList_h[i];
        if((clv < 0) || (clv > padded_domElems * 8))
        {
            fprintf(
                stderr,
                "AllocateNodeElemIndexes(): nodeElemCornerList entry out "
                "of range!\n");
            exit(1);
        }
    }

    domain->nodeElemStart = nodeElemStart_h;
    domain->nodeElemCount = nodeElemCount_h;
    domain->nodeElemCornerList = nodeElemCornerList_h;

    /* Create a material IndexSet (entire domain same material for now) */
    Vector_h<Index_t> matElemlist_h(domElems);
    for(Index_t i = 0; i < domElems; ++i)
    {
        matElemlist_h[i] = i;
    }
    domain->matElemlist = matElemlist_h;
    Vector_h<Real_t> constraints_h(4, 1e20);
    constraints_h[2] = -1.0;
    constraints_h[3] = -1.0;
    domain->constraints_h = constraints_h;
    domain->constraints_d = constraints_h;
    /*domain->dtcourant_d = dtcourant;
    domain->dthydro_d = dthydro;
    domain->bad_vol_d = bad_vol;
    domain->bad_q_d = bad_q;*/
    /* cudaMallocHost(&domain->dtcourant_h,sizeof(Real_t),0);
    cudaMallocHost(&domain->dthydro_h,sizeof(Real_t),0);
    cudaMallocHost(&domain->bad_vol_h,sizeof(Index_t),0);//check
    cudaMallocHost(&domain->bad_q_h,sizeof(Index_t),0);//check*/
    /* initialize material parameters */
    domain->time_h = Real_t(0.);
    domain->dtfixed = Real_t(-1.0e-6);
    domain->deltatimemultlb = Real_t(1.1);
    domain->deltatimemultub = Real_t(1.2);
    domain->stoptime = Real_t(1.0e-2);
    domain->dtmax = Real_t(1.0e-2);
    domain->cycle = 0;

    domain->e_cut = Real_t(1.0e-7);
    domain->p_cut = Real_t(1.0e-7);
    domain->q_cut = Real_t(1.0e-7);
    domain->u_cut = Real_t(1.0e-7);
    domain->v_cut = Real_t(1.0e-10);

    domain->hgcoef = Real_t(3.0);
    domain->ss4o3 = Real_t(4.0) / Real_t(3.0);

    domain->qstop = Real_t(1.0e+12);
    domain->monoq_max_slope = Real_t(1.0);
    domain->monoq_limiter_mult = Real_t(2.0);
    domain->qlc_monoq = Real_t(0.5);
    domain->qqc_monoq = Real_t(2.0) / Real_t(3.0);
    domain->qqc = Real_t(2.0);

    domain->pmin = Real_t(0.);
    domain->emin = Real_t(-1.0e+15);

    domain->dvovmax = Real_t(0.1);

    domain->eosvmax = Real_t(1.0e+9);
    domain->eosvmin = Real_t(1.0e-9);

    domain->refdens = Real_t(1.0);
    /* initialize field data */
    Vector_h<Real_t> nodalMass_h(domNodes);
    Vector_h<Real_t> volo_h(domElems);
    Vector_h<Real_t> elemMass_h(domElems);
    for(Index_t i = 0; i < domElems; ++i)
    {
        Real_t x_local[8], y_local[8], z_local[8];
        for(Index_t lnode = 0; lnode < 8; ++lnode)
        {
            Index_t gnode = nodelist_h[lnode * padded_domElems + i];
            x_local[lnode] = x_h[gnode];
            y_local[lnode] = y_h[gnode];
            z_local[lnode] = z_h[gnode];
        }

        // volume calculations
        Real_t volume = CalcElemVolume(x_local, y_local, z_local);
        volo_h[i] = volume;
        elemMass_h[i] = volume;
        for(Index_t j = 0; j < 8; ++j)
        {
            Index_t gnode = nodelist_h[j * padded_domElems + i];
            nodalMass_h[gnode] += volume / Real_t(8.0);
        }
    }
    domain->nodalMass = nodalMass_h;
    domain->volo = volo_h;
    domain->elemMass = elemMass_h;

    /* deposit energy */
    domain->octantCorner = 0;
    // deposit initial energy
    // An energy of 3.948746e+7 is correct for a problem with
    // 45 zones along a side - we need to scale it
    Real_t const ebase = 3.948746e+7;
    Real_t scale = (nx * domain->m_tp) / 45.0;
    Real_t einit = ebase * scale * scale * scale;
    // Real_t einit = ebase;
    if(domain->m_rowLoc + domain->m_colLoc + domain->m_planeLoc == 0)
    {
        // Dump into the first zone (which we know is in the corner)
        // of the domain that sits at the origin
#ifdef ALPAKA
        Real_t arg[] = {einit};
        domain->e.changeValue(0, 1, &arg[0]);
#else
        domain->e[0] = einit;
#endif
    }
    // set initial deltatime base on analytic CFL calculation
    domain->deltatime_h = (.5 * cbrt(domain->volo.accessIndex(1))) / sqrt(2 * einit);
    domain->cost = cost;
    domain->regNumList.resize(domain->numElem); // material indexset
    domain->regElemlist.resize(domain->numElem); // material indexset
    domain->regCSR.resize(nr);
    domain->regReps.resize(nr);
    domain->regSorted.resize(nr);
    // Setup region index sets. For now, these are constant sized
    // throughout the run, but could be changed every cycle to
    // simulate effects of ALE on the lagrange solver

    domain->CreateRegionIndexSets(nr, balance);

    /*cout << "[DEBUG] Printing domain variables:\n\ndomain->m_tp: " <<
    domain->m_tp << endl << "domain->m_numRanks: " << domain->m_numRanks << endl;
    cout << "domain->m_colLoc: " << domain->m_colLoc << endl << "domain->m_rowLoc:
    " << domain->m_rowLoc << endl << "domain->m_planeLoc: " << domain->m_planeLoc
    << endl; cout << "domain->numElem: " << domain->numElem << endl <<
    "domain->padded_numElem: " << domain->padded_numElem << endl <<
    "domain->numNode: " << domain->numNode << endl; cout <<
    "domain->padded_numNode: " << domain->padded_numNode << endl <<
    "domain->numSymmX: " << domain->numSymmX << endl << "domain->numSymmY: " <<
    domain->numSymmY << endl; cout << "domain->numSymmZ: " << domain->numSymmZ <<
    endl << "domain->symmZ: " << domain->symmZ[0] << endl << "domain->symmX: " <<
    domain->symmX[0] << endl << "domain->symmY: " << domain->symmY[0] << endl;
    cout << "domain->nodalMass[100]: " << domain->nodalMass[100] << endl <<
    "domain->volo[0]: " << domain->volo[100] << endl << "domain->elemMass[0]: " <<
    domain->elemMass[100] << endl; cout << "domain->dtcourant_d: " <<
    domain->dtcourant_d[0] << endl; cout << "domain->deltatime_h: " <<
    domain->deltatime_h << endl << "domain->cost: " << domain->cost << endl;
    cout << "nr: " << nr << endl << "balance: " << balance << endl;*/
    // exit(1);
    return domain;
}

/*******************	to support region	*********************/
void Domain::sortRegions(Vector_h<Int_t>& regReps_h, Vector_h<Index_t>& regSorted_h)
{
    Index_t temp;
    Vector_h<Index_t> regIndex;
    regIndex.resize(numReg);
    for(int i = 0; i < numReg; i++)
        regIndex[i] = i;

    for(int i = 0; i < numReg - 1; i++)
        for(int j = 0; j < numReg - i - 1; j++)
            if(regReps_h[j] < regReps_h[j + 1])
            {
                temp = regReps_h[j];
                regReps_h[j] = regReps_h[j + 1];
                regReps_h[j + 1] = temp;

                temp = regElemSize[j];
                regElemSize[j] = regElemSize[j + 1];
                regElemSize[j + 1] = temp;

                temp = regIndex[j];
                regIndex[j] = regIndex[j + 1];
                regIndex[j + 1] = temp;
            }
    for(int i = 0; i < numReg; i++)
        regSorted_h[regIndex[i]] = i;
}

// simple function for int pow x^y, y >= 0
Int_t POW(Int_t x, Int_t y)
{
    Int_t res = 1;
    for(Int_t i = 0; i < y; i++)
        res *= x;
    return res;
}

void Domain::CreateRegionIndexSets(Int_t nr, Int_t b)
{
#if USE_MPI
    Index_t myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    srand(myRank);
#else
    srand(0);
    Index_t myRank = 0;
#endif
    numReg = nr;
    balance = b;

    regElemSize = new Int_t[numReg];
    Index_t nextIndex = 0;

    Vector_h<Int_t> regCSR_h(regCSR.size()); // records the begining and end of each region
    Vector_h<Int_t> regReps_h(regReps.size()); // records the rep number per region
    Vector_h<Index_t> regNumList_h(regNumList.size()); // Region number per domain element
    Vector_h<Index_t> regElemlist_h(regElemlist.size()); // region indexset
    Vector_h<Index_t> regSorted_h(regSorted.size()); // keeps index of sorted regions

    // if we only have one region just fill it
    //  Fill out the regNumList with material numbers, which are always
    //  the region index plus one
    if(numReg == 1)
    {
        while(nextIndex < numElem)
        {
            regNumList_h[nextIndex] = 1;
            nextIndex++;
        }
        regElemSize[0] = 0;
    }
    // If we have more than one region distribute the elements.
    else
    {
        Int_t regionNum;
        Int_t regionVar;
        Int_t lastReg = -1;
        Int_t binSize;
        Int_t elements;
        Index_t runto = 0;
        Int_t costDenominator = 0;
        Int_t* regBinEnd = new Int_t[numReg];
        // Determine the relative weights of all the regions.
        for(Index_t i = 0; i < numReg; ++i)
        {
            regElemSize[i] = 0;
            costDenominator += POW((i + 1), balance); // Total cost of all regions
            regBinEnd[i] = costDenominator; // Chance of hitting a given region is (regBinEnd[i]
                                            // - regBinEdn[i-1])/costDenominator
        }
        // Until all elements are assigned
        while(nextIndex < numElem)
        {
            // pick the region
            regionVar = rand() % costDenominator;
            Index_t i = 0;
            while(regionVar >= regBinEnd[i])
                i++;
            // rotate the regions based on MPI rank.  Rotation is Rank % NumRegions
            regionNum = ((i + myRank) % numReg) + 1;
            // make sure we don't pick the same region twice in a row
            while(regionNum == lastReg)
            {
                regionVar = rand() % costDenominator;
                i = 0;
                while(regionVar >= regBinEnd[i])
                    i++;
                regionNum = ((i + myRank) % numReg) + 1;
            }
            // Pick the bin size of the region and determine the number of elements.
            binSize = rand() % 1000;
            if(binSize < 773)
            {
                elements = rand() % 15 + 1;
            }
            else if(binSize < 937)
            {
                elements = rand() % 16 + 16;
            }
            else if(binSize < 970)
            {
                elements = rand() % 32 + 32;
            }
            else if(binSize < 974)
            {
                elements = rand() % 64 + 64;
            }
            else if(binSize < 978)
            {
                elements = rand() % 128 + 128;
            }
            else if(binSize < 981)
            {
                elements = rand() % 256 + 256;
            }
            else
                elements = rand() % 1537 + 512;
            runto = elements + nextIndex;
            // Store the elements.  If we hit the end before we run out of elements
            // then just stop.
            while(nextIndex < runto && nextIndex < numElem)
            {
                regNumList_h[nextIndex] = regionNum;
                nextIndex++;
            }
            lastReg = regionNum;
        }
    }
    // Convert regNumList to region index sets
    // First, count size of each region
    for(Index_t i = 0; i < numElem; ++i)
    {
        int r = regNumList_h[i] - 1; // region index == regnum-1
        regElemSize[r]++;
    }

    Index_t rep;
    // Second, allocate each region index set
    for(Index_t r = 0; r < numReg; ++r)
    {
        if(r < numReg / 2)
            rep = 1;
        else if(r < (numReg - (numReg + 15) / 20))
            rep = 1 + cost;
        else
            rep = 10 * (1 + cost);
        regReps_h[r] = rep;
    }

    sortRegions(regReps_h, regSorted_h);

    regCSR_h[0] = 0;
    // Second, allocate each region index set
    for(Index_t i = 1; i < numReg; ++i)
    {
        regCSR_h[i] = regCSR_h[i - 1] + regElemSize[i - 1];
    }

    // Third, fill index sets
    for(Index_t i = 0; i < numElem; ++i)
    {
        Index_t r = regSorted_h[regNumList_h[i] - 1]; // region index == regnum-1
        regElemlist_h[regCSR_h[r]] = i;
        regCSR_h[r]++;
    }

    // Copy to device
    regCSR = regCSR_h; // records the begining and end of each region
    regReps = regReps_h; // records the rep number per region
    regNumList = regNumList_h; // Region number per domain element
    regElemlist = regElemlist_h; // region indexset
    regSorted = regSorted_h; // keeps index of sorted regions

} // end of create function

inline void TimeIncrement(Domain* domain)
{
    // To make sure dtcourant and dthydro have been updated on host
    Real_t targetdt = domain->stoptime - domain->time_h;
    domain->constraints_h = domain->constraints_d; // copy all constraint values from device
    if((domain->dtfixed <= Real_t(0.0)) && (domain->cycle != Int_t(0)))
    {
        Real_t ratio;

        /* This will require a reduction in parallel */
        Real_t gnewdt = Real_t(1.0e+20);
        Real_t newdt;
        Real_t dtcourant_d_val = domain->constraints_h[0];
        Real_t dthydro_d_val = domain->constraints_h[1];
        if(dtcourant_d_val < gnewdt)
        {
            gnewdt = dtcourant_d_val / Real_t(2.0);
        }
        if(dthydro_d_val < gnewdt)
        {
            gnewdt = dthydro_d_val * Real_t(2.0) / Real_t(3.0);
        }

#if USE_MPI
        MPI_Allreduce(&gnewdt, &newdt, 1, ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE), MPI_MIN, MPI_COMM_WORLD);
#else
        newdt = gnewdt;
#endif

        Real_t olddt = domain->deltatime_h;
        ratio = newdt / olddt;
        if(ratio >= Real_t(1.0))
        {
            if(ratio < domain->deltatimemultlb)
            {
                newdt = olddt;
            }
            else if(ratio > domain->deltatimemultub)
            {
                newdt = olddt * domain->deltatimemultub;
            }
        }

        if(newdt > domain->dtmax)
        {
            newdt = domain->dtmax;
        }
        domain->deltatime_h = newdt;
    }

    /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
    if((targetdt > domain->deltatime_h) && (targetdt < (Real_t(4.0) * domain->deltatime_h / Real_t(3.0))))
    {
        targetdt = Real_t(2.0) * domain->deltatime_h / Real_t(3.0);
    }

    if(targetdt < domain->deltatime_h)
    {
        domain->deltatime_h = targetdt;
    }

    domain->time_h += domain->deltatime_h;

    ++domain->cycle;
}

inline void CalcVolumeForceForElems(Real_t const hgcoef, Domain* domain)
{
    Index_t numElem = domain->numElem;
    Index_t padded_numElem = domain->padded_numElem;

#ifdef DOUBLE_PRECISION
#    ifdef ALPAKA
    Vector_d<Real_t> fx_elem(padded_numElem * 8);

    Vector_d<Real_t> fy_elem(padded_numElem * 8);
    Vector_d<Real_t> fz_elem(padded_numElem * 8);
#    else
    Vector_d<Real_t>* fx_elem = Allocator<Vector_d<Real_t>>::allocate(padded_numElem * 8);
    Vector_d<Real_t>* fy_elem = Allocator<Vector_d<Real_t>>::allocate(padded_numElem * 8);
    Vector_d<Real_t>* fz_elem = Allocator<Vector_d<Real_t>>::allocate(padded_numElem * 8);
#    endif
#else
    thrust::fill(domain->fx.begin(), domain->fx.end(), 0.);
    thrust::fill(domain->fy.begin(), domain->fy.end(), 0.);
    thrust::fill(domain->fz.begin(), domain->fz.end(), 0.);
#endif

    int num_threads = numElem;
    int const block_size = 64;
    int dimGrid = PAD_DIV(num_threads, block_size);

    bool const hourg_gt_zero = hgcoef > Real_t(0.0);
#ifdef ALPAKA

    using CalcElemForce = lulesh_port_kernels::CalcVolumeForceForElems_kernel_class;
    CalcElemForce ElemForceKernel(
        domain->volo.raw(),
        domain->v.raw(),
        domain->p.raw(),
        domain->q.raw(),
        hgcoef,
        numElem,
        padded_numElem,
        domain->nodelist.raw(),
        domain->ss.raw(),
        domain->elemMass.raw(),
        domain->x.raw(),
        domain->y.raw(),
        domain->z.raw(),
        domain->xd.raw(),
        domain->yd.raw(),
        domain->zd.raw(),
        fx_elem.raw(),
        fy_elem.raw(),
        fz_elem.raw(),
        domain->constraints_d.raw(),
        num_threads,
        hourg_gt_zero);

    using Dim2 = alpaka::DimInt<2>;
    using Idx = std::size_t;
    using Vec2 = alpaka::Vec<Dim2, Idx>;

    alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(ElemForceKernel, Vec2{block_size, dimGrid}, true);
    // cudaDeviceSynchronize();
    // Vector_h h_n(domain->ss);

#else
    if(hourg_gt_zero)
    {
        CalcVolumeForceForElems_kernel<true><<<dimGrid, block_size>>>(
            domain->volo.raw(),
            domain->v.raw(),
            domain->p.raw(),
            domain->q.raw(),
            hgcoef,
            numElem,
            padded_numElem,
            domain->nodelist.raw(),
            domain->ss.raw(),
            domain->elemMass.raw(),
            domain->x.raw(),
            domain->y.raw(),
            domain->z.raw(),
            domain->xd.raw(),
            domain->yd.raw(),
            domain->zd.raw(),
#    ifdef DOUBLE_PRECISION
            fx_elem->raw(),
            fy_elem->raw(),
            fz_elem->raw(),
#    else
            domain->fx.raw(),
            domain->fy.raw(),
            domain->fz.raw(),
#    endif
            domain->bad_vol_h,
            num_threads);
    }
    else
    {
        CalcVolumeForceForElems_kernel<false><<<dimGrid, block_size>>>(
            domain->volo.raw(),
            domain->v.raw(),
            domain->p.raw(),
            domain->q.raw(),
            hgcoef,
            numElem,
            padded_numElem,
            domain->nodelist.raw(),
            domain->ss.raw(),
            domain->elemMass.raw(),
            domain->x.raw(),
            domain->y.raw(),
            domain->z.raw(),
            domain->xd.raw(),
            domain->yd.raw(),
            domain->zd.raw(),

#    ifdef DOUBLE_PRECISION
            fx_elem->raw(),
            fy_elem->raw(),
            fz_elem->raw(),
#    else
            domain->fx.raw(),
            domain->fy.raw(),
            domain->fz.raw(),
#    endif
            domain->bad_vol_h,
            num_threads);
    }
#endif // endif ALPAKA

#ifdef DOUBLE_PRECISION
    num_threads = domain->numNode;

    // Launch boundary nodes first
    dimGrid = PAD_DIV(num_threads, block_size);
#    ifdef ALPAKA
    using AddNodeForce = lulesh_port_kernels::AddNodeForcesFromElems_kernel_class;
    AddNodeForce NodeForceKernel(
        domain->numNode,
        domain->padded_numNode,
        domain->nodeElemCount.raw(),
        domain->nodeElemStart.raw(),
        domain->nodeElemCornerList.raw(),
        fx_elem.raw(),
        fy_elem.raw(),
        fz_elem.raw(),
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
        num_threads);
    // cudaCheckError();
    using Dim2 = alpaka::DimInt<2>;
    using Idx = std::size_t;
    using Vec2 = alpaka::Vec<Dim2, Idx>;

    alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(NodeForceKernel, Vec2{block_size, dimGrid}, true);
    // cudaCheckError();
    // cudaDeviceSynchronize();

#    else

    AddNodeForcesFromElems_kernel<<<dimGrid, block_size>>>(
        domain->numNode,
        domain->padded_numNode,
        domain->nodeElemCount.raw(),
        domain->nodeElemStart.raw(),
        domain->nodeElemCornerList.raw(),
        fx_elem.raw(),
        fy_elem.raw(),
        fz_elem.raw(),
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
        num_threads);
    cudaCheckError();
    cudaDeviceSynchronize();
    Allocator<Vector_d<Real_t>>::free(fx_elem, padded_numElem * 8);
    Allocator<Vector_d<Real_t>>::free(fy_elem, padded_numElem * 8);
    Allocator<Vector_d<Real_t>>::free(fz_elem, padded_numElem * 8);
#    endif

#endif // ifdef DOUBLE_PRECISION
    return;
};

inline void CalcVolumeForceForElems(Domain* domain)
{
    Real_t const hgcoef = domain->hgcoef;

    CalcVolumeForceForElems(hgcoef, domain);
    // cudaCheckError();

    // CalcVolumeForceForElems_warp_per_4cell(hgcoef,domain);
};

inline void checkErrors(Domain* domain, int its, int myRank)
{
    auto bad_vol = domain->constraints_h[2];
    auto bad_q = domain->constraints_h[3];
    if(bad_vol != -1.0)
    {
        printf("Rank %i: Volume Error in cell %d at iteration %d\n", myRank, bad_vol, its);
        exit(VolumeError);
    }

    if(bad_q != -1.0)
    {
        printf("Rank %i: Q Error in cell %d at iteration %d\n", myRank, bad_q, its);
        exit(QStopError);
    }
}

inline void CalcForceForNodes(Domain* domain)
{
#if USE_MPI
    CommRecv(*domain, MSG_COMM_SBN, 3, domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1, true, false);
#endif

    CalcVolumeForceForElems(domain);
    // cudaCheckError();
    //  moved here from the main loop to allow async execution with GPU work
    TimeIncrement(domain);

#if USE_MPI
    // initialize pointers
    domain->d_fx = domain->fx.raw();
    domain->d_fy = domain->fy.raw();
    domain->d_fz = domain->fz.raw();

    Domain_member fieldData[3];
    fieldData[0] = &Domain::get_fx;
    fieldData[1] = &Domain::get_fy;
    fieldData[2] = &Domain::get_fz;

    CommSendGpu(
        *domain,
        MSG_COMM_SBN,
        3,
        fieldData,
        domain->sizeX + 1,
        domain->sizeY + 1,
        domain->sizeZ + 1,
        true,
        false,
        domain->streams[2]);
    CommSBNGpu(*domain, 3, fieldData, &domain->streams[2]);
#endif
}

__global__ void CalcAccelerationForNodes_kernel(
    int numNode,
    Real_t* xdd,
    Real_t* ydd,
    Real_t* zdd,
    Real_t* fx,
    Real_t* fy,
    Real_t* fz,
    Real_t* nodalMass)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < numNode)
    {
        Real_t one_over_nMass = Real_t(1.) / nodalMass[tid];
        xdd[tid] = fx[tid] * one_over_nMass;
        ydd[tid] = fy[tid] * one_over_nMass;
        zdd[tid] = fz[tid] * one_over_nMass;
    }
}

inline void CalcAccelerationForNodes(Domain* domain)
{
    int const dimBlock = 128;
    int dimGrid = PAD_DIV(static_cast<int>(domain->numNode), dimBlock);
    // cudaCheckError();

#ifdef ALPAKA
    using CalcAccelerationNodes = lulesh_port_kernels::CalcAccelerationForNodes_kernel_class;
    CalcAccelerationNodes CalcAccNodeKernel(
        domain->numNode,
        domain->xdd.raw(),
        domain->ydd.raw(),
        domain->zdd.raw(),
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
        domain->nodalMass.raw());
    // cudaCheckError();
    using Dim2 = alpaka::DimInt<2>;
    using Idx = std::size_t;
    using Vec2 = alpaka::Vec<Dim2, Idx>;
    alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(CalcAccNodeKernel, Vec2{dimBlock, dimGrid}, true);
    // cudaCheckError();
    // cudaDeviceSynchronize();
#else
    CalcAccelerationForNodes_kernel<<<dimGrid, dimBlock>>>(
        domain->numNode,
        domain->xdd.raw(),
        domain->ydd.raw(),
        domain->zdd.raw(),
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
        domain->nodalMass.raw());
#endif
}

__global__ void ApplyAccelerationBoundaryConditionsForNodes_kernel(int numNodeBC, Real_t* xyzdd, Index_t* symm)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numNodeBC)
    {
        xyzdd[symm[i]] = Real_t(0.0);
    }
}

inline void ApplyAccelerationBoundaryConditionsForNodes(Domain* domain)
{
    Index_t dimBlock = 128;

    Index_t dimGrid = PAD_DIV(domain->numSymmX, dimBlock);
    if(domain->numSymmX > 0)
    {
// Alpaka Code
#ifdef ALPAKA
        using ApplyAccBoundaryConditionsNodes
            = lulesh_port_kernels::ApplyAccelerationBoundaryConditionsForNodes_kernel_class;
        ApplyAccBoundaryConditionsNodes ApplyAccBoundaryKernel(
            domain->numSymmX,
            domain->xdd.raw(),
            domain->symmX.raw());
        using Dim2 = alpaka::DimInt<2>;
        using Idx = std::size_t;
        using Vec2 = alpaka::Vec<Dim2, Idx>;
        alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(ApplyAccBoundaryKernel, Vec2{dimBlock, dimGrid}, false);
        // cudaDeviceSynchronize();
#else
        // CUDA Code
        ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>(
            domain->numSymmX,
            domain->xdd.raw(),
            domain->symmX.raw());
#endif
    }

    dimGrid = PAD_DIV(domain->numSymmY, dimBlock);
    if(domain->numSymmY > 0)
    {
// Alpaka Code
#ifdef ALPAKA
        using ApplyAccBoundaryConditionsNodes
            = lulesh_port_kernels::ApplyAccelerationBoundaryConditionsForNodes_kernel_class;
        ApplyAccBoundaryConditionsNodes ApplyAccBoundaryKernel(
            domain->numSymmY,
            domain->ydd.raw(),
            domain->symmY.raw());
        using Dim2 = alpaka::DimInt<2>;
        using Idx = std::size_t;
        using Vec2 = alpaka::Vec<Dim2, Idx>;
        alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(ApplyAccBoundaryKernel, Vec2{dimBlock, dimGrid}, true);
        // cudaDeviceSynchronize();
#else
        // CUDA Code
        ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>(
            domain->numSymmY,
            domain->ydd.raw(),
            domain->symmY.raw());
#endif
    }

    dimGrid = PAD_DIV(domain->numSymmZ, dimBlock);
    if(domain->numSymmZ > 0)
    {
// Alpaka Code
#ifdef ALPAKA
        using ApplyAccBoundaryConditionsNodes
            = lulesh_port_kernels::ApplyAccelerationBoundaryConditionsForNodes_kernel_class;
        ApplyAccBoundaryConditionsNodes ApplyAccBoundaryKernel(
            domain->numSymmZ,
            domain->zdd.raw(),
            domain->symmZ.raw());
        using Dim2 = alpaka::DimInt<2>;
        using Idx = std::size_t;
        using Vec2 = alpaka::Vec<Dim2, Idx>;
        alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(ApplyAccBoundaryKernel, Vec2{dimBlock, dimGrid}, true);

        // cudaDeviceSynchronize();

// CUDA Code
#else
        ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>(
            domain->numSymmZ,
            domain->zdd.raw(),
            domain->symmZ.raw());
#endif
    }
}

__global__ void CalcPositionAndVelocityForNodes_kernel(
    int numNode,
    Real_t const deltatime,
    Real_t const u_cut,
    Real_t* __restrict__ x,
    Real_t* __restrict__ y,
    Real_t* __restrict__ z,
    Real_t* __restrict__ xd,
    Real_t* __restrict__ yd,
    Real_t* __restrict__ zd,
    Real_t const* __restrict__ xdd,
    Real_t const* __restrict__ ydd,
    Real_t const* __restrict__ zdd)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numNode)
    {
        Real_t xdtmp, ydtmp, zdtmp, dt;
        dt = deltatime;

        xdtmp = xd[i] + xdd[i] * dt;
        ydtmp = yd[i] + ydd[i] * dt;
        zdtmp = zd[i] + zdd[i] * dt;

        if(FABS(xdtmp) < u_cut)
            xdtmp = 0.0;
        if(FABS(ydtmp) < u_cut)
            ydtmp = 0.0;
        if(FABS(zdtmp) < u_cut)
            zdtmp = 0.0;

        x[i] += xdtmp * dt;
        y[i] += ydtmp * dt;
        z[i] += zdtmp * dt;

        xd[i] = xdtmp;
        yd[i] = ydtmp;
        zd[i] = zdtmp;
    }
}

inline void CalcPositionAndVelocityForNodes(Real_t const u_cut, Domain* domain)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode, dimBlock);
#ifdef ALPAKA
    // Alpaka Code
    using CalcPositionAndVelocityForNodes = lulesh_port_kernels::CalcPositionAndVelocityForNodes_kernel_class;
    CalcPositionAndVelocityForNodes CalcPosAndVeloKernel(
        domain->numNode,
        domain->deltatime_h,
        u_cut,
        domain->x.raw(),
        domain->y.raw(),
        domain->z.raw(),
        domain->xd.raw(),
        domain->yd.raw(),
        domain->zd.raw(),
        domain->xdd.raw(),
        domain->ydd.raw(),
        domain->zdd.raw());

    using Dim2 = alpaka::DimInt<2u>;
    using Idx = std::size_t;
    using Vec2 = alpaka::Vec<Dim2, Idx>;
    alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(CalcPosAndVeloKernel, Vec2{dimBlock, dimGrid}, true);
    // cudaDeviceSynchronize();
#else
    // CUDA Code
    CalcPositionAndVelocityForNodes_kernel<<<dimGrid, dimBlock>>>(
        domain->numNode,
        domain->deltatime_h,
        u_cut,
        domain->x.raw(),
        domain->y.raw(),
        domain->z.raw(),
        domain->xd.raw(),
        domain->yd.raw(),
        domain->zd.raw(),
        domain->xdd.raw(),
        domain->ydd.raw(),
        domain->zdd.raw());
#endif

    // cudaDeviceSynchronize();
    // cudaCheckError();
}

inline void LagrangeNodal(Domain* domain)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
    Domain_member fieldData[6];
#endif

    Real_t u_cut = domain->u_cut;

    /* time of boundary condition evaluation is beginning of step for force and
     * acceleration boundary conditions. */
    CalcForceForNodes(domain);

#if USE_MPI
#    ifdef SEDOV_SYNC_POS_VEL_EARLY
    CommRecv(*domain, MSG_SYNC_POS_VEL, 6, domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1, false, false);
#    endif
#endif

    CalcAccelerationForNodes(domain);

    ApplyAccelerationBoundaryConditionsForNodes(domain);

    CalcPositionAndVelocityForNodes(u_cut, domain);
    // cudaCheckError();

#if USE_MPI
#    ifdef SEDOV_SYNC_POS_VEL_EARLY
    // initialize pointers
    domain->d_x = domain->x.raw();
    domain->d_y = domain->y.raw();
    domain->d_z = domain->z.raw();

    domain->d_xd = domain->xd.raw();
    domain->d_yd = domain->yd.raw();
    domain->d_zd = domain->zd.raw();

    fieldData[0] = &Domain::get_x;
    fieldData[1] = &Domain::get_y;
    fieldData[2] = &Domain::get_z;
    fieldData[3] = &Domain::get_xd;
    fieldData[4] = &Domain::get_yd;
    fieldData[5] = &Domain::get_zd;
printf(
  CommSendGpu(*domain, MSG_SYNC_POS_VEL, 6, fieldData,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           false, false, domain->streams[2]) ;
  CommSyncPosVelGpu(*domain, &domain->streams[2]) ;
#    endif
#endif

  return;
}

inline void CalcKinematicsAndMonotonicQGradient(Domain* domain)
{
    Index_t numElem = domain->numElem;
    Index_t padded_numElem = domain->padded_numElem;

    Index_t num_threads = numElem;

    Index_t const block_size = 64;
    Index_t dimGrid = PAD_DIV(num_threads, block_size);
#ifdef ALPAKA
    using CalcKinematicsAndMonotonicQGradient = lulesh_port_kernels::CalcKinematicsAndMonotonicQGradient_kernel_class;
    // cudaCheckError();
    CalcKinematicsAndMonotonicQGradient CalcKinematicsKernelObj(
        numElem,
        padded_numElem,
        domain->deltatime_h,
        domain->nodelist.raw(),
        domain->volo.raw(),
        domain->v.raw(),
        domain->x.raw(),
        domain->y.raw(),
        domain->z.raw(),
        domain->xd.raw(),
        domain->yd.raw(),
        domain->zd.raw(),
        domain->vnew->raw(),
        domain->delv.raw(),
        domain->arealg.raw(),
        domain->dxx->raw(),
        domain->dyy->raw(),
        domain->dzz->raw(),
        domain->vdov.raw(),
        domain->delx_zeta->raw(),
        domain->delv_zeta->raw(),
        domain->delx_xi->raw(),
        domain->delv_xi->raw(),
        domain->delx_eta->raw(),
        domain->delv_eta->raw(),
        domain->constraints_d.raw(),
        num_threads);

    using Dim2 = alpaka::DimInt<2>;
    using Idx = std::size_t;
    using Vec2 = alpaka::Vec<Dim2, Idx>;
    cudaCheckError();
    alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(CalcKinematicsKernelObj, Vec2{block_size, dimGrid}, true);
    // cudaDeviceSynchronize();
    // cudaCheckError();
#else
    CalcKinematicsAndMonotonicQGradient_kernel<<<dimGrid, block_size>>>(
        numElem,
        padded_numElem,
        domain->deltatime_h,
        domain->nodelist.raw(),
        domain->volo.raw(),
        domain->v.raw(),
        domain->x.raw(),
        domain->y.raw(),
        domain->z.raw(),
        domain->xd.raw(),
        domain->yd.raw(),
        domain->zd.raw(),
        domain->vnew->raw(),
        domain->delv.raw(),
        domain->arealg.raw(),
        domain->dxx->raw(),
        domain->dyy->raw(),
        domain->dzz->raw(),
        domain->vdov.raw(),
        domain->delx_zeta->raw(),
        domain->delv_zeta->raw(),
        domain->delx_xi->raw(),
        domain->delv_xi->raw(),
        domain->delx_eta->raw(),
        domain->delv_eta->raw(),
        domain->bad_vol_h,
        num_threads);
    cudaDeviceSynchronize();
    cudaCheckError();
#endif
}

inline void CalcMonotonicQRegionForElems(Domain* domain)
{
    Real_t const ptiny = Real_t(1.e-36);
    Real_t monoq_max_slope = domain->monoq_max_slope;
    Real_t monoq_limiter_mult = domain->monoq_limiter_mult;

    Real_t qlc_monoq = domain->qlc_monoq;
    Real_t qqc_monoq = domain->qqc_monoq;
    Index_t elength = domain->numElem;

    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(elength, dimBlock);
#ifdef ALPAKA
    using CalcMonotonicQRegionForElems = lulesh_port_kernels::CalcMonotonicQRegionForElems_kernel_class;
    CalcMonotonicQRegionForElems CalcMonotonicQRegionKernel(
        qlc_monoq,
        qqc_monoq,
        monoq_limiter_mult,
        monoq_max_slope,
        ptiny,
        elength,
        domain->regElemlist.raw(),
        domain->elemBC.raw(),
        domain->lxim.raw(),
        domain->lxip.raw(),
        domain->letam.raw(),
        domain->letap.raw(),
        domain->lzetam.raw(),
        domain->lzetap.raw(),
        domain->delv_xi->raw(),
        domain->delv_eta->raw(),
        domain->delv_zeta->raw(),
        domain->delx_xi->raw(),
        domain->delx_eta->raw(),
        domain->delx_zeta->raw(),
        domain->vdov.raw(),
        domain->elemMass.raw(),
        domain->volo.raw(),
        domain->vnew->raw(),
        domain->qq.raw(),
        domain->ql.raw(),
        domain->q.raw(),
        domain->qstop,
        domain->constraints_d.raw());

    using Dim2 = alpaka::DimInt<2u>;
    using Idx = std::size_t;
    using Vec2 = alpaka::Vec<Dim2, Idx>;
    alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(CalcMonotonicQRegionKernel, Vec2{dimBlock, dimGrid}, true);
#else
    CalcMonotonicQRegionForElems_kernel<<<dimGrid, dimBlock>>>(
        qlc_monoq,
        qqc_monoq,
        monoq_limiter_mult,
        monoq_max_slope,
        ptiny,
        elength,
        domain->regElemlist.raw(),
        domain->elemBC.raw(),
        domain->lxim.raw(),
        domain->lxip.raw(),
        domain->letam.raw(),
        domain->letap.raw(),
        domain->lzetam.raw(),
        domain->lzetap.raw(),
        domain->delv_xi->raw(),
        domain->delv_eta->raw(),
        domain->delv_zeta->raw(),
        domain->delx_xi->raw(),
        domain->delx_eta->raw(),
        domain->delx_zeta->raw(),
        domain->vdov.raw(),
        domain->elemMass.raw(),
        domain->volo.raw(),
        domain->vnew->raw(),
        domain->qq.raw(),
        domain->ql.raw(),
        domain->q.raw(),
        domain->qstop,
        domain->bad_q_h);
#endif
    // cudaDeviceSynchronize();
    // cudaCheckError();
}

std::vector<std::string> data;
int globalDataIndex = 0;

template<typename T>
void writeOut(T vec, std::string name)
{
    for(int i = 0; i < vec.size(); i++)
    {
        if(name=="delv"&&i==1000){
            std::cout<<vec[i]<<std::endl;
        }
        if(data[globalDataIndex] != std::to_string(vec[i]))
        {
            std::cout << "failure reading vec at " << i << " here " << std::to_string(vec[i])
                      << " lulesh:" << data[globalDataIndex] << "in file " << name << std::endl;
        }
        globalDataIndex++;
    }
}

void read_data()
{
    std::ifstream inputFile("/home/tim/Studium/Alpaka_Project/value_compare.txt");

    if(inputFile.is_open())
    {
        std::string value;
        while(inputFile >> value)
        {
            data.push_back(value);
        }
        inputFile.close();
        std::cout << "Data has been read from " << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << std::endl;
    }
    std::cout<<" size of Data "<<data.size()<<std::endl;
}

template<typename T>
void writeOutwriteOutWord(T word, std::string name)
{
    bool correct=true;
    if(data[globalDataIndex] != std::to_string(word))
    {
        std::cout << "failure reading word here" << std::to_string(word) << " lulesh: " << data[globalDataIndex]
                  << "in file " << name << std::endl;
        correct=false;
    }
    if(correct)std::cout<<" data correct for "<<name<<std::endl;

    globalDataIndex++;
}

template<typename T>
Vector_h<T> vector_h(Vector_d<T>& v)
{
    Vector_h<T> neu(v);
    neu = v;
    return std::move(neu);
}

void CheckErrorApply(

    Index_t length,
    Real_t rho0,
    Real_t e_cut,
    Real_t emin,
    Vector_d<Real_t>& ql,
    Vector_d<Real_t>& qq,
    Vector_d<Real_t>& vnew,
    Vector_d<Real_t>& v,
    Real_t pmin,
    Real_t p_cut,
    Real_t q_cut,
    Real_t eosvmin,
    Real_t eosvmax,
    Vector_d<Index_t>& regElemlist,
    //        const Index_t*  regElemlist,
    Vector_d<Real_t>& e,
    Vector_d<Real_t>& delv,
    Vector_d<Real_t>& p,
    Vector_d<Real_t>& q,
    Real_t ss4o3,
    Vector_d<Real_t>& ss,
    Real_t v_cut,
    Index_t bad_vol,
    Int_t const cost,
    Vector_d<Index_t>& regCSR,
    Vector_d<Index_t>& regReps,
    Index_t const numReg)
{
    writeOut(Vector_h<Real_t>(ql), "ql");
    writeOut(Vector_h<Real_t>(qq), "qq");
    writeOut(Vector_h<Real_t>(vnew), "vnew");
    writeOut(Vector_h<Real_t>(v), "v");
    writeOut(Vector_h<Index_t>(regElemlist), "regElemlist");
    writeOut(Vector_h<Real_t>(e), "e");
    writeOut(Vector_h<Real_t>(delv), "delv");
    writeOut(Vector_h<Real_t>(p), "p");
    writeOut(Vector_h<Real_t>(q), "q");
    writeOut(Vector_h<Real_t>(ss), "ss");
    writeOut(Vector_h<Index_t>(regCSR), "regCSR");
    writeOut(Vector_h<Index_t>(regReps), "regReps");
    writeOutwriteOutWord(length, "length");
    writeOutwriteOutWord(rho0, "rho0");
    writeOutwriteOutWord(e_cut, "e_cut");
    writeOutwriteOutWord(emin, "emin");
    writeOutwriteOutWord(pmin, "pmin");
    writeOutwriteOutWord(bad_vol, "bad_vol");
    writeOutwriteOutWord(p_cut, "p_cut");
    writeOutwriteOutWord(eosvmin, "eosvmin");
    writeOutwriteOutWord(eosvmax, "eosvmax");
    writeOutwriteOutWord(ss4o3, "ss4o3");
    writeOutwriteOutWord(v_cut, "v_cut");
    writeOutwriteOutWord(cost, "cost");
    writeOutwriteOutWord(numReg, "numReg");
}

void ApplyMaterialPropertiesAndUpdateVolume(Domain* domain)
{
    Index_t length = domain->numElem;
    static int iter = 0;
    if(length != 0)
    {
        #define ITER 2
        Index_t dimBlock = 128;
        Index_t dimGrid = PAD_DIV(length, dimBlock);
#define AlPAKA
#ifdef ALPAKA
        Vector_h constraints_h(domain->constraints_d);
            using ApplyMaterialPropertiesAndUpdateVolume
                = lulesh_port_kernels::ApplyMaterialPropertiesAndUpdateVolume_kernel_class;
            // cudaCheckError();
            ApplyMaterialPropertiesAndUpdateVolume ApplyMaterialPropertiesAndUpdateVolumeKernel;
            using Dim2 = alpaka::DimInt<2>;
            using Idx = std::size_t;
            using Vec2 = alpaka::Vec<Dim2, Idx>;
            if(iter==ITER){
                cudaCheckError();
                            read_data();
                CheckErrorApply(
                    length,
                    domain->refdens,
                    domain->e_cut,
                    domain->emin,
                    domain->ql, // dev
                    domain->qq, // dev
                    *domain->vnew, // dev,
                    domain->v, // dev,
                    domain->pmin,
                    domain->p_cut,
                    domain->q_cut,
                    domain->eosvmin,
                    domain->eosvmax,
                    domain->regElemlist, // dev,
                    domain->e, // dev,
                    domain->delv, // dev,
                    domain->p, // dev,
                    domain->q, // dev,
                    domain->ss4o3,
                    domain->ss, // dev,
                    domain->v_cut,
                    domain->constraints_h[2], // dev,
                    domain->cost,
                    domain->regCSR, // dev,
                    domain->regReps, // dev,
                    domain->numReg);
                cudaCheckError();
            }
            alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(
                ApplyMaterialPropertiesAndUpdateVolumeKernel,
                Vec2{dimBlock, dimGrid},
                true,
                length,
                domain->refdens,
                domain->e_cut,
                domain->emin,
                domain->ql.raw(),
                domain->qq.raw(),
                domain->vnew->raw(),
                domain->v.raw(), // error
                domain->pmin,
                domain->p_cut,
                domain->q_cut,
                domain->eosvmin,
                domain->eosvmax,
                domain->regElemlist.raw(),
                domain->e.raw(), // error
                domain->delv.raw(),
                domain->p.raw(), // error
                domain->q.raw(), // error
                domain->ss4o3,
                domain->ss.raw(), // error
                domain->v_cut,
                domain->constraints_d.raw(),
                domain->cost,
                domain->regCSR.raw(),
                domain->regReps.raw(),
                domain->numReg);
            constraints_h = domain->constraints_d;
            cudaCheckError();

            std::cout << std::endl;
            std::cout << " aft " << std::endl;

            std::cout<<" next iteration"<<std::endl;
            iter++;

#else

        ApplyMaterialPropertiesAndUpdateVolume_kernel<<<dimGrid, dimBlock>>>(
            length,
            domain->refdens,
            domain->e_cut,
            domain->emin,
            domain->ql.raw(),
            domain->qq.raw(),
            domain->vnew->raw(),
            domain->v.raw(),
            domain->pmin,
            domain->p_cut,
            domain->q_cut,
            domain->eosvmin,
            domain->eosvmax,
            domain->regElemlist.raw(),
            domain->e.raw(),
            domain->delv.raw(),
            domain->p.raw(),
            domain->q.raw(),
            domain->ss4o3,
            domain->ss.raw(),
            domain->v_cut,
            domain->bad_vol_h,
            domain->cost,
            domain->regCSR.raw(),
            domain->regReps.raw(),
            domain->numReg);
#endif
            // cudaDeviceSynchronize();
            // cudaCheckError();
        }
    }

    inline void LagrangeElements(Domain * domain)
    {
        int allElem = domain->numElem + /* local elem */
                      2 * domain->sizeX * domain->sizeY + /* plane ghosts */
                      2 * domain->sizeX * domain->sizeZ + /* row ghosts */
                      2 * domain->sizeY * domain->sizeZ; /* col ghosts */

        domain->vnew = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);
        domain->dxx = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);
        domain->dyy = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);
        domain->dzz = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);

        domain->delx_xi = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);
        domain->delx_eta = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);
        domain->delx_zeta = Allocator<Vector_d<Real_t>>::allocate(domain->numElem);

        domain->delv_xi = Allocator<Vector_d<Real_t>>::allocate(allElem);
        domain->delv_eta = Allocator<Vector_d<Real_t>>::allocate(allElem);
        domain->delv_zeta = Allocator<Vector_d<Real_t>>::allocate(allElem);

#if USE_MPI
        CommRecv(*domain, MSG_MONOQ, 3, domain->sizeX, domain->sizeY, domain->sizeZ, true, true);
#endif

        /*********************************************/
        /*  Calc Kinematics and Monotic Q Gradient   */
        /*********************************************/
        CalcKinematicsAndMonotonicQGradient(domain);

#if USE_MPI
        Domain_member fieldData[3];

        // initialize pointers
        domain->d_delv_xi = domain->delv_xi->raw();
        domain->d_delv_eta = domain->delv_eta->raw();
        domain->d_delv_zeta = domain->delv_zeta->raw();

        fieldData[0] = &Domain::get_delv_xi;
        fieldData[1] = &Domain::get_delv_eta;
        fieldData[2] = &Domain::get_delv_zeta;

        CommSendGpu(
            *domain,
            MSG_MONOQ,
            3,
            fieldData,
            domain->sizeX,
            domain->sizeY,
            domain->sizeZ,
            true,
            true,
            domain->streams[2]);
        CommMonoQGpu(*domain, domain->streams[2]);
#endif

        Allocator<Vector_d<Real_t>>::free(domain->dxx, domain->numElem);
        Allocator<Vector_d<Real_t>>::free(domain->dyy, domain->numElem);
        Allocator<Vector_d<Real_t>>::free(domain->dzz, domain->numElem);

        /**********************************
         *    Calc Monotic Q Region
         **********************************/
        CalcMonotonicQRegionForElems(domain);

        Allocator<Vector_d<Real_t>>::free(domain->delx_xi, domain->numElem);
        Allocator<Vector_d<Real_t>>::free(domain->delx_eta, domain->numElem);
        Allocator<Vector_d<Real_t>>::free(domain->delx_zeta, domain->numElem);

        Allocator<Vector_d<Real_t>>::free(domain->delv_xi, allElem);
        Allocator<Vector_d<Real_t>>::free(domain->delv_eta, allElem);
        Allocator<Vector_d<Real_t>>::free(domain->delv_zeta, allElem);

        ApplyMaterialPropertiesAndUpdateVolume(domain);
        Allocator<Vector_d<Real_t>>::free(domain->vnew, domain->numElem);
    }

    template<int block_size>
    __global__
#ifdef DOUBLE_PRECISION
        __launch_bounds__(128, 16)
#else
    __launch_bounds__(128, 16)
#endif
            void
            CalcTimeConstraintsForElems_kernel(
                Index_t length,
                Real_t qqc2,
                Real_t dvovmax,
                Index_t * matElemlist,
                Real_t * ss,
                Real_t * vdov,
                Real_t * arealg,
                Real_t * dev_mindtcourant,
                Real_t * dev_mindthydro)
    {
        int tid = threadIdx.x;
        int i = blockDim.x * blockIdx.x + tid;

        __shared__ volatile Real_t s_mindthydro[block_size];
        __shared__ volatile Real_t s_mindtcourant[block_size];

        Real_t mindthydro = Real_t(1.0e+20);
        Real_t mindtcourant = Real_t(1.0e+20);

        Real_t dthydro = mindthydro;
        Real_t dtcourant = mindtcourant;

        while(i < length)
        {
            Index_t indx = matElemlist[i];
            Real_t vdov_tmp = vdov[indx];

            // Computing dt_hydro
            if(vdov_tmp != Real_t(0.))
            {
                Real_t dtdvov = dvovmax / (FABS(vdov_tmp) + Real_t(1.e-20));
                if(dthydro > dtdvov)
                {
                    dthydro = dtdvov;
                }
            }
            if(dthydro < mindthydro)
                mindthydro = dthydro;

            // Computing dt_courant
            Real_t ss_tmp = ss[indx];
            Real_t area_tmp = arealg[indx];
            Real_t dtf = ss_tmp * ss_tmp;

            dtf += ((vdov_tmp < 0.) ? qqc2 * area_tmp * area_tmp * vdov_tmp * vdov_tmp : 0.);

            dtf = area_tmp / SQRT(dtf);

            /* determine minimum timestep with its corresponding elem */
            if(vdov_tmp != Real_t(0.) && dtf < dtcourant)
            {
                dtcourant = dtf;
            }

            if(dtcourant < mindtcourant)
                mindtcourant = dtcourant;

            i += gridDim.x * blockDim.x;
        }

        s_mindthydro[tid] = mindthydro;
        s_mindtcourant[tid] = mindtcourant;

        __syncthreads();

        // Do shared memory reduction
        if(block_size >= 1024)
        {
            if(tid < 512)
            {
                s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 512]);
                s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 512]);
            }
            __syncthreads();
        }

        if(block_size >= 512)
        {
            if(tid < 256)
            {
                s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 256]);
                s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 256]);
            }
            __syncthreads();
        }

        if(block_size >= 256)
        {
            if(tid < 128)
            {
                s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 128]);
                s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 128]);
            }
            __syncthreads();
        }

        if(block_size >= 128)
        {
            if(tid < 64)
            {
                s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 64]);
                s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 64]);
            }
            __syncthreads();
        }

        if(tid < 32)
        {
            s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 32]);
            s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 32]);
        }

        if(tid < 16)
        {
            s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 16]);
            s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 16]);
        }
        if(tid < 8)
        {
            s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 8]);
            s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 8]);
        }
        if(tid < 4)
        {
            s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 4]);
            s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 4]);
        }
        if(tid < 2)
        {
            s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 2]);
            s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 2]);
        }
        if(tid < 1)
        {
            s_mindthydro[tid] = min(s_mindthydro[tid], s_mindthydro[tid + 1]);
            s_mindtcourant[tid] = min(s_mindtcourant[tid], s_mindtcourant[tid + 1]);
        }

        // Store in global memory
        if(tid == 0)
        {
            dev_mindtcourant[blockIdx.x] = s_mindtcourant[0];
            dev_mindthydro[blockIdx.x] = s_mindthydro[0];
        }
    }

    template<int block_size>
    __global__ void CalcMinDtOneBlock(
        Real_t * dev_mindthydro,
        Real_t * dev_mindtcourant,
        Real_t * dtcourant,
        Real_t * dthydro,
        Index_t shared_array_size)
    {
        __shared__ volatile Real_t s_data[block_size];
        int tid = threadIdx.x;

        if(blockIdx.x == 0)
        {
            if(tid < shared_array_size)
                s_data[tid] = dev_mindtcourant[tid];
            else
                s_data[tid] = 1.0e20;

            __syncthreads();

            if(block_size >= 1024)
            {
                if(tid < 512)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 512]);
                }
                __syncthreads();
            }
            if(block_size >= 512)
            {
                if(tid < 256)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 256]);
                }
                __syncthreads();
            }
            if(block_size >= 256)
            {
                if(tid < 128)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 128]);
                }
                __syncthreads();
            }
            if(block_size >= 128)
            {
                if(tid < 64)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 64]);
                }
                __syncthreads();
            }
            if(tid < 32)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 32]);
            }
            if(tid < 16)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 16]);
            }
            if(tid < 8)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 8]);
            }
            if(tid < 4)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 4]);
            }
            if(tid < 2)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 2]);
            }
            if(tid < 1)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 1]);
            }

            if(tid < 1)
            {
                *(dtcourant) = s_data[0];
            }
        }
        else if(blockIdx.x == 1)
        {
            if(tid < shared_array_size)
                s_data[tid] = dev_mindthydro[tid];
            else
                s_data[tid] = 1.0e20;

            __syncthreads();

            if(block_size >= 1024)
            {
                if(tid < 512)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 512]);
                }
                __syncthreads();
            }
            if(block_size >= 512)
            {
                if(tid < 256)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 256]);
                }
                __syncthreads();
            }
            if(block_size >= 256)
            {
                if(tid < 128)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 128]);
                }
                __syncthreads();
            }
            if(block_size >= 128)
            {
                if(tid < 64)
                {
                    s_data[tid] = min(s_data[tid], s_data[tid + 64]);
                }
                __syncthreads();
            }
            if(tid < 32)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 32]);
            }
            if(tid < 16)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 16]);
            }
            if(tid < 8)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 8]);
            }
            if(tid < 4)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 4]);
            }
            if(tid < 2)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 2]);
            }
            if(tid < 1)
            {
                s_data[tid] = min(s_data[tid], s_data[tid + 1]);
            }

            if(tid < 1)
            {
                *(dthydro) = s_data[0];
            }
        }
    }

    inline void CalcTimeConstraintsForElems(Domain * domain)
    {
        Real_t qqc = domain->qqc;
        Real_t qqc2 = Real_t(64.0) * qqc * qqc;
        Real_t dvovmax = domain->dvovmax;

        Index_t const length = domain->numElem;

        int const max_dimGrid = 1024;
        int const dimBlock = 128;
        int dimGrid = std::min(max_dimGrid, PAD_DIV(length, dimBlock));

        Vector_d<Real_t>* dev_mindtcourant = Allocator<Vector_d<Real_t>>::allocate(dimGrid);
        Vector_d<Real_t>* dev_mindthydro = Allocator<Vector_d<Real_t>>::allocate(dimGrid);
        // cudaDeviceSynchronize();
#ifdef ALPAKA
        using CalcTimeConstraintsForElems = lulesh_port_kernels::CalcTimeConstraintsForElems_kernel_class<dimBlock>;
        // cudaCheckError();
        CalcTimeConstraintsForElems CalcTimeConstraintsKernel(
            length,
            qqc2,
            dvovmax,
            domain->matElemlist.raw(),
            domain->ss.raw(),
            domain->vdov.raw(),
            domain->arealg.raw(),
            dev_mindtcourant->raw(),
            dev_mindthydro->raw());

        using Dim2 = alpaka::DimInt<2>;
        using Idx = std::size_t;
        using Vec2 = alpaka::Vec<Dim2, Idx>;

        alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(CalcTimeConstraintsKernel, Vec2{dimBlock, dimGrid}, true);

        // cudaDeviceSynchronize();
        // cudaCheckError();

        // TODO: CalcMinDtOneBlock
        using CalcMinDtOneBlock = lulesh_port_kernels::CalcMinDtOneBlock_class<max_dimGrid>;
        // cudaCheckError();
        CalcMinDtOneBlock CalcMinDtOneBlockKernel(
            dev_mindthydro->raw(),
            dev_mindtcourant->raw(),
            domain->constraints_d.raw(),
            dimGrid);

        using Dim2 = alpaka::DimInt<2>;
        using Idx = std::size_t;
        using Vec2 = alpaka::Vec<Dim2, Idx>;

        alpaka_utils::alpakaExecuteBaseKernel<Dim2, Idx>(
            CalcMinDtOneBlockKernel,
            Vec2{max_dimGrid, 2},
            true); // Should be started with two blocks!

        // cudaDeviceSynchronize();
        // cudaCheckError();

#else
    cudaFuncSetCacheConfig(CalcTimeConstraintsForElems_kernel<dimBlock>, cudaFuncCachePreferShared);

    CalcTimeConstraintsForElems_kernel<dimBlock><<<dimGrid, dimBlock>>>(
        length,
        qqc2,
        dvovmax,
        domain->matElemlist.raw(),
        domain->ss.raw(),
        domain->vdov.raw(),
        domain->arealg.raw(),
        dev_mindtcourant->raw(),
        dev_mindthydro->raw());

    // TODO: if dimGrid < 1024, should launch less threads
    CalcMinDtOneBlock<max_dimGrid><<<2, max_dimGrid, max_dimGrid * sizeof(Real_t), domain->streams[1]>>>(
        dev_mindthydro->raw(),
        dev_mindtcourant->raw(),
        domain->dtcourant_h,
        domain->dthydro_h,
        dimGrid);
#endif

        // cudaEventRecord(domain->time_constraint_computed,domain->streams[1]);

        Allocator<Vector_d<Real_t>>::free(dev_mindtcourant, dimGrid);
        Allocator<Vector_d<Real_t>>::free(dev_mindthydro, dimGrid);
    }

    inline void LagrangeLeapFrog(Domain * domain)
    {
        /* calculate nodal forces, accelerations, velocities, positions, with
         * applied boundary conditions and slide surface considerations */
        LagrangeNodal(domain);

        /* calculate element quantities (i.e. velocity gradient & q), and update
         * material states */
        LagrangeElements(domain);

        CalcTimeConstraintsForElems(domain);
    }

    void printUsage(char* argv[])
    {
        printf("Usage: \n");
        printf("Unstructured grid:  %s -u <file.lmesh> \n", argv[0]);
        printf("Structured grid:    %s -s numEdgeElems \n", argv[0]);
        printf("\nExamples:\n");
        printf("%s -s 45\n", argv[0]);
        printf("%s -u sedov15oct.lmesh\n", argv[0]);
    }

#ifdef SAMI

#    ifdef __cplusplus
    extern "C"
    {
#    endif
#    include "silo.h"
#    ifdef __cplusplus
    }
#    endif

#    define MAX_LEN_SAMI_HEADER 10

#    define SAMI_HDR_NUMBRICK 0
#    define SAMI_HDR_NUMNODES 3
#    define SAMI_HDR_NUMMATERIAL 4
#    define SAMI_HDR_INDEX_START 6
#    define SAMI_HDR_MESHDIM 7

#    define MAX_ADJACENCY 14 /* must be 14 or greater */

    void DumpSAMI(Domain * domain, char* name)
    {
        DBfile* fp;
        int headerLen = MAX_LEN_SAMI_HEADER;
        int headerInfo[MAX_LEN_SAMI_HEADER];
        char varName[] = "brick_nd0";
        char coordName[] = "x";
        int version = 121;
        int numElem = int(domain->numElem);
        int numNode = int(domain->numNode);
        int count;

        int* materialID;
        int* nodeConnect;
        double* nodeCoord;

        if((fp = DBCreate(name, DB_CLOBBER, DB_LOCAL, NULL, DB_PDB)) == NULL)
        {
            printf("Couldn't create file %s\n", name);
            exit(1);
        }

        for(int i = 0; i < MAX_LEN_SAMI_HEADER; ++i)
        {
            headerInfo[i] = 0;
        }
        headerInfo[SAMI_HDR_NUMBRICK] = numElem;
        headerInfo[SAMI_HDR_NUMNODES] = numNode;
        headerInfo[SAMI_HDR_NUMMATERIAL] = 1;
        headerInfo[SAMI_HDR_INDEX_START] = 1;
        headerInfo[SAMI_HDR_MESHDIM] = 3;

        DBWrite(fp, "mesh_data", headerInfo, &headerLen, 1, DB_INT);

        count = 1;
        DBWrite(fp, "version", &version, &count, 1, DB_INT);

        nodeConnect = new int[numElem];

        Vector_h<Index_t> nodelist_h = domain->nodelist;

        for(Index_t i = 0; i < 8; ++i)
        {
            for(Index_t j = 0; j < numElem; ++j)
            {
                nodeConnect[j] = int(nodelist_h[i * domain->padded_numElem + j]) + 1;
            }
            varName[8] = '0' + i;
            DBWrite(fp, varName, nodeConnect, &numElem, 1, DB_INT);
        }

        delete[] nodeConnect;

        nodeCoord = new double[numNode];

        Vector_h<Real_t> x_h = domain->x;
        Vector_h<Real_t> y_h = domain->y;
        Vector_h<Real_t> z_h = domain->z;

        for(Index_t i = 0; i < 3; ++i)
        {
            for(Index_t j = 0; j < numNode; ++j)
            {
                Real_t coordVal;
                switch(i)
                {
                case 0:
                    coordVal = double(x_h[j]);
                    break;
                case 1:
                    coordVal = double(y_h[j]);
                    break;
                case 2:
                    coordVal = double(z_h[j]);
                    break;
                }
                nodeCoord[j] = coordVal;
            }
            coordName[0] = 'x' + i;
            DBWrite(fp, coordName, nodeCoord, &numNode, 1, DB_DOUBLE);
        }

        delete[] nodeCoord;

        materialID = new int[numElem];

        for(Index_t i = 0; i < numElem; ++i)
            materialID[i] = 1;

        DBWrite(fp, "brick_material", materialID, &numElem, 1, DB_INT);

        delete[] materialID;

        DBClose(fp);
    }
#endif

#ifdef SAMI
    void DumpDomain(Domain * domain)
    {
        char meshName[64];
        printf("Dumping SAMI file\n");
        sprintf(meshName, "sedov_%d.sami", int(domain->cycle));

        DumpSAMI(domain, meshName);
    }
#endif

    void write_solution(Domain * locDom)
    {
        Vector_h<Real_t> x_h = locDom->x;
        Vector_h<Real_t> y_h = locDom->y;
        Vector_h<Real_t> z_h = locDom->z;

        std::stringstream filename;
        filename << "xyz.asc";

        FILE* fout = fopen(filename.str().c_str(), "wb");

        for(Index_t i = 0; i < locDom->numNode; i++)
        {
            fprintf(fout, "%10d\n", i);
            fprintf(fout, "%.10f\n", x_h[i]);
            fprintf(fout, "%.10f\n", y_h[i]);
            fprintf(fout, "%.10f\n", z_h[i]);
        }
        fclose(fout);
    }

    ///////////////////////////////////////////////////////////////////////////
    void InitMeshDecomp(Int_t numRanks, Int_t myRank, Int_t * col, Int_t * row, Int_t * plane, Int_t * side)
    {
        Int_t testProcs;
        Int_t dx, dy, dz;
        Int_t myDom;

        // Assume cube processor layout for now
        testProcs = Int_t(cbrt(Real_t(numRanks)) + 0.5);
        if(testProcs * testProcs * testProcs != numRanks)
        {
            printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
        }
        if(sizeof(Real_t) != 4 && sizeof(Real_t) != 8)
        {
            printf("MPI operations only support float and double right now...\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
        }
        if(MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL)
        {
            printf("corner element comm buffers too small.  Fix code.\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
        }

        dx = testProcs;
        dy = testProcs;
        dz = testProcs;

        // temporary test
        if(dx * dy * dz != numRanks)
        {
            printf("error -- must have as many domains as procs\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
        }
        Int_t remainder = dx * dy * dz % numRanks;
        if(myRank < remainder)
        {
            myDom = myRank * (1 + (dx * dy * dz / numRanks));
        }
        else
        {
            myDom = remainder * (1 + (dx * dy * dz / numRanks)) + (myRank - remainder) * (dx * dy * dz / numRanks);
        }

        *col = myDom % dx;
        *row = (myDom / dx) % dy;
        *plane = myDom / (dx * dy);
        *side = testProcs;

        return;
    }

    void VerifyAndWriteFinalOutput(
        Real_t elapsed_time,
        Domain & locDom,
        Int_t its,
        Int_t nx,
        Int_t numRanks,
        bool structured)
    {
        size_t free_mem, total_mem, used_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        used_mem = total_mem - free_mem;
#if LULESH_SHOW_PROGRESS == 0
        printf("   Used Memory         =  %8.4f Mb\n", used_mem / (1024. * 1024.));
#endif

        // GrindTime1 only takes a single domain into account, and is thus a good way
        // to measure processor speed indepdendent of MPI parallelism. GrindTime2
        // takes into account speedups from MPI parallelism
        Real_t grindTime1;
        Real_t grindTime2;
        if(structured)
        {
            grindTime1 = ((elapsed_time * 1e6) / its) / (nx * nx * nx);
            grindTime2 = ((elapsed_time * 1e6) / its) / (nx * nx * nx * numRanks);
        }
        else
        {
            grindTime1 = ((elapsed_time * 1e6) / its) / (locDom.numElem);
            grindTime2 = ((elapsed_time * 1e6) / its) / (locDom.numElem * numRanks);
        }
        // Copy Energy back to Host
        std::cout << structured << std::endl;
        if(structured)
        {
            Real_t e_zero;
            // Real_t* d_ezero_ptr = locDom.e.raw() + locDom.octantCorner; /* octant
            // corner supposed to be 0 */
            Vector_h e_all(locDom.e);
            e_zero = e_all[locDom.octantCorner];
            // cudaMemcpy(&e_zero, d_ezero_ptr, sizeof(Real_t), cudaMemcpyDeviceToHost);

            printf("Run completed:  \n");
            printf("   Problem size        =  %i \n", nx);
            printf("   MPI tasks           =  %i \n", numRanks);
            printf("   Iteration count     =  %i \n", its);
            printf("   Final Origin Energy = %12.6e \n", e_zero);

            Real_t MaxAbsDiff = Real_t(0.0);
            Real_t TotalAbsDiff = Real_t(0.0);
            Real_t MaxRelDiff = Real_t(0.0);
            for(Index_t j = 0; j < nx; ++j)
            {
                for(Index_t k = j + 1; k < nx; ++k)
                {
                    Real_t AbsDiff = FABS(e_all[j * nx + k] - e_all[k * nx + j]);
                    TotalAbsDiff += AbsDiff;

                    if(MaxAbsDiff < AbsDiff)
                        MaxAbsDiff = AbsDiff;

                    Real_t RelDiff = AbsDiff / e_all[k * nx + j];

                    if(MaxRelDiff < RelDiff)
                        MaxRelDiff = RelDiff;
                }
            }

            // Quick symmetry check
            printf("   Testing Plane 0 of Energy Array on rank 0:\n");
            printf("        MaxAbsDiff   = %12.6e\n", MaxAbsDiff);
            printf("        TotalAbsDiff = %12.6e\n", TotalAbsDiff);
            printf("        MaxRelDiff   = %12.6e\n\n", MaxRelDiff);
        }

        // Timing information
        printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
        printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n", grindTime1, grindTime2);
        printf("FOM                  = %10.8g (z/s)\n\n",
               1000.0 / grindTime2); // zones per second

        bool write_solution_flag = true;
        if(write_solution_flag)
        {
            write_solution(&locDom);
        }

        return;
    }

    int main(int argc, char* argv[])
    {
        if(argc < 3)
        {
            printUsage(argv);
            exit(LFileError);
        }
        if(strcmp(argv[1], "-u") != 0 && strcmp(argv[1], "-s") != 0)
        {
            printUsage(argv);
            exit(LFileError);
        }
        int num_iters = -1;
        if(argc == 5)
        {
            num_iters = atoi(argv[4]);
        }

        bool structured = (strcmp(argv[1], "-s") == 0);
        Int_t numRanks;
        Int_t myRank;

#ifdef TEST

        if(test::test_main())
        {
            std::cout << " Some Tests failed << ABORTING LULESH >> " << std::endl;
            return 1;
        }
#endif
#if USE_MPI
        Domain_member fieldData;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#else
    numRanks = 1;
    myRank = 0;
#endif

        /* assume cube subdomain geometry for now */
        Index_t nx = atoi(argv[2]);

        Domain* locDom;

        // Set up the mesh and decompose. Assumes regular cubes for now
        Int_t col, row, plane, side;
        using std::cout;
        using std::endl;

        InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

        // TODO: change default nr to 11
        Int_t nr = 11;
        Int_t balance = 1;
        Int_t cost = 1;

        // TODO: modify this constructor to account for new fields
        // TODO: setup communication buffers
        locDom = NewDomain(argv, numRanks, col, row, plane, nx, side, structured, nr, balance, cost);
#if USE_MPI
        // copy to the host for mpi transfer
        locDom->h_nodalMass = locDom->nodalMass;

        fieldData = &Domain::get_nodalMass;

        // Initial domain boundary communication
        CommRecv(*locDom, MSG_COMM_SBN, 1, locDom->sizeX + 1, locDom->sizeY + 1, locDom->sizeZ + 1, true, false);
        CommSend(
            *locDom,
            MSG_COMM_SBN,
            1,
            &fieldData,
            locDom->sizeX + 1,
            locDom->sizeY + 1,
            locDom->sizeZ + 1,
            true,
            false);
        CommSBN(*locDom, 1, &fieldData);

        // copy back to the device
        locDom->nodalMass = locDom->h_nodalMass;

        // End initialization
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // timestep to solution
        int its = 0;

        if(myRank == 0)
        {
            if(structured)
                printf("Running until t=%f, Problem size=%dx%dx%d\n", locDom->stoptime, nx, nx, nx);
            else
                printf("Running until t=%f, Problem size=%d \n", locDom->stoptime, locDom->numElem);
        }

        cudaProfilerStart();

#if USE_MPI
        double start = MPI_Wtime();
#else
    timeval start;
    gettimeofday(&start, NULL);
#endif

        while(true)
        {
            // this has been moved after computation of volume forces to hide launch
            // latencies
            // TimeIncrement(locDom) ;

            LagrangeLeapFrog(locDom);

            checkErrors(locDom, its, myRank);

#if LULESH_SHOW_PROGRESS
            if(myRank == 0)
                printf("cycle = %d, time = %e, dt=%e\n", its + 1, double(locDom->time_h), double(locDom->deltatime_h));
#endif
            its++;
            if(its == num_iters)
                break;
        }
        // make sure GPU finished its work
        // cudaDeviceSynchronize();
        // Use reduced max elapsed time
        double elapsed_time;
#if USE_MPI
        elapsed_time = MPI_Wtime() - start;
#else
    timeval end;
    gettimeofday(&end, NULL);
    elapsed_time = (double) (end.tv_sec - start.tv_sec) + ((double) (end.tv_usec - start.tv_usec)) / 1'000'000;
#endif

        double elapsed_timeG;
#if USE_MPI
        MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    elapsed_timeG = elapsed_time;
#endif

        cudaProfilerStop();

        if(myRank == 0)
            VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, its, nx, numRanks, structured);

#ifdef SAMI
        DumpDomain(locDom);
#endif

#if USE_MPI
        MPI_Finalize();
#endif

        return 0;
    }
