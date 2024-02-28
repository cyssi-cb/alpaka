


#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
using Real_t = double;
using Index_t = std::int32_t;
using Int_t = std::int32_t;

template<typename Dim, typename Idx, typename kernel, typename... Args>
static int alpakaExecuteBaseKernel(
    kernel const& obj,
    alpaka::Vec<Dim, Idx> const threadsPerGrid,
    bool const blocking,
    Args&&... args)
{
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, Idx>;
    // using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<1>, Idx>;

    using Vec2_ = alpaka::Vec<alpaka::DimInt<2>, std::size_t>;
    using Queue_ = alpaka::Queue<Acc, alpaka::Blocking>;

    static auto const devAcc = std::make_shared<alpaka::Dev<Acc>>(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0));

    Queue_ queue(*devAcc);
    auto const elementsPerThread = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(1));
    using WorkDiv = alpaka::WorkDivMembers<alpaka::DimInt<1>, Idx>;
    auto const workDiv = WorkDiv{Idx(threadsPerGrid[1u]), Idx(threadsPerGrid[0u]), Idx(1)};
    /*WorkDiv const workDiv = alpaka::getValidWorkDiv<Acc>(
        *devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);*/
    // alpaka::trait::GetAccDevProps<alpaka::Dev<Acc>>(*devAcc);
    auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, obj, std::forward<Args>(args)...);
    alpaka::enqueue(queue, taskKernel);
    alpaka::wait(queue);
    return EXIT_SUCCESS;
};

namespace lulesh_port_kernels
{
    ALPAKA_FN_ACC auto sqrt(Real_t arg1) -> Real_t
    {
        return alpaka::math::sqrt(alpaka::math::ConceptMathRsqrt(), arg1);
    };

    ALPAKA_FN_ACC auto FABS(Real_t arg1) -> Real_t
    {
        return alpaka::math::abs(alpaka::math::ConceptMathAbs(), arg1);
    }

    ALPAKA_FN_ACC auto giveMyRegion(Index_t const* regCSR, Index_t const i, Index_t const numReg) -> Index_t
    {
        for(Index_t reg = 0; reg < numReg - 1; reg++)
            if(i < regCSR[reg])
                return reg;
        return (numReg - 1);
    }

    ALPAKA_FN_ACC auto UpdateVolumesForElems_device(Index_t numElem, Real_t& v_cut, Real_t* vnew, Real_t* v, Index_t i)
        -> void
    {
        Real_t tmpV;
        tmpV = vnew[i];

        if(FABS(tmpV - Real_t(1.0)) < v_cut)
            tmpV = Real_t(1.0);
        v[i] = tmpV;
    }

    ALPAKA_FN_ACC auto CalcSoundSpeedForElems_device(
        Real_t& vnewc,
        Real_t rho0,
        Real_t& enewc,
        Real_t& pnewc,
        Real_t& pbvc,
        Real_t& bvc,
        Real_t ss4o3,
        Index_t nz,
        Real_t* ss,
        Index_t iz) -> void
    {
        Real_t ssTmp = (pbvc * enewc + vnewc * vnewc * bvc * pnewc) / rho0;
        if(ssTmp <= Real_t(.1111111e-36))
        {
            ssTmp = Real_t(.3333333e-18);
        }
        else
        {
            ssTmp = sqrt(ssTmp);
        }
        ss[iz] = ssTmp;
    }

    ALPAKA_FN_ACC auto CalcPressureForElems_device(
        Real_t& p_new,
        Real_t& bvc,
        Real_t& pbvc,
        Real_t& e_old,
        Real_t& compression,
        Real_t& vnewc,
        Real_t pmin,
        Real_t p_cut,
        Real_t eosvmax) -> void
    {
        Real_t c1s = Real_t(2.0) / Real_t(3.0);
        Real_t p_temp = p_new;

        bvc = c1s * (compression + Real_t(1.));
        pbvc = c1s;

        p_temp = bvc * e_old;

        if(FABS(p_temp) < p_cut)
            p_temp = Real_t(0.0);

        if(vnewc >= eosvmax) /* impossible condition here? */
            p_temp = Real_t(0.0);

        if(p_temp < pmin)
            p_temp = pmin;

        p_new = p_temp;
    }

    ALPAKA_FN_ACC auto ApplyMaterialPropertiesForElems_device(
        Real_t& eosvmin,
        Real_t& eosvmax,
        Real_t* vnew,
        Real_t* v,
        Real_t& vnewc,
        Index_t* bad_vol,
        Index_t zn) -> void
    {
        vnewc = vnew[zn];

        if(eosvmin != Real_t(0.))
        {
            if(vnewc < eosvmin)
                vnewc = eosvmin;
        }

        if(eosvmax != Real_t(0.))
        {
            if(vnewc > eosvmax)
                vnewc = eosvmax;
        }

        // Now check for valid volume
        Real_t vc = v[zn];
        if(eosvmin != Real_t(0.))
        {
            if(vc < eosvmin)
                vc = eosvmin;
        }
        if(eosvmax != Real_t(0.))
        {
            if(vc > eosvmax)
                vc = eosvmax;
        }
        if(vc <= 0.)
        {
            *bad_vol = zn;
        }
    }

    ALPAKA_FN_ACC auto CalcEnergyForElems_device(
        Real_t& p_new,
        Real_t& e_new,
        Real_t& q_new,
        Real_t& bvc,
        Real_t& pbvc,
        Real_t& p_old,
        Real_t& e_old,
        Real_t& q_old,
        Real_t& compression,
        Real_t& compHalfStep,
        Real_t& vnewc,
        Real_t& work,
        Real_t& delvc,
        Real_t pmin,
        Real_t p_cut,
        Real_t e_cut,
        Real_t q_cut,
        Real_t emin,
        Real_t& qq,
        Real_t& ql,
        Real_t rho0,
        Real_t eosvmax,
        Index_t length) -> void
    {
        Real_t const sixth = Real_t(1.0) / Real_t(6.0);
        Real_t pHalfStep;

        e_new = e_old - Real_t(0.5) * delvc * (p_old + q_old) + Real_t(0.5) * work;

        if(e_new < emin)
        {
            e_new = emin;
        }

        CalcPressureForElems_device(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc, pmin, p_cut, eosvmax);

        Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep);

        if(delvc > Real_t(0.))
        {
            q_new = Real_t(0.);
        }
        else
        {
            Real_t ssc = (pbvc * e_new + vhalf * vhalf * bvc * pHalfStep) / rho0;

            if(ssc <= Real_t(.1111111e-36))
            {
                ssc = Real_t(.3333333e-18);
            }
            else
            {
                ssc = sqrt(ssc);
            }

            q_new = (ssc * ql + qq);
        }

        e_new = e_new + Real_t(0.5) * delvc * (Real_t(3.0) * (p_old + q_old) - Real_t(4.0) * (pHalfStep + q_new));

        e_new += Real_t(0.5) * work;

        if(FABS(e_new) < e_cut)
        {
            e_new = Real_t(0.);
        }
        if(e_new < emin)
        {
            e_new = emin;
        }

        CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut, eosvmax);

        Real_t q_tilde;

        if(delvc > Real_t(0.))
        {
            q_tilde = Real_t(0.);
        }
        else
        {
            Real_t ssc = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;

            if(ssc <= Real_t(.1111111e-36))
            {
                ssc = Real_t(.3333333e-18);
            }
            else
            {
                ssc = sqrt(ssc);
            }

            q_tilde = (ssc * ql + qq);
        }

        e_new = e_new
                - (Real_t(7.0) * (p_old + q_old) - Real_t(8.0) * (pHalfStep + q_new) + (p_new + q_tilde)) * delvc
                      * sixth;

        if(FABS(e_new) < e_cut)
        {
            e_new = Real_t(0.);
        }
        if(e_new < emin)
        {
            e_new = emin;
        }

        CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut, eosvmax);

        if(delvc <= Real_t(0.))
        {
            Real_t ssc = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;

            if(ssc <= Real_t(.1111111e-36))
            {
                ssc = Real_t(.3333333e-18);
            }
            else
            {
                ssc = sqrt(ssc);
            }

            q_new = (ssc * ql + qq);

            if(FABS(q_new) < q_cut)
                q_new = Real_t(0.);
        }

        return;
    }

    class ApplyMaterialPropertiesAndUpdateVolume_kernel_class
    {
    public:
        ApplyMaterialPropertiesAndUpdateVolume_kernel_class(){};

        template<typename TAcc>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            Index_t length,
            Real_t rho0,
            Real_t e_cut,
            Real_t emin,
            Real_t* ql,
            Real_t* qq,
            Real_t* vnew,
            Real_t* v,
            Real_t pmin,
            Real_t p_cut,
            Real_t q_cut,
            Real_t eosvmin,
            Real_t eosvmax,
            Index_t* regElemlist,
            //        const Index_t*  regElemlist,
            Real_t* e,
            Real_t* delv,
            Real_t* p,
            Real_t* q,
            Real_t ss4o3,
            Real_t* ss,
            Real_t v_cut,
            Index_t* bad_vol_h,
            Int_t const cost,
            Index_t const* regCSR,
            Index_t const* regReps,
            Index_t const numReg) const -> void
        {
            /*using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;
            using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;*/

            // Vec const globalThreadIdx =
            // alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            // printf("global ThreadIdx %d,%d\n",globalThreadIdx[0],globalThreadIdx[1]);
            // Vec const globalThreadExtent =
            // alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
            // printf("global ThreadExtent %d,%d\n",globalThreadExtent[0],globalThreadExtent[1]);
            // Vec1 const linearizedGlobalThreadIdx =
            // alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);
            Index_t i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

            // Index_t i = static_cast<Index_t>(linearizedGlobalThreadIdx[0u]);

            Real_t e_old, delvc, p_old, q_old, e_temp, delvc_temp, p_temp, q_temp;
            Real_t compression, compHalfStep;
            Real_t qq_old, ql_old, qq_temp, ql_temp, work;
            Real_t p_new, e_new, q_new;
            Real_t bvc, pbvc, vnewc;

            if(i < length)
            {
                Index_t zidx = regElemlist[i];
                // printf("tidx in block: %d, blockIdx: %d, globalIdx: %d, Zidx: %d\n ", i % 128, i / 128, i, zidx);
                // printf("%d,%d,%d,%d\n ", i % 128, i / 128, i, zidx);
                lulesh_port_kernels::ApplyMaterialPropertiesForElems_device(
                    eosvmin,
                    eosvmax,
                    vnew,
                    v,
                    vnewc,
                    bad_vol_h,
                    zidx);
                /********************** Start EvalEOSForElems **************************/
                // Here we need to find out what region this element belongs to and what
                // is the rep value!
                Index_t region = lulesh_port_kernels::giveMyRegion(regCSR, i, numReg);
                Index_t rep = regReps[region];

                e_temp = e[zidx];
                p_temp = p[zidx];
                q_temp = q[zidx];
                qq_temp = qq[zidx];
                ql_temp = ql[zidx];
                delvc_temp = delv[zidx];
                for(int r = 0; r < rep; r++)
                {
                    e_old = e_temp;
                    p_old = p_temp;
                    q_old = q_temp;
                    qq_old = qq_temp;
                    ql_old = ql_temp;
                    delvc = delvc_temp;
                    work = Real_t(0.);

                    Real_t vchalf;
                    compression = Real_t(1.) / vnewc - Real_t(1.);
                    vchalf = vnewc - delvc * Real_t(.5);
                    compHalfStep = Real_t(1.) / vchalf - Real_t(1.);

                    if(eosvmin != Real_t(0.))
                    {
                        if(vnewc <= eosvmin)
                        { /* impossible due to calling func? */
                            compHalfStep = compression;
                        }
                    }
                    if(eosvmax != Real_t(0.))
                    {
                        if(vnewc >= eosvmax)
                        { /* impossible due to calling func? */
                            p_old = Real_t(0.);
                            compression = Real_t(0.);
                            compHalfStep = Real_t(0.);
                        }
                    }

                    //    qq_old = qq[zidx] ;
                    //    ql_old = ql[zidx] ;
                    //    work = Real_t(0.) ;

                    lulesh_port_kernels::CalcEnergyForElems_device(
                        p_new,
                        e_new,
                        q_new,
                        bvc,
                        pbvc,
                        p_old,
                        e_old,
                        q_old,
                        compression,
                        compHalfStep,
                        vnewc,
                        work,
                        delvc,
                        pmin,
                        p_cut,
                        e_cut,
                        q_cut,
                        emin,
                        qq_old,
                        ql_old,
                        rho0,
                        eosvmax,
                        length);
                } // end for r
                p[zidx] = p_new;
                e[zidx] = e_new;
                q[zidx] = q_new;

                lulesh_port_kernels::CalcSoundSpeedForElems_device(
                    vnewc,
                    rho0,
                    e_new,
                    p_new,
                    pbvc,
                    bvc,
                    ss4o3,
                    length,
                    ss,
                    zidx);

                /********************** End EvalEOSForElems **************************/

                lulesh_port_kernels::UpdateVolumesForElems_device(length, v_cut, vnew, v, zidx);
            }
            return;
        };
    };
} // namespace lulesh_port_kernels
