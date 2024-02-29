#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

namespace alpaka_utils
{
    /*
       Executes a blocking kernel object (class or struct) on the
       alpaka::ExampleDefaultAcc the workdiv can be specified through the
       threadsPerGrid parameter Executes a kernel object (class or struct) on the
       alpaka::ExampleDefaultAcc the workdiv can be specified through the
       threadsPerGrid parameter
    */
    // using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, std::size_t>;
    // using Queue_ =alpaka::Queue<Acc, alpaka::Blocking>
    // bool set=false;
    template<typename Acc>
    class Alpaka_Domain{
        using Queue_ = alpaka::Queue<Acc, alpaka::Blocking>;
        private:
            std::shared_ptr<Queue_> queue;
            std::shared_ptr<alpaka::Dev<Acc>> dev;
        public:


            Alpaka_Domain(){
                dev = std::make_shared<alpaka::Dev<Acc>>(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0));
                queue=std::make_shared<Queue_>(*dev);
            };
            Queue_ & getQueue(){
                return *queue;
            }


    };
    template<typename Dim, typename Idx, typename kernel, typename... Args>
    static int alpakaExecuteBaseKernel(
        kernel const& obj,
        alpaka::Vec<Dim, Idx> const threadsPerGrid,
        bool const blocking,
        Args&&... args)
    {
        // using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, Idx>;
        using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, Idx>;
        //define accelerator, queue and device
        static Alpaka_Domain<Acc> dom = Alpaka_Domain<Acc>();

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
        alpaka::enqueue(dom.getQueue(), taskKernel);
        alpaka::wait(dom.getQueue());
        return EXIT_SUCCESS;

    };

} // namespace alpaka_utils
