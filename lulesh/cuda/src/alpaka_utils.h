#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
namespace alpaka_utils{
    /*
        Executes a blocking kernel object (class or struct) on the alpaka::ExampleDefaultAcc the workdiv
        can be specified through the threadsPerGrid parameter
        Executes a kernel object (class or struct) on the alpaka::ExampleDefaultAcc the workdiv
        can be specified through the threadsPerGrid parameter
    */
    template <typename Dim,typename Idx,typename kernel>
    int alpakaExecuteBaseKernel(const kernel &obj,const alpaka::Vec<Dim, Idx> threadsPerGrid, const bool blocking){
        using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
        //std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
        if(blocking){
            #define Queue_ alpaka::Queue<Acc, alpaka::Blocking>
        }
        else {
            #define Queue_ alpaka::Queue<Acc, alpaka::NonBlocking>
        }
        
        auto const platformAcc = alpaka::Platform<Acc>{};
        auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
        Queue_ queue(devAcc);
        auto const elementsPerThread = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(1));
        using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
        WorkDiv const workDiv = alpaka::getValidWorkDiv<Acc>(
            devAcc,
            threadsPerGrid,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            obj);

        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        return EXIT_SUCCESS;
    };
   
    
}