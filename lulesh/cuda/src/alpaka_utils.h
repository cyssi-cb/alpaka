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
        std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
        if(blocking){
            #define Queue_ alpaka::Queue<Acc, alpaka::Blocking>
        }
        else {
            #define Queue_ alpaka::Queue<Acc, alpaka::NonBlocking>
        }
        std::cout << "[DEBUG] Before platform" << std::endl;
        auto const platformAcc = alpaka::Platform<Acc>{};
        std::cout << "[DEBUG] Before devAcc" << std::endl;
        auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
        std::cout << "[DEBUG] Before Queue" << std::endl;
        Queue_ queue(devAcc);
        std::cout << "[DEBUG] Before elementsperthread" << std::endl;
        auto const elementsPerThread = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(1));
        std::cout << "[DEBUG] Before workdiv" << std::endl;
        using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
        std::cout << "[DEBUG] Before workdiv call" << std::endl;
        WorkDiv const workDiv = alpaka::getValidWorkDiv<Acc>(
            devAcc,
            threadsPerGrid,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        std::cout << "[DEBUG] Before taskKernel" << std::endl;
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            obj);
	std::cout << "[DEBUG] Before enqueue" << std::endl;
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        std::cout << "[DEBUG] Before return" << std::endl;
        return EXIT_SUCCESS;
    };
   
    
}
