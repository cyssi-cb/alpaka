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
    bool set=false;
    template <typename Dim,typename Idx,typename kernel>
    static int alpakaExecuteBaseKernel(const kernel &obj,const alpaka::Vec<Dim, Idx> threadsPerGrid, const bool blocking){
        using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
        using Vec2_ = alpaka::Vec<alpaka::DimInt<2>, std::size_t>;
        cudaCheckError();
        std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
        if(blocking){
            #define Queue_ alpaka::Queue<Acc, alpaka::Blocking>
        }
        else {
            #define Queue_ alpaka::Queue<Acc, alpaka::NonBlocking>
        }
        static auto const devAcc=std::make_shared<alpaka::Dev<Acc>>(alpaka::getDevByIdx(alpaka::Platform<Acc>{},0));
        std::cout<<"befdevAcc"<<std::endl;
        //static std::shared_ptr<alpaka::Dev<Acc>> ptr(devAcc(alpaka::getDevByIdx(alpaka::Platform<Acc>{},0)));
        std::cout<<"aftdevAcc"<<std::endl;
        Queue_ queue(*devAcc);
        std::cout << "[DEBUG] Before elementsperthread" << std::endl;
        auto const elementsPerThread = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(1));
        std::cout << "[DEBUG] Before workdiv" << std::endl;
        using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
        std::cout << "[DEBUG] Before workdiv call" << std::endl;
        WorkDiv const workDiv = alpaka::getValidWorkDiv<Acc>(
            *devAcc,
            threadsPerGrid,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        std::cout << "[DEBUG] Before taskKernel" << std::endl;
        //alpaka::trait::GetAccDevProps<alpaka::Dev<Acc>>(*devAcc);
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
