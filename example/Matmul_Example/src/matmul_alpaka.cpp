/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>
#include <inttypes.h>
template <typename TElem,typename TIdx>
void native_serial_implementation(TElem * MatrixA, TElem * MatrixB,TElem * MatrixC, TIdx size){
    for(TIdx i(0);i<size;i++)
        for(TIdx j(0);j<size;j++)
            for(TIdx k(0);k<size;k++)MatrixC[i*size+j]+=MatrixA[i*size+k]*MatrixB[k*size+j];

}
template <typename TData,typename TIdx>
void printMatrix(TData* Matrix,TIdx size){
    for(TIdx i(0);i<size;i++){
        for(TIdx j(0);j<size;j++){
            
            std::cout<<Matrix[i*size+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
}
template <typename TData,typename TIdx>
TIdx compareResult(TData* CPU_serial, TData* GPU,TIdx size){
    for(TIdx i(0);i<size;i++)
        for(TIdx j(0);j<size;j++)if(CPU_serial[i*size+j]!=GPU[i*size+j]){
            std::cout<<"CPU "<<CPU_serial[i*size+j]<< " not equal to "<<" GPU "<<GPU[i*size+j]<<std::endl;
            return i*size+j;};
    TIdx ret(-1);
    return ret;
}
//! A vector addition kernel.
template <typename TIdx,typename TElem>
class MulKernel
{
private: 
    const TIdx csize;
    TElem const * const A;
    TElem const * const B;
    TElem * const C;

public:
    
    MulKernel(TElem const * const MatrixA, TElem const * const MatrixB,TElem * const MatrixC, TIdx size):csize(size),A(MatrixA),B(MatrixB),C(MatrixC){};
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //!  \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TIdx const& numElements) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        //printf("col:%u",static_cast<unsigned>(globalThreadIdx[1u]));
        TIdx row=static_cast<unsigned>(globalThreadIdx[0u]);
        TIdx col=static_cast<unsigned>(globalThreadIdx[1u]);
        /*if(row*csize+col==0){
        for(std::uint32_t i=0;i<1000;i++){
            printf("id:%u, , %u \n",i,A[i]);
        }
        }*/
        //for(;k<csize;k++)this->MatrixC[row*csize+col]=1;
        
        for(TIdx k(0);k<csize;++k)C[row*csize+col]+=A[row*csize+k]*B[k*csize+col];
        /*if(row*csize+col==0){
            for(TIdx t(0);t<csize;t++){
                for(TIdx x(0);x<csize;x++){
                    printf("id:%u, mat:%u , %u \n",(t*csize+x),A[t*csize+x],B[t*csize+x]);
                }
            }
        }*/
        //printf("[row:%u, col:%u, id:%u, mat1:%u, mat2:%u  \n",row,col,(row*csize+col),A[row*csize+col],B[row*csize+col]);
        //printf(" hier\n");
    };
};

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Define the index domain
    using Dim = alpaka::DimInt<2u>;
    using Dim1D = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Define the work division
    using Vec2D= alpaka::Vec<Dim,Idx>;
    using Vec1D= alpaka::Vec<Dim1D,Idx>;
    Idx const size(12);

    auto const elementsPerThread=Vec2D::all(static_cast<Idx>(1));
    Idx const numElements(size*size);
    Vec1D const extent1D=Vec1D(numElements);
    Vec2D const extent=Vec2D{size,size};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, Data, Dim1D, Idx>;
    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost,extent1D));
    BufHost bufHostB(alpaka::allocBuf<Data, Idx>(devHost, extent1D));
    BufHost bufHostC(alpaka::allocBuf<Data, Idx>(devHost, extent1D));
    BufHost bufHostTest(alpaka::allocBuf<Data, Idx>(devHost, extent1D));
    // Initialize the host input vectors A and B
    Data* const pBufHostA(alpaka::getPtrNative(bufHostA));
    Data* const pBufHostB(alpaka::getPtrNative(bufHostB));
    Data* const pBufHostC(alpaka::getPtrNative(bufHostC));
    Data* const pbufHostTest(alpaka::getPtrNative(bufHostTest));

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);
    using std::cout;
    using std::endl;
    for(Idx i(0); i < size; ++i)
    {
        for(Idx j(0);j<size;++j){
                    pBufHostA[i*size+j] = dist(eng);
                    pBufHostB[i*size+j]= dist(eng);
                    pBufHostC[i*size+j] = 0;
                    pbufHostTest[i*size+j]=0;
        }

    }
    /*for(std::uint32_t a=0;a<1000;a++){
            printf("id:%u, , %u \n",a,pBufHostA[a]);
        }*/
    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::Buf<DevAcc, Data, Dim1D, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent1D));
    BufAcc bufAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent1D));
    BufAcc bufAccC(alpaka::allocBuf<Data, Idx>(devAcc, extent1D));
    
    // Copy Host -> Acc
    alpaka::memcpy(queue, bufAccA, bufHostA,extent1D);
    alpaka::memcpy(queue, bufAccB, bufHostB,extent1D);
    alpaka::memcpy(queue, bufAccC, bufHostC,extent1D);

    // Instantiate the kernel function object
    MulKernel kernel(alpaka::getPtrNative(bufAccA),alpaka::getPtrNative(bufAccB),alpaka::getPtrNative(bufAccC),size);
    //MulKernel kernel(size);

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        numElements);

    // Enqueue the kernel execution task
    {
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::memcpy(queue, bufHostC, bufAccC,extent1D);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }
    native_serial_implementation(pBufHostA,pBufHostB,pbufHostTest,size);
    Idx res=static_cast<Idx>(compareResult(pbufHostTest,pBufHostC,size));
    if(res!=static_cast<Idx>(-1)){
        std::cout << "First wrong result at" << res << std::endl;
        cout<<" Host: "<<endl;
        printMatrix(pbufHostTest,size);
        cout<<" GPU: "<<endl;
        printMatrix(pBufHostC,size);

        return EXIT_FAILURE;
    }
    else {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    
#endif
}
