#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#define ALPAKA
template <class T>
class Vector_h;

template <class T>
class Vector_d;
template<typename TSource,typename Ttarget,typename Textend>
void alpaka_copy(TSource source,Ttarget target,Textend extend){
  using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>
  using DevAcc = alpaka::Dev<Acc>;
  using QueueProperty = alpaka::Blocking;
  using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
  auto const platform = alpaka::Platform<Acc>{};
  auto const devAcc = alpaka::getDevByIdx(platform, 0);
  QueueAcc queue(devAcc);
  alpaka::memcpy(queue, target,source,extend);
}

// host vector
#ifdef ALPAKA
template <typename T>
class Vector_h{
  using Idx =std::size_t;
  Idx size;
  using Dim alpaka::DimInt<1u>;
  
  using DevHost = alpaka::DevCpu;
  auto const platformHost = alpaka::PlatformCpu{};
  auto const devHost = alpaka::getDevByIdx(platformHost, 0);
  using Vec= alpaka::Vec<Dim,Idx;
  using BufHost = alpaka::Buf<DevAcc, T, Dim, Idx>;
  BufHost bufHost;
  T * const pHost;
  Vec const extent1D;
  auto init(Idx N)->void{
    extend1D=Vec(N);
    bufHost=alpaka::allocBuf<Data, Idx>(devHost,extent1D);
    pHost=alpaka::getPtrNative(bufHost);
  }
  Vector_h() {}
  inline Vector_h(Idx N):size(N){ init(N);};
  Vector_h(Idx N,T v):size(N){
    init(N);
    for(Idx i(0);i<N;i++)pHost[i]=v;
  }
  Vector_h(const Vector_h<T>& source):size(source.size()){
    init(N);
    T* const pSourceHost =source.raw();
    for(Idx i(0);i<N;i++){
          pHost[i]=a[i];
    }
  }
  Vector_h(const Vector_d_<T> &source):size(source.size()){
    init(N);
    alpaka_copy(a.bufAcc,this->bufHost,extent1D);//copy device to host
  }
  Vector_h<T>& operator=(const Vector_h<T> &a) { 
    if(this.size()!=a.size())return *this;
    for(Idx i(0);i<size;i++)pHost[i]=a.pHost[i];
    return *this;
  }
  Vector_h<T> &operator=(const Vector_d<T> &a){
    alpaka_copy(a.bufAcc,this->bufHost,extent1D);/copy device to source
  }
  auto size()->Idx{
    return this->size;
  }
  inline T* raw() { 
    return pHost;
  } 
  inline const T* raw() const { 
    return pHost;
  } 
}
#else
template <class T>
class Vector_h: public thrust::host_vector<T> {
  public:

  // Constructors
  Vector_h() {}
  inline Vector_h(int N) : thrust::host_vector<T>(N) {}
  inline Vector_h(int N, T v) : thrust::host_vector<T>(N,v) {}
  inline Vector_h(const Vector_h<T>& a) : thrust::host_vector<T>(a) {}
  inline Vector_h(const Vector_d<T>& a) : thrust::host_vector<T>(a) {}

  template< typename OtherVector >
    inline void copy( const OtherVector &a ) { 
      this->assign( a.begin( ), a.end( ) ); 
    }

  inline Vector_h<T>& operator=(const Vector_h<T> &a) { copy(a); return *this; }
  inline Vector_h<T>& operator=(const Vector_d<T> &a) { copy(a); return *this; }

  inline T* raw() { 
    if(bytes()>0) return thrust::raw_pointer_cast(this->data()); 
    else return 0;
  } 

  inline const T* raw() const { 
    if(bytes()>0) return thrust::raw_pointer_cast(this->data()); 
    else return 0;
  } 

  inline size_t bytes() const { return this->size()*sizeof(T); }

};
#endif
// device vector
#ifdef ALPAKA
template <typename T>
class Vector_h{
  using Idx =std::size_t;
  Idx size;
  using Dim alpaka::DimInt<1u>;
  using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
  using DevAcc = alpaka::Dev<Acc>;
  auto const platform = alpaka::Platform<Acc>{};
  auto const devAcc = alpaka::getDevByIdx(platform, 0);
  using Vec= alpaka::Vec<Dim,Idx;
  using BufAcc = alpaka::Buf<DevAcc, Data, Dim1D, Idx>;
  BufAcc bufAcc;
  T * const pAcc;
  Vec const extent1D;
  auto init(Idx N)->void{
    extend1D=Vec(N);
    bufAcc=alpaka::allocBuf<Data, Idx>(devAcc,extent1D);
    pAcc=alpaka::getPtrNative(bufAcc);
  }
  Vector_h() {}
  inline Vector_h(Idx N):size(N){ init(N);};
  Vector_h(Idx N,T v):size(N){
    init(N);
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, T, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost,extent1D));
    T* const pBufHostA(alpaka::getPtrNative(bufHostA));
    for(Idx i(0); i < size; ++i)pBufHostA[i]=v;
    alpaka_copy(bufHostA,this->bufAcc,extend1D);
  }
  Vector_h(const Vector_h<T>& source):size(source.size()){
    init(size);
    alpaka_copy(source.bufHost,this->bufAcc,extend1D);
  }
  Vector_h(const Vector_d_<T> &source):size(source.size()){
    init(size);
    alpaka_copy(source.bufAcc,this->bufAcc,extend1D);//copy device to host
  }
  Vector_h<T>& operator=(const Vector_h<T> &a) { 
    if(this.size()!=a.size())return *this;
    alpaka_copy(a.bufAcc,this->bufAcc,extend1D);
    return *this;
  }
  Vector_h<T> &operator=(const Vector_d<T> &a){
    if(this.size()!=a.size())return *this;
    alpaka_copy(a.bufHost,this->bufAcc,extend1D);
    return *this;
  }
  auto size()->Idx{
    return this->size;
  }
  inline T* raw() { 
    return pAcc;
  } 
  inline const T* raw() const { 
    return pAcc;
  } 
}
#endif

