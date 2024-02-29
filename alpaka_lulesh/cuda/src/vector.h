#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#define ALPAKA

template <class T> class Vector_h;

template <class T> class Vector_d;

#define ALPAKA

// host vector
#ifdef ALPAKA
using Dim1 = alpaka::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::ExampleDefaultAcc<Dim1, Idx>;
using DevAcc = alpaka::Dev<Acc>;
template <typename TSource, typename Ttarget, typename Textend>
void alpaka_copy(TSource &source, Ttarget &target, Textend &extend) {

  using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;
  auto const platform = alpaka::Platform<Acc>{};
  auto const devAcc = alpaka::getDevByIdx(platform, 0);
  QueueAcc queue(devAcc);
  // alpaka::memcpy(queue, target,source,extend);
  alpaka::memcpy(queue, target, source, extend);//big buffer on device-->copy--> small buffer on device[1] --> host buffer
  alpaka::wait(queue);
};
template <class T> class Vector_h {
  Idx size_param;

  using DevHost = alpaka::DevCpu;

  using Vec = alpaka::Vec<Dim1, Idx>;
  using Buffer = alpaka::Buf<alpaka::DevCpu, T, Dim1, Idx>;
  std::shared_ptr<Buffer> bufHost;
  T *pHost;
  Vec extent1D;

public:
  Vector_h() : size_param(0), extent1D(Vec(0)) {
    bufHost = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    this->pHost = alpaka::getPtrNative(*this->bufHost);
  }

  Vector_h(Idx N) : extent1D(Vec(N)), size_param(N) {
    bufHost = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    this->pHost = alpaka::getPtrNative(*this->bufHost);
  };
  // memory error on calling bufHost(alpaka::allocBuf<T,
  // Idx>(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0),extent1D)
  Vector_h(Idx N, T v) : extent1D(Vec(N)), size_param(N) {
    bufHost = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    this->pHost = alpaka::getPtrNative(*this->bufHost);
    for (Idx i(0); i < N; i++)
      pHost[i] = v;
  }
  Vector_h(Vector_h<T> &source)
      : extent1D(Vec(source.size())), size_param(source.size()) {
    bufHost = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    this->pHost = alpaka::getPtrNative(*this->bufHost);
    T *const pSourceHost = source.raw();
    for (Idx i(0); i < source.size(); i++) {
      pHost[i] = source[i];
    }
  }
  Vector_h(Vector_d<T> &source)
      : extent1D(Vec(source.size())), size_param(source.size()) {
    bufHost = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    this->pHost = alpaka::getPtrNative(*this->bufHost);
    alpaka_copy(source.getBuf(), *this->bufHost,
                this->extent1D); // copy device to host
  }
  Vector_h<T> &operator=(Vector_h<T> &a) {
    if (this->size_param != a.size()) {
      this->resize(a.size());
    }
    for (Idx i(0); i < this->size_param; i++)
      pHost[i] = a.pHost[i];
    return *this;
  }
  Vector_h<T> &operator=(Vector_d<T> &a) {
    if (this->size_param != a.size()) {
      this->resize(a.size());
    }
    alpaka_copy(a.getBuf(), *this->bufHost,
                this->extent1D); // copy device to source
    return *this;
  };
  Idx size() { return this->size_param; }

  template <typename TIDX> void resize(TIDX N) {
    this->size_param = static_cast<Idx>(N);
    this->extent1D = Vec(N);
    this->bufHost = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    this->pHost = alpaka::getPtrNative(*this->bufHost);
  }
  Buffer &getBuf() { return *bufHost; }
  void fill(T v) {
    for (Idx i(0); i < this->size_param; i++)
      pHost[i] = v;
  }

  template <typename TIDX> T &operator[](TIDX index) { return pHost[index]; }
  inline T *raw() { return pHost; }
};

// device vector
template <class T> class Vector_d {

  Idx size_param;
  using Acc = alpaka::ExampleDefaultAcc<Dim1, Idx>;
  using DevAcc = alpaka::Dev<Acc>;

  using Vec = alpaka::Vec<Dim1, Idx>;
  using Buffer = alpaka::Buf<DevAcc, T, Dim1, Idx>;
  std::shared_ptr<Buffer> bufAcc;
  T *pAcc;
  Vec extent1D;
  // TODO
public:
  Vector_d() : size_param(0), extent1D(Vec(0)) {
    this->bufAcc = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0), this->extent1D));
    this->pAcc = alpaka::getPtrNative(*this->bufAcc);
  };

  Vector_d(Idx N) : size_param(N), extent1D(Vec(N)) {
    this->bufAcc = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0), this->extent1D));

    this->pAcc = alpaka::getPtrNative(*this->bufAcc);
  };

  Vector_d(Idx N, T v) : size_param(N), extent1D(Vec(N)) {
    this->bufAcc = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0), this->extent1D));
    this->pAcc = alpaka::getPtrNative(*this->bufAcc);
    this->fill(v);
  }

  Vector_d(Vector_h<T> &source)
      : size_param(source.size()), extent1D(Vec(source.size())) {
    this->bufAcc = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0), this->extent1D));
    this->pAcc = alpaka::getPtrNative(*this->bufAcc);
    alpaka_copy(source.getBuf(), *this->bufAcc, this->extent1D);
  }

  Vector_d(Vector_d<T> &source)
      : size_param(source.size()), extent1D(Vec(source.size())) {
    this->bufAcc = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0), this->extent1D));
    this->pAcc = alpaka::getPtrNative(*this->bufAcc);
    alpaka_copy(source.getBuf(), *this->bufAcc,
                this->extent1D); // copy device to host
  }

  Vector_d<T> &operator=(Vector_h<T> &a) {
    if (this->size_param != a.size()) {
      this->resize(a.size());
    }
    alpaka_copy(a.getBuf(), *this->bufAcc, this->extent1D);
    return *this;
  }
  Vector_d<T> &operator=(Vector_d<T> &a) {
    if (this->size_param != a.size()) {
      this->resize(a.size());
    }
    alpaka_copy(a.getBuf(), *this->bufAcc, this->extent1D);
    return *this;
  }

  template <typename TIDX> T accessIndex(TIDX index) {
    Vector_h host(*this);

    return host[index];
  }
  Buffer &getBuf() { return *bufAcc; }
  template <typename TIDX> void resize(TIDX N) {
    this->size_param = static_cast<Idx>(N);
    this->extent1D = Vec(N);
    this->bufAcc = std::make_shared<Buffer>(alpaka::allocBuf<T, Idx>(
        alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0), this->extent1D));
    this->pAcc = alpaka::getPtrNative(*this->bufAcc);
  }
  void fill(T v) {
    this->extent1D = Vec(this->size_param);
    using BufHost = alpaka::Buf<alpaka::DevCpu, T, Dim1, Idx>;
    std::shared_ptr<BufHost> hostbuf =
        std::make_shared<BufHost>(alpaka::allocBuf<T, Idx>(
            alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), this->extent1D));
    T *hostp = alpaka::getPtrNative(*hostbuf);

    for (int i = 0; i < this->size_param; i++)
      hostp[i] = v;
    alpaka_copy(*hostbuf, *this->bufAcc, this->extent1D);
    // free(hostVec);
  }
  /*
    changes Values in a certain range warning this operation copies the current
    values from the device inserts the new values and copies back to the device
    this is slow and not recommended use a kernel instead
  */
  void changeValue(Idx from, Idx numelems, T *vec) {
    Vector_h<T> hostvec(*this);
    for (Idx i(from); i < from + numelems && i < this->size_param; i++)
      hostvec[i] = vec[i];
    alpaka_copy(hostvec.getBuf(), *this->bufAcc, this->extent1D);
  }
  auto size() -> Idx { return this->size_param; }
  inline T *raw() { return pAcc; }
};
#else
template <class T> class Vector_h : public thrust::host_vector<T> {
public:
  // Constructors
  Vector_h() {}
  inline Vector_h(int N) : thrust::host_vector<T>(N) {}
  inline Vector_h(int N, T v) : thrust::host_vector<T>(N, v) {}
  inline Vector_h(const Vector_h<T> &a) : thrust::host_vector<T>(a) {}
  inline Vector_h(const Vector_d<T> &a) : thrust::host_vector<T>(a) {}

  template <typename OtherVector> inline void copy(const OtherVector &a) {
    this->assign(a.begin(), a.end());
  }

  inline Vector_h<T> &operator=(const Vector_h<T> &a) {
    copy(a);
    return *this;
  }
  inline Vector_h<T> &operator=(const Vector_d<T> &a) {
    copy(a);
    return *this;
  }

  inline T *raw() {
    if (bytes() > 0)
      return thrust::raw_pointer_cast(this->data());
    else
      return 0;
  }

  inline const T *raw() const {
    if (bytes() > 0)
      return thrust::raw_pointer_cast(this->data());
    else
      return 0;
  }

  inline size_t bytes() const { return this->size() * sizeof(T); }
};
template <class T> class Vector_d : public thrust::device_vector<T> {
public:
  Vector_d() {}
  inline Vector_d(int N) : thrust::device_vector<T>(N) {}
  inline Vector_d(int N, T v) : thrust::device_vector<T>(N, v) {}
  inline Vector_d(const Vector_d<T> &a) : thrust::device_vector<T>(a) {}
  inline Vector_d(const Vector_h<T> &a) : thrust::device_vector<T>(a) {}

  template <typename OtherVector> inline void copy(const OtherVector &a) {
    this->assign(a.begin(), a.end());
  }

  inline Vector_d<T> &operator=(const Vector_d<T> &a) {
    copy(a);
    return *this;
  }
  inline Vector_d<T> &operator=(const Vector_h<T> &a) {
    copy(a);
    return *this;
  }

  inline T *raw() {
    if (bytes() > 0)
      return thrust::raw_pointer_cast(this->data());
    else
      return 0;
  }

  inline const T *raw() const {
    if (bytes() > 0)
      return thrust::raw_pointer_cast(this->data());
    else
      return 0;
  }
  inline size_t bytes() const { return this->size() * sizeof(T); }
};

#endif
