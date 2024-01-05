#pragma once
#include "../src/vector.h"
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#define ALPAKA
#define Real_t double
namespace test{
    using std::cout;
    using std::endl;
    using Idx=std::size_t;
    #define failure 1
    #define success 0
    int test_resize__fill_device_vector(){
        try{
        Vector_d<Real_t> d;
        d.resize(300);
        d.fill(.0);
        }catch(std::exception& e){
            cout<<"error filling resized device Vector"<<endl;
            std::cout << e.what() << std::endl;
            return failure;
        }
        return success;
    }
    int test_resize_device_vector(){
        try{
        Vector_d<Real_t> d;
        d.resize(300);
        }catch(...){
            cout<<"error resizing device Vector"<<endl;
            return failure;
        }
        return success;
    }
    int create_host_vector(int N){
        try{
            
            Vector_h<Real_t> h(300,.0);
        }catch(...){
            cout<<"error creating host Vector"<<endl;
            return failure;
        }
        return success;
    };
    int create_empty_device_vector(){
        try{
            Vector_d<Real_t> d;
        }
        catch(...){
            cout<<"erro creating empty device Vector"<<endl;
            return failure;
        }
        return success;
    }
    void tests(int &failed){
        failed+=test::create_host_vector(300);
        failed+=test::create_empty_device_vector();
        failed+=test::test_resize_device_vector();
        failed+=test::test_resize__fill_device_vector();
    };
    int test_main(){
        int failed=0;
        tests(failed);
        if(!failed){
            cout<<"passed All tests"<<endl;
            return success;
        }
        else {
            cout<<failed<<" tests failed"<<endl;
            return failure;
        }
    };
}