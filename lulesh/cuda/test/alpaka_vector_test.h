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
    Idx numElems(300);
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
    int resize_empty_device_vector(){
        try{
        Vector_d<Real_t> d;
        d.resize(numElems);
        }catch(...){
            cout<<"error resizing empty device Vector"<<endl;
            return failure;
        }
        return success;
    }
    int create_device_vector(){
        try{
        Vector_d<Real_t> d(numElems,.0);
        }catch(std::exception& e){
            cout<<"error creating device Vector"<<endl;
            std::cout << e.what() << std::endl;
            return failure;
        }
        return success;
    }
    int fill_device_vector(){
        try{
            Vector_d<Real_t> d(numElems);
            d.fill(.0);
        }catch(std::exception& e){
            cout<<"error filling device Vector"<<endl;
            std::cout << e.what() << std::endl;
            return failure;
        }
        return success;
    }
    int resize_device_vector(){
        try{
        Vector_d<Real_t> d(numElems,.0);
        d.resize(numElems*2);
        }catch(...){
            cout<<"error resizing non-empty device Vector"<<endl;
            return failure;
        }
        return success;
    }
    int resize_fill_device_vector(){
        try{
        Vector_d<Real_t> d;
        d.resize(numElems);
        d.fill(.0);
        }catch(std::exception& e){
            cout<<"error filling resized device Vector"<<endl;
            std::cout << e.what() << std::endl;
            return failure;
        }
        return success;
    }

    int create_empty_host_vector(){
        try{
            
            Vector_h<Real_t> h;
        }catch(...){
            cout<<"error creating empty host Vector"<<endl;
            return failure;
        }
        return success;
    };
    int create_host_vector(){
        try{
            
            Vector_h<Real_t> h(numElems,.0);
        }catch(...){
            cout<<"error creating host Vector"<<endl;
            return failure;
        }
        return success;
    };
    int copy_to_device_and_back(){
        try{

            Vector_d<Real_t> deviceVec(numElems,.0);

            Vector_h<Real_t> hostVec(numElems,1.1);

            deviceVec=hostVec;//copy to device

            Vector_h<Real_t> test(numElems,.0);

            test=deviceVec;//copy from device

            for(Idx i(0);i<numElems;i++)if(test[i]!=1.1){
                cout<<"wrong result when trying to copy from and back to device "<<test[i]<<endl;
                return failure;
            }

        }catch(std::exception& e){
                cout<<"copy to and from device failed"<<endl;
                std::cout << e.what() << std::endl;
                return failure;
        }
        return success;
    }
    void tests_host(int &failed){
        failed+=test::create_empty_host_vector();
        failed+=test::create_host_vector();
    }
    void tests_device(int &failed){
        failed+=test::create_empty_device_vector();
        failed+=test::resize_empty_device_vector();
        failed+=test::create_device_vector();
        failed+=test::fill_device_vector();
        failed+=test::resize_device_vector();
        failed+=test::resize_fill_device_vector();
        failed+=test::copy_to_device_and_back();
    };
    int test_main(){
        int failed=0;
        tests_host(failed);
        tests_device(failed);
        if(!failed){
            //cout<<"passed All tests"<<endl;
            return success;
        }
        else {
            cout<<failed<<" tests failed"<<endl;
            return failure;
        }
    };
}
