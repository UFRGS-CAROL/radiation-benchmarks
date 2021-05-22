#include <algorithm>
#include <functional>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

//include from cuda/common
#include <include/generic_log.h>

int main(){
    constexpr auto size = 2048 * 2048lu;
    constexpr auto iterations = 1000000lu;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    
    rad::Log dummy_log("DummyApp", "iterations:" + std::to_string(iterations) + " size:" + std::to_string(size), 10);
    
    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_real_distribution<double> di(-100, 100);

    
    std::vector<double> s(size);
    
    std::generate(s.begin(), s.end(), [&]{return di(random_engine); });


    duration<double, std::milli> ms_double;
     
     for(auto it = 0lu; it < iterations; it++){
         auto t1 = high_resolution_clock::now();
         dummy_log.start_iteration();
         std::sort(s.begin(), s.end());
         dummy_log.end_iteration();
         auto t2 = high_resolution_clock::now();
         ms_double = t2 - t1;
         std::cout <<  "Iteration " << it << " " << ms_double.count() << "ms" << std::endl;

    }
    
}
