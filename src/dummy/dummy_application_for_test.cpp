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
    std::uniform_int_distribution<int> di_int(0, iterations);

    
    std::vector<double> s(size);
    std::vector<int> faulty_iterations(20);
    std::generate(faulty_iterations.begin(), faulty_iterations.end(), [&]{return di_int(random_engine);});
    
    std::generate(s.begin(), s.end(), [&]{return di(random_engine); });
    
    auto s_gold = s;

    std::cout << dummy_log << std::endl;
    std::cout << "Errors at iterations: ";
    for(auto i : faulty_iterations) std::cout << i << " ";
    std::cout << std::endl;

    duration<double, std::milli> ms_double;
     
     for(auto it = 0lu; it < iterations; it++){
         auto t1 = high_resolution_clock::now();
         dummy_log.start_iteration();
         std::sort(s.begin(), s.end());
         dummy_log.end_iteration();
         s = s_gold;
         auto t2 = high_resolution_clock::now();
         ms_double = t2 - t1;
         std::cout <<  "Iteration " << it << " " << ms_double.count() << "ms" << std::endl;
         
         if(std::find(begin(faulty_iterations), end(faulty_iterations), it) != std::end(faulty_iterations)){
             dummy_log.log_error_detail("Error at iteration " + std::to_string(it));
             dummy_log.update_errors();
         }
    }
    
}
