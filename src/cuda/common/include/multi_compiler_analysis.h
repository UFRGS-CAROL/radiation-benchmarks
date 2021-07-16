//
// Created by fernando on 9/24/20.
//

#ifndef COMMON_MULTI_COMPILER_ANALYSIS_H
#define COMMON_MULTI_COMPILER_ANALYSIS_H

#include <string>

namespace rad {
    static void check_nvcc(){
#ifndef __NVCC__
#error "Cannot use this library without NVCC, please add the reader only on files that will be compiled by NVCC"
#endif
    }
    static std::string get_cuda_cc_version() {
        check_nvcc();
        long version_major, version_minor;
#ifdef __CUDACC_VER_MAJOR__
        version_major = __CUDACC_VER_MAJOR__;
    version_minor = __CUDACC_VER_MINOR__;
#elif defined(__CUDACC_VER__)
        version_major = __CUDACC_VER__ / 10000;
    version_minor = __CUDACC_VER__ % 10000;
#else
#warning "Neither __CUDACC_VER__ nor __CUDACC_VER_MAJOR/MINOR__ are defined, using 7 and 0 as major and minor"
        version_major = 7;
        version_minor = 0;
#endif
        std::string ret = "MAJOR_" + std::to_string(version_major);
        ret += "_MINOR_" + std::to_string(version_minor);

        return ret;
    }

#define XSTR(x) #x
#define STRING(x) XSTR(x)

    std::string extract_nvcc_opt_flags_str() {
        check_nvcc();

        std::string opt_flags;
#ifdef NVCCOPTFLAGS
        opt_flags = STRING(NVCCOPTFLAGS);
#else
        opt_flags = "";
#endif
        return opt_flags;
    }

    std::string get_multi_compiler_header() {
#ifndef __NVCC__
#error "Calling get_multi_compiler_header() from a non cuda file, please call it from NVCC source files"
#endif
    	std::string test_info = " nvcc_version:" + get_cuda_cc_version();
    	test_info += " nvcc_optimization_flags:" + extract_nvcc_opt_flags_str();
    	return test_info;
    }

}


#endif //COMMON_MULTI_COMPILER_ANALYSIS_H
