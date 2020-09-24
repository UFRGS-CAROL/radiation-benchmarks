//
// Created by fernando on 9/24/20.
//

#ifndef COMMON_MULTI_COMPILER_ANALYSIS_H
#define COMMON_MULTI_COMPILER_ANALYSIS_H

#include <string>

namespace rad {
    static std::string get_cuda_cc_version() {
        long version_major, version_minor;


#ifdef __CUDACC_VER_MAJOR__
        version_major = __CUDACC_VER_MAJOR__;
    version_minor = __CUDACC_VER_MINOR__;
#elif defined(__CUDACC_VER__)
        version_major = __CUDACC_VER__ / 10000;
    version_minor = __CUDACC_VER__ % 10000;
#else
#warning "Neither __CUDACC_VER__ or __CUDACC_VER_MAJOR/MINOR__ are defined, using 7 and 0 as major and minor"
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
        std::string opt_flags;
#ifdef NVCCOPTFLAGS
        opt_flags = STRING(NVCCOPTFLAGS);
#else
        opt_flags = "";
#endif
        return opt_flags;
    }

}


#endif //COMMON_MULTI_COMPILER_ANALYSIS_H
