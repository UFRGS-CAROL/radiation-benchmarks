#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"

const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 1)
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
    default:
      break;
  }
  return "Unknown cudnn status";
}


namespace caffe {
namespace cudnn {

float dataType<float>::oneval = 1.0f;
float dataType<float>::zeroval = 0.0f;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);

float dataType<float16>::oneval = 1.0f;
float dataType<float16>::zeroval = 0.0f;
const void* dataType<float16>::one =
    static_cast<void *>(&dataType<float16>::oneval);
const void* dataType<float16>::zero =
    static_cast<void *>(&dataType<float16>::zeroval);

}  // namespace cudnn
}  // namespace caffe
#endif
