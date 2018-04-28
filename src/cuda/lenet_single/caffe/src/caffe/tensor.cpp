#include <memory>
#include <vector>

#include "caffe/tensor.hpp"

namespace caffe {

Tensor::Tensor(Type dtype)
    : type_(dtype),
      synced_arrays_(make_shared<vector<shared_ptr<SyncedMemory>>>(Type_ARRAYSIZE)),
      count_(0) {}

const shared_ptr<SyncedMemory>& Tensor::synced_mem() const {
  const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
  CHECK(mem);
  CHECK(mem->is_valid());
  return mem;
}

shared_ptr<SyncedMemory>& Tensor::mutable_synced_mem(bool flush) {
  shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
  // We are about to assign something here, thus validate in advance:
  if (mem) {
    mem->validate();
  }
  if (flush) {  // Example: Reshape
    invalidate_others();
  }
  return mem;
}

void Tensor::invalidate_others() {
  CHECK(synced_arrays_->at(type_)->is_valid()) << "No valid arrays left";
  for (size_t i = 0; i < synced_arrays_->size(); ++i) {
    if (i != type_) {
      shared_ptr<SyncedMemory>& mem = synced_arrays_->at(i);
      if (mem) {
        mem->invalidate();
      }
    }
  }
}

void Tensor::Reshape(int count) {
  shared_ptr<SyncedMemory>& mem = mutable_synced_mem(false);
  if (!mem || count != count_) {
    mem = make_shared<SyncedMemory>(even(count) * tsize(type_));
    count_ = count;
  }
}

void Tensor::convert(Type new_type) {
  if (new_type == type_) {
    return;
  }

  const shared_ptr<SyncedMemory>& current_mem = synced_mem();
  shared_ptr<SyncedMemory>& new_mem = synced_arrays_->at(new_type);

  if (!new_mem || !new_mem->is_valid()) {
    const std::size_t new_cap = even(count_) * tsize(new_type);
    if (!new_mem || new_mem->size() != new_cap) {
      new_mem = make_shared<SyncedMemory>(new_cap);
    }
    const bool data_gpu = is_gpu_head();
    if (current_mem->head() != SyncedMemory::UNINITIALIZED) {
      copy_helper(data_gpu, count_,
          data_gpu ? current_mem->gpu_data() : current_mem->cpu_data(),
          type_,
          data_gpu ? new_mem->mutable_gpu_data() : new_mem->mutable_cpu_data(),
          new_type);
    }
  } // we just trust its current status otherwise
  type_ = new_type;
  new_mem->validate();
}

void Tensor::copy_helper(bool use_gpu, int count, const void* p_src, Type src_type,
    void* p_dst, Type dst_type) {
  bool failed = false;
  if (is_type<float>(src_type)) {
    if (is_type<float>(dst_type)) {  // FP32 -> FP32
      caffe_copy(count, static_cast<const float*>(p_src),
          static_cast<float*>(p_dst));
    } else if (is_type<float16>(dst_type)) {  // FP32 -> FP16
      caffe_convert(use_gpu, count, static_cast<const float*>(p_src),
          static_cast<float16*>(p_dst));
    } else if (is_type<double>(dst_type)) {  // FP32 -> FP64
      caffe_convert(use_gpu, count, static_cast<const float*>(p_src),
          static_cast<double*>(p_dst));
    } else {
      failed = true;
    }
  } else if (is_type<float16>(src_type)) {
    if (is_type<float>(dst_type)) {  // FP16 -> FP32
      caffe_convert(use_gpu, count, static_cast<const float16*>(p_src),
          static_cast<float*>(p_dst));
    } else if (is_type<float16>(dst_type)) {  // FP16 -> FP16
      caffe_copy(count, static_cast<const float16*>(p_src),
          static_cast<float16*>(p_dst));
    } else if (is_type<double>(dst_type)) {  // FP16 -> FP64
      caffe_convert(use_gpu, count, static_cast<const float16*>(p_src),
          static_cast<double*>(p_dst));
    } else {
      failed = true;
    }
  } else if (is_type<double>(src_type)) {
    if (is_type<float>(dst_type)) {  // FP64 -> FP32
      caffe_convert(use_gpu, count, static_cast<const double*>(p_src),
          static_cast<float*>(p_dst));
    } else if (is_type<float16>(dst_type)) {  // FP64 -> FP16
      caffe_convert(use_gpu, count, static_cast<const double*>(p_src),
          static_cast<float16*>(p_dst));
    } else if (is_type<double>(dst_type)) {  // FP64 -> FP64
      caffe_copy(count, static_cast<const double*>(p_src),
          static_cast<double*>(p_dst));
    } else {
      failed = true;
    }
  } else if (is_type<unsigned int>(src_type) && is_type<unsigned int>(dst_type)) {
    caffe_copy(count, static_cast<const unsigned int*>(p_src), static_cast<unsigned int*>(p_dst));
  } else if (is_type<int>(src_type) && is_type<int>(dst_type)) {
    caffe_copy(count, static_cast<const int*>(p_src), static_cast<int*>(p_dst));
  } else {
    failed = true;
  }
  if (failed) {
    LOG(FATAL) << "Failed to copy elements of type " << Type_Name(src_type) << " to "
               << Type_Name(dst_type);
  }
}

void Tensor::scale(float scale, void* handle) {
  shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
  if (is_gpu_head()) {
    cublasHandle_t cublas_handle =
        handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
    gpu_scal(count_, type_, mem->mutable_gpu_data(), scale, cublas_handle);
  } else {
    cpu_scal(count_, type_, mem->mutable_cpu_data(), scale);
  }
}

void Tensor::set(float value) {
  shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
  if (is_gpu_head()) {
    void* data = mem->mutable_gpu_data();
    if (is_type<float>(type_)) {
      caffe_gpu_set(count_, value, static_cast<float*>(data));
    } else if (is_type<float16>(type_)) {
      caffe_gpu_set(count_, static_cast<float16>(value), static_cast<float16*>(data));
    } else if (is_type<double>(type_)) {
      caffe_gpu_set(count_, static_cast<double>(value), static_cast<double*>(data));
    } else {
      LOG(FATAL) << "Unsupported data type: " << Type_Name(type_);
    }
  } else {
    void* data = mem->mutable_cpu_data();
    if (is_type<float>(type_)) {
      caffe_set(count_, value, static_cast<float*>(data));
    } else if (is_type<float16>(type_)) {
      caffe_set(count_, static_cast<float16>(value), static_cast<float16*>(data));
    } else if (is_type<double>(type_)) {
      caffe_set(count_, static_cast<double>(value), static_cast<double*>(data));
    } else {
      LOG(FATAL) << "Unsupported data type: " << Type_Name(type_);
    }
  }
}

float Tensor::asum(int group) const {
  const shared_ptr<SyncedMemory>& mem = synced_mem();
  float asum = 0.F;
  if (!mem || count_ <= 0) {
    return asum;
  }
  if (is_gpu_head()) {
    if (is_type<float>(type_)) {
      caffe_gpu_asum(count_, static_cast<const float*>(mem->gpu_data()), &asum, group);
    } else if (is_type<float16>(type_)) {
      caffe_gpu_asum(count_, static_cast<const float16*>(mem->gpu_data()), &asum, group);
    } else if (is_type<double>(type_)) {
      caffe_gpu_asum(count_, static_cast<const double*>(mem->gpu_data()), &asum, group);
    } else {
      LOG(FATAL) << "Unknown data type: " << Type_Name(type_);
    }
    return asum;
  }
  if (is_type<float>(type_)) {
    asum = caffe_cpu_asum(count_, static_cast<const float*>(mem->cpu_data()));
  } else if (is_type<float16>(type_)) {
    asum = caffe_cpu_asum(count_, static_cast<const float16*>(mem->cpu_data()));
  } else if (is_type<double>(type_)) {
    asum = caffe_cpu_asum(count_, static_cast<const double*>(mem->cpu_data()));
  } else {
    LOG(FATAL) << "Unknown data type: " << Type_Name(type_);
  }
  return asum;
}

float Tensor::amax(int group) const {
  const shared_ptr<SyncedMemory>& mem = synced_mem();
  float amax = 0.F;
  if (!mem || count_ <= 0) {
    return amax;
  }
  if (is_gpu_head()) {
    if (is_type<float>(type_)) {
      caffe_gpu_amax(count_, static_cast<const float*>(mem->gpu_data()), &amax, group);
    } else if (is_type<float16>(type_)) {
      caffe_gpu_amax(count_, static_cast<const float16*>(mem->gpu_data()), &amax, group);
    } else if (is_type<double>(type_)) {
      caffe_gpu_amax(count_, static_cast<const double*>(mem->gpu_data()), &amax, group);
    } else {
      LOG(FATAL) << "Unknown data type: " << Type_Name(type_);
    }
    return amax;
  }
  if (is_type<float>(type_)) {
    amax = caffe_cpu_amax(count_, static_cast<const float*>(mem->cpu_data()));
  } else if (is_type<float16>(type_)) {
    amax = caffe_cpu_amax(count_, static_cast<const float16*>(mem->cpu_data()));
  } else if (is_type<double>(type_)) {
    amax = caffe_cpu_amax(count_, static_cast<const double*>(mem->cpu_data()));
  } else {
    LOG(FATAL) << "Unknown data type: " << Type_Name(type_);
  }
  return amax;
}

float Tensor::sumsq(int group) const {
  const shared_ptr<SyncedMemory>& mem = synced_mem();
  float sumsq = 0.F;
  if (!mem || count_ <= 0) {
    return sumsq;
  }
  if (is_gpu_head()) {
    if (is_type<float>(type_)) {
      caffe_gpu_sumsq(count_, static_cast<const float*>(mem->gpu_data()), &sumsq, group);
    } else if (is_type<float16>(type_)) {
      caffe_gpu_sumsq(count_, static_cast<const float16*>(mem->gpu_data()), &sumsq, group);
    } else if (is_type<double>(type_)) {
      caffe_gpu_sumsq(count_, static_cast<const double*>(mem->gpu_data()), &sumsq, group);
    } else {
      LOG(FATAL) << "Unknown data type: " << Type_Name(type_);
    }
    return sumsq;
  }
  if (is_type<float>(type_)) {
    sumsq = caffe_cpu_sumsq(count_, static_cast<const float*>(mem->cpu_data()));
  } else if (is_type<float16>(type_)) {
    sumsq = caffe_cpu_sumsq(count_, static_cast<const float16*>(mem->cpu_data()));
  } else if (is_type<double>(type_)) {
    sumsq = caffe_cpu_sumsq(count_, static_cast<const double*>(mem->cpu_data()));
  } else {
    LOG(FATAL) << "Unknown data type: " << Type_Name(type_);
  }
  return sumsq;
}

void Tensor::cpu_scal(int count, Type dtype, void* data, float scal) {
  if (is_type<float>(dtype)) {
    caffe_scal(count, scal, static_cast<float*>(data));
  } else if (is_type<float16>(dtype)) {
    caffe_scal(count, static_cast<float16>(scal), static_cast<float16*>(data));
  } else if (is_type<double>(dtype)) {
    caffe_scal(count, static_cast<double>(scal), static_cast<double*>(data));
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

void Tensor::gpu_scal(int count, Type dtype, void* data, float scal, cublasHandle_t cublas_handle) {
  if (is_type<float>(dtype)) {
    caffe_gpu_scal(count, scal, static_cast<float*>(data), cublas_handle);
  } else if (is_type<float16>(dtype)) {
    caffe_gpu_scal(count, static_cast<float16>(scal), static_cast<float16*>(data), cublas_handle);
  } else if (is_type<double>(dtype)) {
    caffe_gpu_scal(count, static_cast<double>(scal), static_cast<double*>(data), cublas_handle);
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(dtype);
  }
}

size_t Tensor::gpu_memory_use(bool own_only) const {
  size_t ret = 0ULL;
  for (size_t i = 0; i < synced_arrays_->size(); ++i) {
    if (synced_arrays_->at(i)) {
      ret += synced_arrays_->at(i)->gpu_memory_use(own_only);
    }
  }
  return ret;
}

std::string Tensor::to_string(int indent) const {  // debug helper
  const std::string idt(indent, ' ');
  std::ostringstream os;
  os << idt << "Tensor " << this << ", count_: " << count_ << ", type: " << Type_Name(type_)
     << std::endl;
  os << idt << "synced_arrays_: " << synced_arrays_.get();
  for (size_t i = 0; i < synced_arrays_->size(); ++i) {
    os << " " << synced_arrays_->at(i).get();
  }
  os << std::endl;

  for (size_t i = 0; i < synced_arrays_->size(); ++i) {
    if (synced_arrays_->at(i)) {
      os << idt << Type_Name((Type) i);
      if (type_ == i) {
        os << " ***Current***";
      }
      os << std::endl << idt << " data:" << std::endl;
      os << synced_arrays_->at(i)->to_string(indent + 2, (Type) i);

//      if (synced_arrays_->at(i)->is_valid()) {
//        os << idt << "ASUM: " << asum(0) << std::endl;
//      } else {
//        os << idt << "NOT VALID" << std::endl;
//      }
    }
  }
  return os.str();
}

}  // namespace caffe
