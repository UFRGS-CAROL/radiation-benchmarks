#include <vector>
#include <device_launch_parameters.h>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/half.cuh"

namespace caffe {

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

template <typename Ftype, typename Btype>
void ConcatLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;

  if (bottom.size() == 1) {
    return;
  }
  for (int i = 0; i < bottom.size(); ++i) {
    bottom_data = bottom[i]->gpu_data<Ftype>();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    if (tp<Ftype>() == FLOAT16) {
      Concat<half>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream() >>> (
          nthreads, reinterpret_cast<const half*>(bottom_data), kForward, num_concats_,
          concat_input_size_, top_concat_axis, bottom_concat_axis, offset_concat_axis,
          reinterpret_cast<half*>(top_data));
    } else {
      Concat<Ftype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream() >>> (
          nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
              top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    }
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    offset_concat_axis += bottom_concat_axis;
  }
}

template <typename Ftype, typename Btype>
void ConcatLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
  if (bottom.size() == 1) {
    return;
  }
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
      Concat<Btype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ConcatLayer);

}  // namespace caffe
