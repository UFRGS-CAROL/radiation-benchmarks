#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Gtype, typename Wtype>
void adagrad_reg_update_and_clear_gpu(int N,
    Gtype *g, Wtype *w, Wtype *h,
    float delta, float local_rate, const std::string& regularization_type, float local_decay,
    void *handle, bool clear_grads);

template<typename Dtype>
float AdaGradSolver<Dtype>::ComputeUpdateValue(int param_id, void *handle, float rate,
    bool clear_grads) {
  shared_ptr<Blob> param = this->net_->learnable_params()[param_id];

  float wgrad_sq = 1.F;  // stub

  shared_ptr<TBlob<Dtype>> history = this->history_[param_id];
  shared_ptr<TBlob<Dtype>> update = this->update_[param_id];
  const vector<float> &net_params_lr = this->net_->params_lr();
  float delta = std::max(this->param_.delta(), 0.001f);
  float local_rate = rate * net_params_lr[param_id];
  if (Caffe::mode() == Caffe::CPU) {
    // compute square of gradient in update
    caffe_powx<Dtype>(param->count(), param->cpu_diff<Dtype>(), Dtype(2.F),
        update->mutable_cpu_data());
    // update history
    caffe_add<Dtype>(param->count(), update->cpu_data(), history->cpu_data(),
        history->mutable_cpu_data());
    // prepare update
    caffe_powx<Dtype>(param->count(), history->cpu_data(), Dtype(0.5),
        update->mutable_cpu_data());
    caffe_add_scalar<Dtype>(param->count(), delta, update->mutable_cpu_data());
    caffe_div<Dtype>(param->count(), param->cpu_diff<Dtype>(), update->cpu_data(),
        update->mutable_cpu_data());
    // scale and copy
    caffe_cpu_axpby<Dtype>(param->count(), local_rate, update->cpu_data(), Dtype(0.),
        param->mutable_cpu_diff<Dtype>());

    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
    const std::string& regularization_type = this->param_.regularization_type();
    const float decay = this->local_decay(param_id);
    const Type gtype = param->diff_type();
    if (gtype == tp<float16>()) {
      adagrad_reg_update_and_clear_gpu<float16, Dtype>(param->count(),
          param->mutable_gpu_diff<float16>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          delta, local_rate, regularization_type, decay,  handle, clear_grads);
    } else if (gtype == tp<float>()) {
      adagrad_reg_update_and_clear_gpu<float, Dtype>(param->count(),
          param->mutable_gpu_diff<float>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          delta, local_rate, regularization_type, decay,  handle, clear_grads);
    } else if (gtype == tp<double>()) {
      adagrad_reg_update_and_clear_gpu<double, Dtype>(param->count(),
          param->mutable_gpu_diff<double>(),
          param->mutable_gpu_data<Dtype>(),
          history->mutable_gpu_data(),
          delta, local_rate, regularization_type, decay,  handle, clear_grads);
    } else {
      LOG(FATAL) << "Gradient type " << Type_Name(gtype) << " is not supported";
    }
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return wgrad_sq;
}

INSTANTIATE_CLASS(AdaGradSolver);
REGISTER_SOLVER_CLASS(AdaGrad);

}  // namespace caffe
