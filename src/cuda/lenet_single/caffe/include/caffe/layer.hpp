#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/device_alternate.hpp"

#if defined(USE_CUDNN)

#include "caffe/util/cudnn.hpp"

#endif

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

class Solver;
class Net;
class Flag;

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
class LayerBase {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit LayerBase(const LayerParameter& param, int prec = 0)
      : layer_param_(param),
        debug_(false),
        fm_by_user_(false),
        bm_by_user_(false),
        parent_net_(nullptr),
        net_inititialized_flag_(nullptr),
        is_shared_(false) {
    InitMutex();
  }

  /**
   * @brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;

  /**
   * @brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  virtual inline bool ShareInParallel() const { return false; }

  /** @brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  inline bool IsShared() const { return is_shared_; }

  /** @brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  inline void SetShared(bool is_shared) {
    CHECK(ShareInParallel() || !is_shared) << type() << "Layer does not support sharing.";
    is_shared_ = is_shared;
  }

  virtual ~LayerBase() {}

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the layer name.
   */
  const std::string& name() const {
    return layer_param_.name();
  }

  // Iteration counter maintained by Solver
  int iter() const;
  int parent_rank() const;

  Net* parent_net() {
    return parent_net_;
  }

  const Solver* parent_solver() const;

  void set_parent_net(Net* parent_net) {
    parent_net_ = parent_net;
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {}

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }

  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }

  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }

  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }

  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }

  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }

  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  virtual float Forward(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  virtual void Backward(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) = 0;

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return ((int)param_propagate_down_.size() > param_id) ? param_propagate_down_[param_id] : false;
  }

  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if ((int)param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob>>& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  virtual float loss(int top_index) const = 0;

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  void set_loss(const int top_index, const float value) {
    _set_loss(top_index, value);
  }

  void fm_by_user(bool val) {
    fm_by_user_ = val;
  }

  void bm_by_user(bool val) {
    bm_by_user_ = val;
  }

  bool is_fm_by_user() const {
    return fm_by_user_;
  }

  bool is_bm_by_user() const {
    return bm_by_user_;
  }

  void set_net_initialized_flag(Flag* init_flag) {
    net_inititialized_flag_ = init_flag;
  }

  /**
   * Some layers need to be initialized after first iteration
   * They should override this function and return a flag
   * @return Flag*
   */
  virtual Flag* layer_inititialized_flag() {
    return nullptr;
  }

  virtual bool skip_apply_update(int blob_id) const {
    return false;
  }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false) = 0;

  std::string print_current_device() const;

 protected:
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob>> blobs_;
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;

  bool debug_;
  bool fm_by_user_, bm_by_user_;
  Net* parent_net_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  /** Lock forward_mutex_ if this layer is shared */
  void Lock();
  /** Unlock forward_mutex_ if this layer is shared */
  void Unlock();

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom, Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob*>& bottom, const vector<Blob*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
        << type() << " Layer takes " << ExactNumBottomBlobs() << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
        << type() << " Layer takes at least " << MinBottomBlobs() << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
        << type() << " Layer takes at most " << MaxBottomBlobs() << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
        << type() << " Layer produces " << ExactNumTopBlobs() << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
        << type() << " Layer produces at least " << MinTopBlobs() << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
        << type() << " Layer produces at most " << MaxTopBlobs() << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
        << type() << " Layer produces one top blob as output for each bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  virtual void SetLossWeights(const vector<Blob*>& top) = 0;

  /** Gets set when Net::Init is over */
  Flag* net_inititialized_flag_;

 private:
  /** Whether this layer is actually shared by other nets*/
  bool is_shared_;

  /** The mutex for sequential forward if this layer is shared */
  shared_ptr<std::mutex> forward_mutex_;

  /** Initialize forward_mutex_ */
  void InitMutex();

  virtual void _set_loss(int top_index, const float value) = 0;

  DISABLE_COPY_MOVE_AND_ASSIGN(LayerBase);
};  // class LayerBase


template<typename Ftype, typename Btype>
class Layer : public LayerBase {
 public:
  explicit Layer(const LayerParameter& param);

  virtual float Forward(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom);

  void ToProto(LayerParameter* param, bool write_diff = false) override;

  /**
 * @brief Returns the scalar loss associated with a top blob at a given index.
 */
  float loss(int top_index) const override {
    return (loss_.size() > top_index) ? loss_[top_index] : 0.F;
  }

 protected:
  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<float> loss_;

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;

  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
    Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) = 0;

  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) {
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  virtual void SetLossWeights(const vector<Blob*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
            "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const float loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == 0.F) { continue; }
        this->set_loss(top_id, loss_weight);
        top[top_id]->set_diff(loss_weight);
      }
    }
  }

 private:
  virtual void _set_loss(int top_index, const float value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, 0);
    }
    loss_[top_index] = value;
  }
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template<typename Ftype, typename Btype>
inline float Layer<Ftype, Btype>::Forward(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  // Lock during forward to ensure sequential forward
  Lock();
  float loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
    case Caffe::CPU:
      Forward_cpu(bottom, top);
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        if (this->loss(top_id) == 0.F) { continue; }
        const int count = top[top_id]->count();
        const Ftype* data = top[top_id]->cpu_data<Ftype>();
        const Ftype* loss_weights = top[top_id]->cpu_diff<Ftype>();
        loss += caffe_cpu_dot(count, data, loss_weights);
      }
      break;
    case Caffe::GPU:
      Forward_gpu(bottom, top);
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        if (this->loss(top_id) == 0.F) { continue; }
        const int count = top[top_id]->count();
        const Ftype* data = top[top_id]->gpu_data<Ftype>();
        const Ftype* loss_weights = top[top_id]->gpu_diff<Ftype>();
        float blob_loss = 0.F;
        caffe_gpu_dot(count, data, loss_weights, &blob_loss);
        loss += blob_loss;
      }
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
  Unlock();
  return loss;
}

template<typename Ftype, typename Btype>
inline void
Layer<Ftype, Btype>::Backward(const vector<Blob*>& top, const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      Backward_cpu(top, propagate_down, bottom);
      break;
    case Caffe::GPU:
      Backward_gpu(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
