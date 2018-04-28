#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <google/protobuf/text_format.h>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class GradientBasedSolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GradientBasedSolverTest() :
      seed_(1701), num_(4), channels_(3), height_(10), width_(10),
      share_(false) {
        input_file_ = new string(
        CMAKE_SOURCE_DIR "caffe/test/test_data/solver_data_list.txt" CMAKE_EXT);
      }
  ~GradientBasedSolverTest() {
    delete input_file_;
  }

  string snapshot_prefix_;
  shared_ptr<SGDSolver<Dtype> > solver_;
  int seed_;
  // Dimensions are determined by generate_sample_data.py
  // TODO this is brittle and the hdf5 file should be checked instead.
  int num_, channels_, height_, width_;
  bool share_;
  float delta_;  // Stability constant for RMSProp, AdaGrad, AdaDelta and Adam

  // Test data: check out generate_sample_data.py in the same directory.
  string* input_file_;

  virtual const char* solver_type() = 0;
  virtual void InitSolver(const SolverParameter& param) = 0;

  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    param.set_type(solver_type());
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }
    InitSolver(param);
    delta_ =  param.delta();
  }

  string RunLeastSquaresSolver(const float learning_rate,
      const float weight_decay, const float momentum, const int num_iters,
      const int iter_size = 1, const int devices = 1,
      const bool snapshot = false, const char* from_snapshot = NULL) {
    ostringstream proto;
    int device_id = 0;
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaGetDevice(&device_id));
    }
    proto <<
       "snapshot_after_train: " << snapshot << " "
       "max_iter: " << num_iters << " "
       "base_lr: " << learning_rate << " "
       "lr_policy: 'fixed' "
       "iter_size: " << iter_size << " "
       "device_id: " << device_id << " "
       "net_param { "
       "  name: 'TestNetwork' "
       "  default_forward_type: " << Type_Name(tp<Dtype>()) << " "
       "  default_backward_type: " << Type_Name(tp<Dtype>()) << " "
       "  default_forward_math: " << Type_Name(tp<Dtype>()) << " "
       "  default_backward_math: " << Type_Name(tp<Dtype>()) << " "
       "  layer { "
       "    name: 'data' "
       "    type: 'HDF5Data' "
       "    hdf5_data_param { "
       "      source: '" << *(this->input_file_) << "' "
       "      batch_size: " << num_ / iter_size << " "
       "    } "
       "    top: 'data' "
       "    top: 'targets' "
       "  } ";
    if (share_) {
      proto <<
         "  layer { "
         "    name: 'slice' "
         "    type: 'Slice' "
         "    bottom: 'data' "
         "    top: 'data1' "
         "    top: 'data2' "
         "    slice_param { "
         "      axis: 0 "
         "    } "
         "  } ";
    }
    proto <<
       "  layer { "
       "    name: 'innerprod' "
       "    type: 'InnerProduct' "
       "    param { name: 'weights' } "
       "    param { name: 'bias' } "
       "    inner_product_param { "
       "      num_output: 1 "
       "      weight_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "      bias_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "    } "
       "    bottom: '" << string(share_ ? "data1": "data") << "' "
       "    top: '" << string(share_ ? "innerprod1": "innerprod") << "' "
       "  } ";
    if (share_) {
      proto <<
         "  layer { "
         "    name: 'innerprod2' "
         "    type: 'InnerProduct' "
         "    param { name: 'weights' } "
         "    param { name: 'bias' } "
         "    inner_product_param { "
         "      num_output: 1 "
         "      weight_filler { "
         "        type: 'gaussian' "
         "        std: 1.0 "
         "      } "
         "      bias_filler { "
         "        type: 'gaussian' "
         "        std: 1.0 "
         "      } "
         "    } "
         "    bottom: 'data2' "
         "    top: 'innerprod2' "
         "  } "
         "  layer { "
         "    name: 'concat' "
         "    type: 'Concat' "
         "    bottom: 'innerprod1' "
         "    bottom: 'innerprod2' "
         "    top: 'innerprod' "
         "    concat_param { "
         "      axis: 0 "
         "    } "
         "  } ";
    }
    proto <<
       "  layer { "
       "    name: 'loss' "
       "    type: 'EuclideanLoss' "
       "    bottom: 'innerprod' "
       "    bottom: 'targets' "
       "  } "
       "} ";
    if (weight_decay != 0) {
      proto << "weight_decay: " << weight_decay << " ";
    }
    if (momentum != 0) {
      proto << "momentum: " << momentum << " ";
    }
    snapshot_prefix_ = MakeTempDir();
    proto << "snapshot_prefix: '" << snapshot_prefix_ << "/' ";
    if (snapshot) {
      proto << "snapshot: " << num_iters << " ";
    }
    Caffe::set_random_seed(this->seed_);
    this->InitSolverFromProtoString(proto.str());
    if (from_snapshot != NULL) {
      this->solver_->Restore(from_snapshot);
      for (int i = 0; i < this->solver_->iter(); ++i) {
        this->solver_->net()->Forward();
      }
    }
    if (devices == 1) {
      this->solver_->Solve();
    } else {
      LOG(INFO) << "Multi-GPU test on " << devices << " devices";
      vector<int> gpus;
      // put current device at the beginning
      int device_id = solver_->param().device_id();
      gpus.push_back(device_id);
      for (int i = 0; gpus.size() < devices; ++i) {
        if (i != device_id)
          gpus.push_back(i);
      }
      Caffe::set_solver_count(gpus.size());
      P2PManager sync(this->solver_, gpus.size(), this->solver_->param());
      sync.Run(gpus);
      Caffe::set_solver_count(1);
    }
    if (snapshot) {
      ostringstream resume_file;
      resume_file << snapshot_prefix_ << "/_iter_" << num_iters
                  << ".solverstate";
      string resume_filename = resume_file.str();
      return resume_filename;
    }
    return string();
  }

  // Compute an update value given the current state of the train net,
  // using the analytical formula for the least squares gradient.
  // updated_params will store the updated weight and bias results,
  // using the blobs' diffs to hold the update values themselves.
  void ComputeLeastSquaresUpdate(const float learning_rate,
      const float weight_decay, const float momentum, const int num_iters,
      vector<shared_ptr<TBlob<Dtype> > >* updated_params) {
    const int N = num_;
    const int D = channels_ * height_ * width_;

    // Run a forward pass, and manually compute the update values from the
    // result.
    Net& net = *this->solver_->net();
    net.Forward();
    ASSERT_TRUE(net.has_blob("data"));
    const Blob& data = *net.blob_by_name("data");
    ASSERT_TRUE(net.has_blob("targets"));
    const Blob& targets = *net.blob_by_name("targets");
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<Blob> >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    const int num_param_blobs = 2;
    ASSERT_EQ(num_param_blobs, param_blobs.size());
    const Blob& weights = *param_blobs[0];
    const Blob& bias = *param_blobs[1];
    ASSERT_EQ(D * N, data.count());
    ASSERT_EQ(N, targets.count());
    ASSERT_EQ(D, weights.count());
    ASSERT_EQ(1, bias.count());

    updated_params->clear();
    updated_params->resize(num_param_blobs);
    for (int i = 0; i < num_param_blobs; ++i) {
      (*updated_params)[i].reset(new TBlob<Dtype>());
    }
    TBlob<Dtype>& updated_weights = *(*updated_params)[0];
    updated_weights.ReshapeLike(weights);
    TBlob<Dtype>& updated_bias = *(*updated_params)[1];
    updated_bias.ReshapeLike(bias);

    for (int i = 0; i <= D; ++i) {
      // Compute the derivative with respect to the ith weight (i.e., the ith
      // element of the gradient).
      Dtype grad = 0;
      for (int j = 0; j <= D; ++j) {
        // Compute element (i, j) of X^T * X.
        Dtype element = 0;
        for (int k = 0; k < N; ++k) {
          // (i, k) in X^T (== (k, i) in X) times (k, j) in X.
          const Dtype element_i = (i == D) ? Dtype(1.) :
              data.template cpu_data<Dtype>()[k * D + i];
          const Dtype element_j = (j == D) ? Dtype(1.) :
              data.template cpu_data<Dtype>()[k * D + j];
          element += element_i * element_j;
        }
        if (j == D) {
          grad += element * bias.template cpu_data<Dtype>()[0];
        } else {
          grad += element * weights.template cpu_data<Dtype>()[j];
        }
      }
      for (int k = 0; k < N; ++k) {
        const Dtype element_i = (i == D) ? Dtype(1.) :
            data.template cpu_data<Dtype>()[k * D + i];
        grad -= element_i * targets.template cpu_data<Dtype>()[k];
      }
      // Scale the gradient over the N samples.
      grad /= N;
      // Add the weight decay to the gradient.
      grad += weight_decay *
          ((i == D) ? bias.template cpu_data<Dtype>()[0] :
              weights.template cpu_data<Dtype>()[i]);
      // Finally, compute update.
      const vector<shared_ptr<TBlob<Dtype> > >& history = solver_->history();
      if (solver_->type() != string("AdaDelta")
          && solver_->type() != string("Adam")) {
        ASSERT_EQ(2, history.size());  // 1 blob for weights, 1 for bias
      } else {
        ASSERT_EQ(4, history.size());  // additional blobs for update history
      }
      Dtype update_value = learning_rate * grad;
      const Dtype history_value = (i == D) ?
            history[1]->cpu_data()[0] : history[0]->cpu_data()[i];
      const Dtype temp = momentum * history_value;
      if (solver_->type() == string("SGD")) {
        update_value += temp;
      } else if (solver_->type() == string("Nesterov")) {
        update_value += temp;
        // step back then over-step
        update_value = (1. + momentum) * update_value - temp;
      } else if (solver_->type() == string("AdaGrad")) {
      if (is_type<Dtype>(FLOAT16)) {
        // bginsburg: adagrad definition was modified for fp16
        update_value = update_value /
            (std::sqrt(history_value* history_value + grad * grad) + delta_);
      } else {
        update_value = update_value /
            (std::sqrt(history_value + grad * grad) + delta_);
      }

      } else if (solver_->type() == string("RMSProp")) {
        const Dtype rms_decay = 0.95;
        update_value /= std::sqrt(rms_decay*history_value
            + grad * grad * (1. - rms_decay)) + delta_;
      } else if (solver_->type() == string("AdaDelta")) {
        const Dtype update_history_value = (i == D) ?
            history[1 + num_param_blobs]->cpu_data()[0] :
            history[0 + num_param_blobs]->cpu_data()[i];
        const Dtype weighted_gradient_average =
            momentum * history_value + (1. - momentum) * (grad * grad);
        update_value = grad * std::sqrt((update_history_value + delta_) /
            (weighted_gradient_average + delta_)) * learning_rate;
        // not actually needed, just here for illustrative purposes
        // const Dtype weighted_update_average =
        //   momentum * update_history_value + (1 - momentum) * (update_value);
      } else if (solver_->type() == string("Adam")) {
        const float momentum2 = 0.999;
        const Dtype m = history_value;
        const Dtype v = (i == D) ?
            history[1 + num_param_blobs]->cpu_data()[0] :
            history[0 + num_param_blobs]->cpu_data()[i];
        const Dtype val_m = (1. - momentum) * grad + momentum * m;
        const Dtype val_v = (1. - momentum2) * grad * grad + momentum2 * v;
        Dtype alpha_t = learning_rate *
            std::sqrt(Dtype(1.) - pow(momentum2, float(num_iters))) /
            (Dtype(1.) - pow(momentum, float(num_iters)));
        update_value = alpha_t * val_m / (std::sqrt(val_v) + delta_);
      } else {
        LOG(FATAL) << "Unknown solver type: " << solver_->type();
      }
      if (i == D) {
        updated_bias.template mutable_cpu_diff()[0] = update_value;
        updated_bias.template mutable_cpu_data()[0] =
            bias.template cpu_data<Dtype>()[0] - update_value;
      } else {
        updated_weights.mutable_cpu_diff()[i] = update_value;
        updated_weights.template mutable_cpu_data()[i] =
            weights.template cpu_data<Dtype>()[i] - update_value;
      }
    }
  }

  void CheckLeastSquaresUpdate(const vector<shared_ptr<TBlob<Dtype> > >& updated_params) {
    const int D = channels_ * height_ * width_;

    const TBlob<Dtype>& updated_weights = *updated_params[0];
    const TBlob<Dtype>& updated_bias = *updated_params[1];

    Net& net = *this->solver_->net();
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<Blob> >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    ASSERT_EQ(2, param_blobs.size());
    const Blob& solver_updated_weights = *param_blobs[0];
    ASSERT_EQ(D, solver_updated_weights.count());
    const double kPrecision = tol<Dtype>(5.e-1, 2.);
    const double kMinPrecision = 1.5e-1;
    for (int i = 0; i < D; ++i) {
      const Dtype expected_updated_weight = updated_weights.cpu_data()[i];
      const Dtype solver_updated_weight = solver_updated_weights.cpu_data<Dtype>()[i];
      const Dtype error_margin = std::max(kMinPrecision, kPrecision *
          std::max(fabs(expected_updated_weight), fabs(solver_updated_weight)));
      EXPECT_NEAR(expected_updated_weight, solver_updated_weight, error_margin);
    }
    const Blob& solver_updated_bias_blob = *param_blobs[1];
    ASSERT_EQ(1, solver_updated_bias_blob.count());
    const Dtype expected_updated_bias = updated_bias.cpu_data()[0];
    const Dtype solver_updated_bias = solver_updated_bias_blob.template cpu_data<Dtype>()[0];
    const Dtype error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_updated_bias), fabs(solver_updated_bias)));
    EXPECT_NEAR(expected_updated_bias, solver_updated_bias, error_margin);

    // Check the solver's history -- should contain the previous update value.
    if (solver_->type() == string("SGD")) {
      const vector<shared_ptr<TBlob<Dtype> > >& history = solver_->history();
      ASSERT_EQ(2, history.size());
      for (int i = 0; i < D; ++i) {
        const Dtype expected_history = updated_weights.cpu_diff()[i];
        const Dtype solver_history = history[0]->cpu_data()[i];
        const Dtype error_margin_hist = std::max(kMinPrecision, kPrecision *
            std::min(fabs(expected_history), fabs(solver_history)));
        EXPECT_NEAR(expected_history, solver_history, error_margin_hist * 2.F);
      }
      const Dtype expected_history = updated_bias.cpu_diff()[0];
      const Dtype solver_history = history[1]->cpu_data()[0];
      const Dtype error_margin_hist = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_history), fabs(solver_history)));
      EXPECT_NEAR(expected_history, solver_history,
          tol<Dtype>(error_margin_hist, error_margin_hist * 2.F));
    }
  }

  void CheckAccumulation(const float kLearningRate, const float kWeightDecay,
      const float kMomentum, const int kNumIters, const int kIterSize) {
    const double kPrecision = 1e-2;
    const double kMinPrecision = tol<Dtype>(1e-5, 6e-2);
    // Solve without accumulation and save parameters.
    this->RunLeastSquaresSolver(kLearningRate, kWeightDecay, kMomentum,
        kNumIters);
    // Save parameters for comparison.
    Net& net = *this->solver_->net();
    const vector<shared_ptr<Blob> >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    vector<shared_ptr<TBlob<double> > > noaccum_params(param_blobs.size());
    for (int i = 0; i < param_blobs.size(); ++i) {
      noaccum_params[i].reset(new TBlob<double>());
      noaccum_params[i]->CopyFrom(*param_blobs[i], false, true);
    }
    // Solve by equivalent accumulation of gradients over divided batches.
    this->RunLeastSquaresSolver(kLearningRate, kWeightDecay, kMomentum,
        kNumIters, kIterSize);
    Net& net_accum = *this->solver_->net();
    const vector<shared_ptr<Blob> >& accum_params =
        net_accum.layer_by_name("innerprod")->blobs();
    // Compare accumulated parameters against no accumulation standard.
    const int D = this->channels_ * this->height_ * this->width_;
    for (int i = 0; i < D; ++i) {
      const double expected_param = noaccum_params[0]->cpu_data()[i];
      const double accum_param = accum_params[0]->cpu_data<double>()[i];
      const double error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_param), fabs(accum_param)));
      EXPECT_NEAR(expected_param, accum_param, error_margin);
    }
    ASSERT_EQ(1, accum_params[1]->count());
    const double expected_bias = noaccum_params[1]->cpu_data()[0];
    const double accum_bias = accum_params[1]->cpu_data<double>()[0];
    const double error_margin = std::max(kMinPrecision, kPrecision *
        std::min(fabs(expected_bias), fabs(accum_bias)));
    EXPECT_NEAR(expected_bias, accum_bias, error_margin);
  }

  // Test that the correct update is computed for a regularized least squares
  // problem:
  //
  //            E = (1/(2n)) || X w - y ||^2 + (lambda / 2) || w ||^2
  //   \nabla_w E = (1/n) (X^T X w - X^T y) + lambda * w
  //
  // X \in R^{n x (d+1)} (each example is a row, (d+1)th element is always 1)
  // w \in R^{(d+1) x 1} ((d+1)th element is the bias)
  // y \in R^{n x 1}
  // lambda is weight_decay
  //
  // TestLeastSquaresUpdate works "inductively", assuming that the solver
  // correctly updates the net K (= iter_to_check) times, then given the history
  // from the Kth update, we compute the (K+1)th update and check that it
  // matches the solver's (K+1)th update.

  void TestLeastSquaresUpdate(const float learning_rate = 1.0,
         const float weight_decay = 0.0, const float momentum = 0.0,
         const int iter_to_check = 0) {
    const int kIterSize = 1;
    const int kNum = num_;
    // Test over all numbers of devices.
    int available_devices = 1;
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaGetDeviceCount(&available_devices));
    }
    for (int devices = 1; devices <= available_devices; ++devices) {
      // Configure batch size for single / multi device equivalence.
      // Constant data is needed for multi device as for accumulation.
      num_ = kNum * devices * devices;

      // Initialize the solver and run K (= iter_to_check) solver iterations
      // (on single device).
      RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                            iter_to_check, kIterSize, 1);

      // Compute the (K+1)th update using the analytic least squares gradient.
      vector<shared_ptr<TBlob<Dtype> > > updated_params;
      ComputeLeastSquaresUpdate(learning_rate, weight_decay, momentum,
          iter_to_check + 1, &updated_params);

      // Reinitialize the solver and run K+1 solver iterations.
      num_ = kNum * devices;
      RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
          iter_to_check + 1, kIterSize, devices);

      // Check that the solver's solution matches ours.
      CheckLeastSquaresUpdate(updated_params);

      // Reset initial value of num_
      num_ = kNum;
    }
  }

  void TestSnapshot(const Dtype learning_rate = 1.0,
      const Dtype weight_decay = 0.0, const float momentum = 0.0,
      const int num_iters = 1) {
    // Run the solver for num_iters * 2 iterations.
    const int total_num_iters = num_iters * 2;
    bool snapshot = false;
    const int kIterSize = 1;
    const int kDevices = 1;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices, snapshot);

    // Save the resulting param values.
    vector<shared_ptr<TBlob<Dtype> > > param_copies;
    const vector<shared_ptr<Blob>>& orig_params =
        solver_->net()->learnable_params();
    param_copies.resize(orig_params.size());
    for (int i = 0; i < orig_params.size(); ++i) {
      param_copies[i].reset(new TBlob<Dtype>());
      param_copies[i]->CopyDataFrom(*orig_params[i], true);
      param_copies[i]->CopyDiffFrom(*orig_params[i], true);
    }

    // Save the solver history
    vector<shared_ptr<TBlob<Dtype> > > history_copies;
    const vector<shared_ptr<TBlob<Dtype> > >& orig_history = solver_->history();
    history_copies.resize(orig_history.size());
    for (int i = 0; i < orig_history.size(); ++i) {
      history_copies[i].reset(new TBlob<Dtype>());
      history_copies[i]->CopyDataFrom(*orig_history[i], true);
      history_copies[i]->CopyDiffFrom(*orig_history[i], true);
    }

    // Run the solver for num_iters iterations and snapshot.
    snapshot = true;
    string snapshot_name = RunLeastSquaresSolver(learning_rate, weight_decay,
        momentum, num_iters, kIterSize, kDevices, snapshot);

    // Reinitialize the solver and run for num_iters more iterations.
    snapshot = false;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices,
        snapshot, snapshot_name.c_str());

    // Check that params now match.
    const vector<shared_ptr<Blob>>& params = solver_->net()->learnable_params();
    for (int i = 0; i < params.size(); ++i) {
      for (int j = 0; j < params[i]->count(); ++j) {
        Dtype rel_error = std::max<Dtype>(1., fabs(params[i]->cpu_data<Dtype>()[j])) *
            tol<Dtype>(1.e-2, 1.e-1);
        EXPECT_NEAR(param_copies[i]->cpu_data()[j], params[i]->cpu_data<Dtype>()[j],
            rel_error) << "param " << i << " data differed at dim " << j;
        EXPECT_NEAR(param_copies[i]->cpu_diff()[j], params[i]->cpu_diff<Dtype>()[j],
            rel_error) << "param " << i << " diff differed at dim " << j;
      }
    }

    // Check that history now matches.
    const vector<shared_ptr<TBlob<Dtype> > >& history = solver_->history();
    for (int i = 0; i < history.size(); ++i) {
      for (int j = 0; j < history[i]->count(); ++j) {
        Dtype rel_error = std::max<Dtype>(1., fabs(history[i]->cpu_data()[j])) *
            tol<Dtype>(1.e-5, 1.e-4);
        EXPECT_NEAR(history_copies[i]->cpu_data()[j], history[i]->cpu_data()[j],
            rel_error) << "history blob " << i << " data differed at dim " << j;
        EXPECT_NEAR(history_copies[i]->cpu_diff()[j], history[i]->cpu_diff()[j],
            rel_error) << "history blob " << i << " diff differed at dim " << j;
      }
    }
  }
};


template <typename TypeParam>
class SGDSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new SGDSolver<Dtype>(param));
  }
  virtual const char *solver_type() {
    return "SGD";
  }
};

TYPED_TEST_CASE(SGDSolverTest, TestDtypesGPUOnly);

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateLROneHundredth) {
  const float kLearningRate = 0.01F;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecay) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecayMultiIter) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0;
  const float kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0;
  const float kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverything) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  if (tp<Dtype>() != FLOAT) {
    return;  // we don't support Solver&Net in fp16 precision yet
  }
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.5;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(SGDSolverTest, TestSnapshot) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestSnapshotShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class AdaGradSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaGradSolver<Dtype>(param));
  }
  virtual const char *solver_type() {
    return "AdaGrad";
  }
};

TYPED_TEST_CASE(AdaGradSolverTest, TestDtypesGPUOnly);

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateLROneHundredth) {
  const float kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithWeightDecay) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithEverything) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest,
      TestAdaGradLeastSquaresUpdateWithEverythingShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaGradSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaGradSolverTest, TestSnapshot) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest, TestSnapshotShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class NesterovSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new NesterovSolver<Dtype>(param));
  }
  virtual const char *solver_type() {
    return "Nesterov";
  }
};

TYPED_TEST_CASE(NesterovSolverTest, TestDtypesGPUOnly);

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateLROneHundredth) {
  const float kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithWeightDecay) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(NesterovSolverTest,
           TestNesterovLeastSquaresUpdateWithWeightDecayMultiIter) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithMomentum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0;
  const float kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0;
  const float kMomentum = 0.5;
  const int kNumIters = is_type<Dtype>(FLOAT16) ? 3 : 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithEverything) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest,
           TestNesterovLeastSquaresUpdateWithEverythingShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(NesterovSolverTest, TestSnapshot) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestSnapshotShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template <typename TypeParam>
class AdaDeltaSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaDeltaSolver<Dtype>(param));
  }
  virtual const char *solver_type() {
    return "AdaDelta";
  }
};

TYPED_TEST_CASE(AdaDeltaSolverTest, TestDtypesGPUOnly);

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdate) {
  const float kLearningRate = 0.1;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithWeightDecay) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.05;  // 0.95;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithHalfMomentum) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.0;
  const float kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithMomentum) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.0;
  const float kMomentum = 0.95;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.0;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithEverything) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.1;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithEverythingShare) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.1;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.1;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.1;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaDeltaSolverTest, TestSnapshot) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.1;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestSnapshotShare) {
  const float kLearningRate = 0.1;
  const float kWeightDecay = 0.1;
  const float kMomentum = 0.95;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template <typename TypeParam>
class AdamSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    SolverParameter new_param = param;
    const float momentum = 0.9;
    new_param.set_momentum(momentum);
    const float momentum2 = 0.999;
    new_param.set_momentum2(momentum2);
    this->solver_.reset(new AdamSolver<Dtype>(new_param));
  }
  virtual const char *solver_type() {
    return "Adam";
  }
};

TYPED_TEST_CASE(AdamSolverTest, TestDtypesGPUOnly);

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdate) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0;
  const float kMomentum = 0.9;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithWeightDecay) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithEverything) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithEverythingShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdamSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdamSolverTest, TestSnapshot) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestSnapshotShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template <typename TypeParam>
class RMSPropSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    const Dtype rms_decay = 0.95;
    SolverParameter new_param = param;
    new_param.set_rms_decay(rms_decay);
    this->solver_.reset(new RMSPropSolver<Dtype>(new_param));
  }
  virtual const char *solver_type() {
    return "RMSProp";
  }
};

TYPED_TEST_CASE(RMSPropSolverTest, TestDtypesGPUOnly);

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithWeightDecay) {
  const float kLearningRate = 1.0;
  const float kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithRmsDecay) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.0;
  const float kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithEverything) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest,
      TestRMSPropLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  if (tp<Dtype>() != FLOAT) {
    return;  // TODO
  }
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(RMSPropSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0.0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(RMSPropSolverTest, TestSnapshot) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestSnapshotShare) {
  const float kLearningRate = 0.01;
  const float kWeightDecay = 0.5;
  const float kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

}  // namespace caffe
