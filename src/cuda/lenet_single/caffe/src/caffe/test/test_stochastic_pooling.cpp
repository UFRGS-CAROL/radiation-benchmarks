#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;

namespace caffe {

template <typename TypeParam>
class StochasticPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  StochasticPoolingLayerTest()
      : blob_bottom_(new TBlob<Dtype>()),
        blob_top_(new TBlob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.1);
    filler_param.set_max(1.);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~StochasticPoolingLayerTest() {
    delete blob_bottom_; delete blob_top_;
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

template <typename Dtype>
class CPUStochasticPoolingLayerTest
  : public StochasticPoolingLayerTest<CPUDevice<Dtype> > {
};

TYPED_TEST_CASE(CPUStochasticPoolingLayerTest, TestDtypes);

TYPED_TEST(CPUStochasticPoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

template <typename Dtype>
class GPUStochasticPoolingLayerTest
  : public StochasticPoolingLayerTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUStochasticPoolingLayerTest, TestDtypes);

TYPED_TEST(GPUStochasticPoolingLayerTest, TestStochastic) {
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
  PoolingLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check if the output is correct - it should do random sampling
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  TypeParam total = 0;
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int ph = 0; ph < this->blob_top_->height(); ++ph) {
        for (int pw = 0; pw < this->blob_top_->width(); ++pw) {
          TypeParam pooled = top_data[this->blob_top_->offset(n, c, ph, pw)];
          total += pooled;
          int hstart = ph * 2;
          int hend = min(hstart + 3, this->blob_bottom_->height());
          int wstart = pw * 2;
          int wend = min(wstart + 3, this->blob_bottom_->width());
          bool has_equal = false;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              has_equal |= (pooled == bottom_data[this->blob_bottom_->
                  offset(n, c, h, w)]);
            }
          }
          EXPECT_TRUE(has_equal);
        }
      }
    }
  }
  // When we are doing stochastic pooling, the average we get should be higher
  // than the simple data average since we are weighting more on higher-valued
  // ones.
  EXPECT_GE(total / this->blob_top_->count(), 0.55);
}

TYPED_TEST(GPUStochasticPoolingLayerTest, TestStochasticTestPhase) {
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
  PoolingLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check if the output is correct - it should do random sampling
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int ph = 0; ph < this->blob_top_->height(); ++ph) {
        for (int pw = 0; pw < this->blob_top_->width(); ++pw) {
          TypeParam pooled = top_data[this->blob_top_->offset(n, c, ph, pw)];
          int hstart = ph * 2;
          int hend = min(hstart + 3, this->blob_bottom_->height());
          int wstart = pw * 2;
          int wend = min(wstart + 3, this->blob_bottom_->width());
          bool smaller_than_max = false;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              smaller_than_max |= (pooled <= bottom_data[this->blob_bottom_->
                  offset(n, c, h, w)]);
            }
          }
          EXPECT_TRUE(smaller_than_max);
        }
      }
    }
  }
}

TYPED_TEST(GPUStochasticPoolingLayerTest, TestGradient) {
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
  PoolingLayer<TypeParam, TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(tol<TypeParam>(1e-4, 3e-3), tol<TypeParam>(1e-1, 1e-1),
                                     1701, 0.F, 1.F);
  // it is too expensive to call curand multiple times, so we don't do an
  // exhaustive gradient check.
  checker.CheckGradient(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
