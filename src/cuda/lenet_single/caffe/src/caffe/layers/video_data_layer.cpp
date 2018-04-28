#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdint.h>
#include <algorithm>
#include <csignal>
#include <map>
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
VideoDataLayer<Ftype, Btype>::VideoDataLayer(const LayerParameter& param, size_t solver_rank)
  : BasePrefetchingDataLayer<Ftype, Btype>(param, solver_rank) {}

template <typename Ftype, typename Btype>
VideoDataLayer<Ftype, Btype>::~VideoDataLayer() {
  this->StopInternalThread();
  if (cap_.isOpened()) {
    cap_.release();
  }
}

template <typename Ftype, typename Btype>
void VideoDataLayer<Ftype, Btype>::DataLayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const VideoDataParameter& video_data_param =
      this->layer_param_.video_data_param();
  video_type_ = video_data_param.video_type();
  skip_frames_ = video_data_param.skip_frames();
  CHECK_GE(skip_frames_, 0);
  TBlob<Btype> transformed_datum;
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img;
  if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
    const int device_id = video_data_param.device_id();
    if (!cap_.open(device_id)) {
      LOG(FATAL) << "Failed to open webcam: " << device_id;
    }
    cap_ >> cv_img;
  } else if (video_type_ == VideoDataParameter_VideoType_VIDEO) {
    CHECK(video_data_param.has_video_file()) << "Must provide video file!";
    const string& video_file = video_data_param.video_file();
    if (!cap_.open(video_file)) {
      LOG(FATAL) << "Failed to open video: " << video_file;
    }
    total_frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
    processed_frames_ = 0;
    // Read image to infer shape.
    cap_ >> cv_img;
    // Set index back to the first frame.
    cap_.set(CV_CAP_PROP_POS_FRAMES, 0);
  } else {
    LOG(FATAL) << "Unknow video type!";
  }
  CHECK(cv_img.data) << "Could not load image!";
  // Use data_transformer to infer the expected blob shape from a cv_image.
  top_shape_ = this->bdt(0)->InferBlobShape(cv_img);
  transformed_datum.Reshape(top_shape_);
  top_shape_[0] = batch_size;
  top[0]->Reshape(top_shape_);
  vector<int> label_shape(1, batch_size);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    top[1]->Reshape(label_shape);
  }
  this->batch_transformer_->reshape(top_shape_, label_shape, this->is_gpu_transform());
}

// This function is called on prefetch thread
template <typename Ftype, typename Btype>
void VideoDataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t queue_id) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_->count());
  TBlob<Btype> transformed_datum;

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  top_shape_[0] = 1;
  transformed_datum.Reshape(top_shape_);
  // Reshape batch according to the batch_size.
  top_shape_[0] = batch_size;
  batch->data_->Reshape(top_shape_);

  Ftype* top_data = batch->data_->mutable_cpu_data<Ftype>();
  Ftype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = batch->label_->mutable_cpu_data<Ftype>();
  }

  int skip_frames = skip_frames_;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    cv::Mat cv_img;
    if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
      cap_ >> cv_img;
    } else if (video_type_ == VideoDataParameter_VideoType_VIDEO) {
      if (processed_frames_ >= total_frames_) {
        LOG(INFO) << "Finished processing video.";
        raise(SIGINT);
      }
      ++processed_frames_;
      cap_ >> cv_img;
    } else {
      LOG(FATAL) << "Unknown video type.";
    }
    CHECK(cv_img.data) << "Could not load image!";
    read_time += timer.MicroSeconds();
    if (skip_frames > 0) {
      --skip_frames;
      --item_id;
    } else {
      skip_frames = skip_frames_;
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_->offset(item_id);
      transformed_datum.set_cpu_data(top_data + offset);
      this->bdt(0)->Transform(cv_img, &(transformed_datum));
      trans_time += timer.MicroSeconds();
    }
    CHECK(cv_img.data) << "Could not load image!";
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_->offset(item_id);
    transformed_datum.set_cpu_data(top_data + offset);
    this->bdt(0)->Transform(cv_img, &(transformed_datum));
    trans_time += timer.MicroSeconds();
    if (this->output_labels_) {
      top_label[item_id] = 0;
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS_FB(VideoDataLayer);
REGISTER_LAYER_CLASS_R(VideoData);

}  // namespace caffe
