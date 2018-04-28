#include <algorithm>
#include <map>
#include <set>
#include <boost/thread.hpp>
#include <hdf5.h>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

constexpr int Net::END_OF_ITERATION;
constexpr int Net::END_OF_TRAIN;

Net::Net(const NetParameter& param,
    size_t solver_rank,
    Flag* solver_init_flag,
    const Net* root_net,
    bool inner_net,
    int level,
    const vector<string>* stages)
    : root_net_(root_net),
      solver_(nullptr),
      solver_rank_(solver_rank),
      solver_init_flag_(solver_init_flag),
      inner_net_(inner_net) {
  Init(param);
}

Net::Net(const string& param_file,
    Phase phase,
    size_t solver_rank,
    Flag* solver_init_flag,
    const Net* root_net,
    bool inner_net,
    int level,
    const vector<string>* stages)
    : root_net_(root_net),
      solver_(nullptr),
      solver_rank_(solver_rank),
      solver_init_flag_(solver_init_flag),
      inner_net_(inner_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); ++i) {
      param.mutable_state()->add_stage(stages->at(i));
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

Net::~Net() {
}

void Net::Init(const NetParameter& in_param) {
  CHECK(inner_net_ || Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  net_param_ = filtered_param;
  batch_per_solver_ = caffe::P2PSync::divide_batch_size(&filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  infer_count_ = 0UL;
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  gpu_top_memory_data_use_ = gpu_top_memory_diff_use_ = 0UL;
  gpu_btm_memory_data_use_ = gpu_btm_memory_diff_use_ = 0UL;
  gpu_shr_memory_data_use_ = gpu_shr_memory_diff_use_ = 0UL;
  gpu_prm_memory_data_use_ = gpu_prm_memory_diff_use_ = 0UL;
  gpu_shp_memory_data_use_ = gpu_shp_memory_diff_use_ = 0UL;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());

  // If user skips default math type we use default data type:
  Type default_fmath, default_bmath;
  if (in_param.has_default_forward_math()) {
    default_fmath = in_param.default_forward_math();
  } else {
    default_fmath = in_param.default_forward_type();
    LOG(INFO) << "Using " << Type_Name(default_fmath) << " as default forward math type";
  }
  if (in_param.has_default_backward_math()) {
    default_bmath = in_param.default_backward_math();
  } else {
    default_bmath = in_param.default_backward_type();
    LOG(INFO) << "Using " << Type_Name(default_bmath) << " as default backward math type";
  }

  wgrad_sq_.store(0LL);
  global_grad_scale_coeff_ = 1.F;
  has_global_grad_scale_param_ = in_param.has_global_grad_scale();
  global_grad_scale_param_ = in_param.global_grad_scale();
  global_grad_scale_adaptive_ = in_param.global_grad_scale_adaptive();

  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !inner_net_ && !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();

    const LayerParameter& layer_param = param.layer(layer_id);
    LayerParameter* mutable_layer_param = param.mutable_layer(layer_id);

    DLOG_IF(INFO, Caffe::root_solver())
        << "Setting types for Layer " << layer_param.name();

    // Inherit phase from net if unset.
    if (!layer_param.has_phase()) {
      mutable_layer_param->set_phase(phase_);
    }
    const bool is_data_layer = layer_param.has_transform_param();

    // Data&Math types
    const bool fm_by_user = layer_param.has_forward_math();
    if (!fm_by_user) {
      if (layer_param.has_forward_type()) {
        mutable_layer_param->set_forward_math(layer_param.forward_type());
      } else {
        mutable_layer_param->set_forward_math(default_fmath);
      }
    }
    const bool bm_by_user = layer_param.has_backward_math();
    if (!bm_by_user) {
      if (layer_param.has_backward_type()) {
        mutable_layer_param->set_backward_math(layer_param.backward_type());
      } else {
        mutable_layer_param->set_backward_math(default_bmath);
      }
    }

    if (!layer_param.has_forward_type()) {
      mutable_layer_param->set_forward_type(in_param.default_forward_type());
    }
    if (!layer_param.has_backward_type()) {
      if (is_data_layer) {
        // In majority of cases we manage to avoid redundant conversion:
        mutable_layer_param->set_backward_type(FLOAT);
      } else {
        mutable_layer_param->set_backward_type(in_param.default_backward_type());
      }
    }

    // Convolution algorithms
    if (param.has_default_conv_algos_override() && layer_param.has_convolution_param() &&
        !layer_param.convolution_param().has_conv_algos_override()) {
      mutable_layer_param->mutable_convolution_param()->
          set_conv_algos_override(param.default_conv_algos_override());
    }

    // cuDNN math
    if (param.has_default_cudnn_math_override() &&
        !layer_param.has_cudnn_math_override()) {
      mutable_layer_param->set_cudnn_math_override(param.default_cudnn_math_override());
    }

    // Setup layer.
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
    } else {
      layers_.push_back(LayerRegistry::CreateLayer(layer_param, solver_rank_));
    }
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
        << "Created Layer " << layer_param.name() << " (" << layer_id << ")";
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    LayerBase* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    layer->fm_by_user(fm_by_user);
    layer->bm_by_user(bm_by_user);

    layers_[layer_id]->set_net_initialized_flag(solver_init_flag_);

    Flag* layer_inititialized_flag = layers_[layer_id]->layer_inititialized_flag();
    if (layer_inititialized_flag != nullptr) {
      layer_inititialized_flags_.push_back(layer_inititialized_flag);
    }

    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      layers_[layer_id]->set_parent_net(this);
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, 0.F);
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << Phase_Name(phase_) << " Top shape for layer " << layer_id << " '"
          << layer_names_[layer_id] << "' " <<  top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id) != 0.F) {
        LOG_IF(INFO, Caffe::root_solver())
          << "    with loss weight " << layer->loss(top_id);
      }
      gpu_top_memory_data_use_ += top_vecs_[layer_id][top_id]->gpu_memory_data_use();
      gpu_top_memory_diff_use_ += top_vecs_[layer_id][top_id]->gpu_memory_diff_use();
    }
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) != 0.F ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (int blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (int layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();

  // invert param_layer_indices_ to give map of
  // (level_id, local param_id) -> global param_id
  for (int i = 0; i < param_layer_indices_.size(); ++i) {
    layer_index_params_[param_layer_indices_[i]] = i;
  }

  learnable_space_size_[0] = 0UL;
  learnable_space_size_[1] = 0UL;
  reduce_buckets_ = (size_t) in_param.reduce_buckets();
  if (Caffe::device_count() > 0) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Top memory (" << Phase_Name(phase_) << ") required for data: "
        << gpu_top_memory_data_use_ << " diff: " << gpu_top_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Bottom memory (" << Phase_Name(phase_) << ") required for data: "
        << gpu_btm_memory_data_use_ << " diff: " << gpu_btm_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Shared (in-place) memory (" << Phase_Name(phase_) << ") by data: "
        << gpu_shr_memory_data_use_ << " diff: " << gpu_shr_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Parameters memory (" << Phase_Name(phase_) << ") required for data: "
        << gpu_prm_memory_data_use_ << " diff: " << gpu_prm_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Parameters shared memory (" << Phase_Name(phase_) << ") by data: "
        << gpu_shp_memory_data_use_ << " diff: " << gpu_shp_memory_diff_use_;
  }
  debug_info_ = param.debug_info();
  trained_layers_shared_ = false;
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

void Net::FilterNet(const NetParameter& param, NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

bool Net::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
void Net::AppendTop(const NetParameter& param, const int layer_id, const int top_id,
    set<string>* available_blobs, map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = (layer_param.top_size() > top_id) ?
      layer_param.top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param.bottom_size() > top_id &&
      blob_name == layer_param.bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param.name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    gpu_shr_memory_data_use_ += top_vecs_[layer_id].back()->gpu_memory_data_use();
    gpu_shr_memory_diff_use_ += top_vecs_[layer_id].back()->gpu_memory_diff_use();
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param.name() << " -> " << blob_name;
    }

    Type ftype = layer_param.has_forward_type() ? layer_param.forward_type() :
        param.default_forward_type();
    Type btype = layer_param.has_backward_type() ? layer_param.backward_type() :
        param.default_backward_type();
    shared_ptr<Blob> blob_pointer = Blob::create(ftype, btype);
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
int Net::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  gpu_btm_memory_data_use_ += bottom_vecs_[layer_id].back()->gpu_memory_data_use();
  gpu_btm_memory_diff_use_ += bottom_vecs_[layer_id].back()->gpu_memory_diff_use();
  return blob_id;
}

void Net::AppendParam(const NetParameter& param, const int layer_id, const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id]);
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

float Net::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  float loss = 0;
  for (int i = start; i <= end; ++i) {
    // LOG(INFO) << " ****** [Forward] (" << i << ") Layer '" << layer_names_[i];
    // << "' FT " << Type_Name(layers_[i]->forward_type())
    // << " BT " << Type_Name(layers_[i]->backward_type());
    float layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  ++infer_count_;
  return loss;
}

float Net::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

float Net::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

const vector<Blob*>& Net::Forward(float* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

const vector<Blob*>& Net::Forward(const vector<Blob*>& bottom, float* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

float Net::ForwardBackward(bool apply_update) {
  float loss;
  Forward(&loss);
  Backward(apply_update);
  return loss;
}

void Net::BackwardFromTo(int start, int end) {
  BackwardFromToAu(start, end, true);
}

void Net::BackwardFromToAu(int start, int end, bool apply_update) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (!layer_need_backward_[i]) {
      continue;
    }

    layers_[i]->Backward(top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);

    if (debug_info_) {
      BackwardDebugInfo(i);
    }
    if (!apply_update) {
      continue;
    }
    for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
      if (layers_[i]->skip_apply_update(j)) {
        continue;
      }
      const int param_id = layer_index_params_[make_pair(i, j)];
      if (param_owners_[param_id] < 0) {
        const int lparam_id = learnable_param_ids_[param_id];
        int t = (int)learnable_params_[lparam_id]->diff_type();
        for (int type_id = 0; type_id < learnable_types().size(); ++type_id) {
          if (t == learnable_types_[type_id]) {
            reduction_queue_[type_id].push(lparam_id);
            break;
          }
        }
      }  // leave it to the owner otherwise
    }
  }
  if (apply_update) {
    for (int type_id = 0; type_id < learnable_types_.size(); ++type_id) {
      reduction_queue_[type_id].push(END_OF_ITERATION);
    }
  }
}

void Net::Finalize() {
  for (int type_id = 0; type_id < learnable_types_.size(); ++type_id) {
    reduction_queue_[type_id].push(END_OF_TRAIN);
  }
}

size_t Net::received_contiguous_count(int type_id, const std::set<int>& au_ids, int& id_from_ret) {
  if (learnable_params_.empty() || au_ids.empty() || param_id_vecs_.empty()) {
    return 0;
  }
  size_t cnt_ret = 0UL, cnt = 0UL;
  const int bottom = *au_ids.begin();
  const int top = *au_ids.rbegin();
  id_from_ret = -1;
  const std::map<size_t, std::set<int>>& ltop = ltop_[type_id];
  for (auto lit = ltop.rbegin(); lit != ltop.rend(); ++lit) {
    if (lit->second.empty() || *lit->second.begin() > top) {
      continue;
    }
    bool layer_complete = true;
    for (auto p = lit->second.begin(); p != lit->second.end(); ++p) {
      int param_id = *p;
      if (param_id < bottom || au_ids.find(param_id) == au_ids.end()) {
        layer_complete = false;
        break;
      }
      cnt += lp_aligned_count(param_id);
    }
    if (layer_complete) {
      id_from_ret = *lit->second.begin();
      cnt_ret = cnt;
    } else {
      break;
    }
  }
  return cnt_ret;
}

void Net::ReduceAndUpdate(int type_id) {
  DLOG(INFO) << "[" << Caffe::current_device()
             << "] Entering ReduceAndUpdate thread " << lwp_id()
             <<  ", type_id " << type_id;

  size_t bucket_size = 0UL;
  cublasHandle_t handle = Caffe::cublas_handle(type_id);
  CHECK_GE(reduce_buckets_, 0);
  if (Caffe::solver_count() > 1 && reduce_buckets_ > 0) {
    bucket_size = align_up<6>(learnable_space_size_[type_id] / reduce_buckets_);
  }
  std::set<int> au_ids;

  const bool clip_grads = solver_->param().clip_gradients() >= 0.F;
  const bool clear_grads = !solver_->param().snapshot_diff() && !clip_grads;
  const bool use_buckets = reduce_buckets_ > 0;
  float rate = -1.F;
  while (!solver_->stop_reducing_requested(type_id)) {
    const int param_id = reduction_queue_[type_id].pop();
    SolverAction::Enum request = solver_->GetRequestedAction();
    if (SolverAction::STOP == request) {
      solver_->request_early_exit();
      break;
    }
    if (param_id == END_OF_TRAIN) {
      break;
    }
    if (rate < 0.F) {
      rate = solver_->GetLearningRate();
    }
    if (param_id != END_OF_ITERATION) {
      if (Caffe::solver_count() > 1) {
        if (!use_buckets && !clip_grads) {
          Reduce(type_id, param_id);
          if (solver_->stop_reducing_requested(type_id)) {
            break;
          }
          add_wgrad_sq(solver_->ApplyUpdate(param_id, handle, rate, true, clear_grads));
          continue;
        }
      } else {
        if (!clip_grads) {
          this->learnable_params()[param_id]->scale_diff(1.F / global_grad_scale(), handle);
          add_wgrad_sq(solver_->ApplyUpdate(param_id, handle, rate, true, clear_grads));
        }
        continue;
      }
    } else if (clip_grads && Caffe::solver_count() == 1) {
      solver_->ClipGradientsAndNormalize(handle, type_id, au_ids);
      for (int i : au_ids) {
        add_wgrad_sq(solver_->ApplyUpdate(i, handle, rate, false, clear_grads));
      }
      au_ids.clear();
    }

    if (!learnable_params_.empty() && Caffe::solver_count() > 1) {
      int id_from = -1;
      // Is bucket big enough? Done with iteration?
      const size_t received_count = received_contiguous_count(type_id, au_ids, id_from);
      if (id_from >= 0) {
        const size_t received_size = received_count * lp_size(id_from);
        if ((received_size >= bucket_size && !clip_grads) || param_id == END_OF_ITERATION) {
//#ifdef DEBUG
//          {
//            size_t c = 0UL;
//            for (int i : au_ids) {
//              if (i < id_from) {
//                continue;
//              }
//              c += lp_aligned_count(i);
//            }
//            CHECK_EQ(c, received_count);
//          }
//#endif
          CHECK_EQ((int) learnable_params_[id_from]->diff_type(), learnable_types_[type_id]);
          ReduceBucket(type_id, received_count, learnable_params_[id_from]->diff_type(),
              learnable_params_ptrs_[type_id][id_from]);
          if (solver_->stop_reducing_requested(type_id)) {
            break;
          }

          if (clip_grads) {
            solver_->ClipGradientsAndNormalize(handle, type_id, au_ids);
          }

          for (int i : au_ids) {
            add_wgrad_sq(solver_->ApplyUpdate(i, handle, rate, !clip_grads, clear_grads));
          }
          au_ids.erase(au_ids.find(id_from), au_ids.end());
        }
      }
    }
    if (param_id == END_OF_ITERATION) {
      CHECK(au_ids.empty());
      rate = -1.F;
      solver_->iteration_complete_signal(type_id);
    } else {
      au_ids.emplace(param_id);
    }
  }
  DLOG(INFO) << "[" << Caffe::current_device()
             << "] Leaving ReduceAndUpdate thread " << lwp_id();
}

void Net::add_wgrad_sq(float wgrad_sq) {
  if (wgrad_sq > 0.F) {
    wgrad_sq_.fetch_add(std::llround(wgrad_sq * GRAD_FACTOR));
  }
}

float Net::wgrad_sq() {
  return wgrad_sq_.exchange(0LL) / GRAD_FACTOR;
}

void Net::update_grad_scale() {
  global_grad_scale_coeff_ = 1.F;
  if (global_grad_scale_enabled()) {
    if (global_grad_scale_adaptive_) {
      const float wgsq = wgrad_sq();
      if (wgsq > 0.F) {
        global_grad_scale_coeff_ = std::sqrt(wgsq) * global_grad_scale_param_;
        return;
      }
    }
    global_grad_scale_coeff_ = global_grad_scale_param_;
  }
}

void Net::Reduce(int type_id, int param_id) {
  Solver::Callback* cb = solver_->callback();
  cb->reduce_barrier(type_id);
  {
    unique_ptr<unique_lock<shared_mutex>> lock;
    if (solver_->is_root()) {
      lock.reset(new unique_lock<shared_mutex>(GPUMemory::read_write_mutex()));
    }
    cb->reduce_barrier(type_id);
    cb->allreduce(type_id, param_id);
    cb->reduce_barrier(type_id);
  }
  this->learnable_params()[param_id]->
      scale_diff(1.F / (Caffe::solver_count() * global_grad_scale()),
      Caffe::cublas_handle(type_id));
  // Also need to barrier to make sure lock isn't undone
  // until all have completed, but the current nature of
  // NCCL makes this unnecessary.
  // solver_->callback()->reduce_barrier();
}

void Net::ReduceBucket(int type_id, size_t count, Type bucket_type, void* bucket) {
  Solver::Callback* cb = solver_->callback();
  cb->reduce_barrier(type_id);
  {
    unique_ptr<unique_lock<shared_mutex>> lock;
    if (solver_->is_root()) {
      lock.reset(new unique_lock<shared_mutex>(GPUMemory::read_write_mutex()));
    }
    cb->reduce_barrier(type_id);
    cb->allreduce_bucket(type_id, count, bucket, bucket_type);
    cb->reduce_barrier(type_id);
  }
  Tensor::gpu_scal(count, bucket_type, bucket, 1.F / (Caffe::solver_count() * global_grad_scale()),
      Caffe::cublas_handle(type_id));
}

void Net::ForwardDebugInfo(const int layer_id) {
  LOG_IF(INFO, Caffe::root_solver())
      << "[Forward] Layer " << layer_names_[layer_id];
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const double data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> top blob " << blob_name
        << ", count: " << blob.count()
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const double data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> param blob " << blob_name
        << ", count: " << blob.count()
        << " data: " << data_abs_val_mean;
  }
}

void Net::BackwardDebugInfo(const int layer_id) {
  LOG_IF(INFO, Caffe::root_solver())
      << "[Backward] Layer " << layer_names_[layer_id];
  const vector<Blob*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const double diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> bottom blob " << blob_name
        << ", count: " << blob.count()
        << ", diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob& blob = *layers_[layer_id]->blobs()[param_id];
    double diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> param blob " << param_id
        << ", count: " << blob.count()
        << ", diff: " << diff_abs_val_mean;
  }
}

void Net::UpdateDebugInfo(const int param_id) {
  const Blob& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const double diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    double data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

void Net::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    LayerBase* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
  trained_layers_shared_ = true;
}

void Net::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

void Net::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

void Net::Backward(bool apply_update) {
  BackwardFromToAu(layers_.size() - 1, 0, apply_update);
  if (debug_info_) {
    float asum_data = 0.F, asum_diff = 0.F, sumsq_data = 0.F, sumsq_diff = 0.F;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const double l2norm_data = std::sqrt(sumsq_data);
    const double l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

void Net::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

void Net::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    const string& source_layer_type = source_layer.type();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    LOG(INFO) << "Copying source layer " << source_layer_name << " Type:"
              << source_layer_type << " #blobs=" << source_layer.blobs_size();
    // check if BN is in legacy DIGITS format?
    if (source_layer_type == "BatchNorm" && source_layer.blobs_size() == 5) {
      for (int j = 0; j < target_blobs.size(); ++j) {
        const bool kReshape = true;
        target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
        DLOG(INFO) << target_blobs[j]->count();
      }
      if (target_blobs[4]->count() == 1) {
        // old format: 0 - scale , 1 - bias,  2 - mean , 3 - var, 4 - reserved
        // new format: 0 - mean  , 1 - var,  2 - reserved , 3- scale, 4 - bias
        LOG(INFO) << "BN legacy DIGITS format detected ... ";
        std::swap(target_blobs[0], target_blobs[2]);
        std::swap(target_blobs[1], target_blobs[3]);
        // ==> 0 - mean , 1 -var,  2 - scale , 3 - bias; 4 - reserved
        std::swap(target_blobs[2], target_blobs[4]);
        std::swap(target_blobs[3], target_blobs[4]);
        LOG(INFO) << "BN Transforming to new format completed.";
      }
    } else {
      for (int j = 0; j < target_blobs.size(); ++j) {
        if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
          shared_ptr<Blob> source_blob = Blob::create(target_blobs[j]->data_type(),
              target_blobs[j]->diff_type());
          const bool kReshape = true;
          source_blob->FromProto(source_layer.blobs(j), kReshape);
          LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
              << source_layer_name << "'; shape mismatch.  Source param shape is "
              << source_blob->shape_string() << "; target param shape is "
              << target_blobs[j]->shape_string() << ". "
              << "To learn this layer's parameters from scratch rather than "
              << "copying from a saved net, rename the layer.";
        }
        const bool kReshape = false;
        target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
      }
    }
  }
}

void Net::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

void Net::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

void Net::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

void Net::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

void Net::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

void Net::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

void Net::ClearParamDiffs() {
  if (Caffe::mode() == Caffe::GPU) {
    caffe_gpu_memset(learnable_space_[0].size(), 0, learnable_space_[0].data());
    caffe_gpu_memset(learnable_space_[1].size(), 0, learnable_space_[1].data());
  } else {
    for (int i = 0; i < learnable_params_.size(); ++i) {
      learnable_params_[i]->set_diff(0.F);
    }
  }
}

void Net::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) {
      gpu_prm_memory_data_use_ += params_[i]->gpu_memory_data_use();
      gpu_prm_memory_diff_use_ += params_[i]->gpu_memory_diff_use();
      continue;
    }
//    DLOG(INFO) << "param " << i << " has owner " << param_owners_[i];
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
    gpu_shp_memory_data_use_ += params_[i]->gpu_memory_data_use();
    gpu_shp_memory_diff_use_ += params_[i]->gpu_memory_diff_use();
  }
}

bool Net::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

const shared_ptr<Blob> Net::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob> blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

bool Net::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

const shared_ptr<LayerBase> Net::layer_by_name(
    const string& layer_name) const {
  shared_ptr<LayerBase> layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

void Net::set_solver(Solver* s) {
  solver_ = s;
  for (auto& layer : layers_) {
    layer->set_parent_net(this);
  }
}

void Net::InitializeLearnableDiffSpace(int type_id) {
  CHECK_GE(type_id, 0);
  CHECK_LT(type_id, 2);
  const Type t = (Type) learnable_types_[type_id];
  if (learnable_params_ptrs_[type_id].size() == learnable_params_.size()) {
    LOG(INFO) << print_current_device() << " Already reserved "
              << learnable_space_size_[type_id] << " bytes of shared learnable space for type "
              << Type_Name(t);
    return;
  }
  learnable_space_size_[type_id] = 0UL;
  learnable_params_ptrs_[type_id].resize(learnable_params_.size(), nullptr);
  for (int i = 0; i < layers_.size(); ++i) {
    for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
      if (!layers_[i]->skip_apply_update(j)) {
        const int lip = layer_index_params_[make_pair(i, j)];
        if (param_owners_[lip] < 0) {
          const int param_id = learnable_param_ids_[lip];
          if (learnable_params_[param_id]->diff_type() == t) {
            learnable_space_size_[type_id] += lp_aligned_count(param_id) * lp_size(param_id);
          }
        }
      }
    }
  }
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters. Times two.
  if (learnable_space_size_[type_id] < 2) {
    learnable_space_size_[type_id] = 2;
  }
  LOG(INFO) << print_current_device() << " Reserving "
            << learnable_space_size_[type_id] << " bytes of shared learnable space for type "
            << Type_Name(t);
  learnable_space_[type_id].reserve(learnable_space_size_[type_id]);
  unsigned char* ptr = reinterpret_cast<unsigned char*>(learnable_space_[type_id].data());
  caffe_gpu_memset(learnable_space_size_[type_id], 0, ptr);
  for (int i = 0; i < layers_.size(); ++i) {
    for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
      if (!layers_[i]->skip_apply_update(j)) {
        const int lip = layer_index_params_[make_pair(i, j)];
        if (param_owners_[lip] < 0) {
          const int param_id = learnable_param_ids_[lip];
          if (learnable_params_[param_id]->diff_type() == t) {
            learnable_params_[param_id]->set_gpu_diff(ptr);
            learnable_params_ptrs_[type_id][param_id] = static_cast<void*>(ptr);
            ptr += lp_aligned_count(param_id) * lp_size(param_id);
            learnable_params_mapped_.push_back(learnable_params_[param_id]);
            ltop_[type_id][i].insert(param_id);
            void *p = learnable_params_[param_id]->
                current_mutable_data_memory(Caffe::mode() == Caffe::GPU);
            (void) p;
          }
        }
      } else {
        DLOG(INFO) << print_current_device()
            << "** Skipping non-learnable blob from " << layers_[i]->name()
            << " of type " << layers_[i]->type();
      }
    }
  }
}

const vector<Type>& Net::learnable_types(bool reset) {
  if (reset || learnable_types_.empty()) {
    learnable_types_.clear();
    int type0 = -1;
    int type1 = -1;
    for (shared_ptr<Blob> lp : learnable_params_) {
      Type t = lp->diff_type();
      if (type0 < 0) {
        type0 = (int) t;
        learnable_types_.push_back(t);
      } else if (type1 < 0 && type0 != (int) t) {
        type1 = (int) t;
        learnable_types_.push_back(t);
      }
    }
    if (learnable_types_.empty() && solver_ != nullptr) {
      learnable_types_.push_back(solver_->data_type());
    }
    CHECK_LE(learnable_types_.size(), 2);
  }
  return learnable_types_;
}

}  // namespace caffe
