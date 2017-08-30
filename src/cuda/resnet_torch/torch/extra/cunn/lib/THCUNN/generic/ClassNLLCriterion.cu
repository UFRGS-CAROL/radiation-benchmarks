#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ClassNLLCriterion.cu"
#else

void THNN_(ClassNLLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           bool sizeAverage,
           THCTensor *weights,
           THCTensor *total_weight,
           long ignore_index) {
  THCUNN_check_dim_size(state, output, 1, 0, 1);
  THCUNN_check_dim_size(state, total_weight, 1, 0, 1);
  ignore_index -= TH_INDEX_BASE;

  if (THCIndexTensor_(nDimension)(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCTensor_(nDimension)(state, input);
  int n_classes = THCTensor_(size)(state, input, n_dims - 1);

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, input, target, weights, output, total_weight
    );
  } else {
    THCUNN_assertSameGPU(
      state, 4, input, target, output, total_weight
    );
  }

  THArgCheck(n_dims <= 2 && n_dims > 0, 2, "vector or matrix expected");

  long batch_size = n_dims == 1 ? 1 : THCTensor_(size)(state, input, 0);
  long num_targets = THCudaLongTensor_size(state, target, 0);
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THCTensor_(nElement)(state, weights) != n_classes) {
    THCDescBuff s1 = THCTensor_(sizeDesc)(state, weights);
    THError("weight tensor should be defined either for all %d classes or no classes"
            " but got weight tensor of shape: %s", n_classes, s1.str);
  }

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  real *input_data = THCTensor_(data)(state, input);
  real *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  real *output_data = THCTensor_(data)(state, output);
  real *total_weight_data = THCTensor_(data)(state, total_weight);

  if (THCTensor_(nDimension)(state, input) == 1) {
    cunn_ClassNLLCriterion_updateOutput_kernel1<real>
      <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        weights_data,
        sizeAverage,
        n_classes,
        ignore_index
    );

  } else if (THCTensor_(nDimension)(state, input) == 2) {
    cunn_ClassNLLCriterion_updateOutput_kernel<real, accreal>
      <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        weights_data,
        sizeAverage,
        THCTensor_(size)(state, input, 0),
        THCTensor_(size)(state, input, 1),
        n_classes,
        ignore_index
    );
  }
  THCudaCheck(cudaGetLastError());

  if (weights) {
    THCTensor_(free)(state, weights);
  }
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, input);
}

void THNN_(ClassNLLCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *gradInput,
           bool sizeAverage,
           THCTensor *weights,
           THCTensor *total_weight,
           long ignore_index) {
  if (THCIndexTensor_(nDimension)(state, target) > 1) {
    THError("multi-target not supported");
  }
  ignore_index -= TH_INDEX_BASE;

  int n_dims = THCTensor_(nDimension)(state, input);
  int n_classes = THCTensor_(size)(state, input, n_dims - 1);

  THArgCheck(THCTensor_(isContiguous)(state, gradInput), 4, "gradInput must be contiguous");

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, weights, input, target, gradInput, total_weight
    );
  }
  else {
    THCUNN_assertSameGPU(
      state, 4, input, target, gradInput, total_weight
    );
  }

  THArgCheck(n_dims <= 2 && n_dims > 0, 2, "vector or matrix expected");

  long batch_size = n_dims == 1 ? 1 : THCTensor_(size)(state, input, 0);
  long num_targets = THCudaLongTensor_size(state, target, 0);
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THCTensor_(nElement)(state, weights) != n_classes) {
    THError("weight tensor should be defined either for all or no classes");
  }

  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  real *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  real *gradInput_data = THCTensor_(data)(state, gradInput);
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  real *total_weight_data = THCTensor_(data)(state, total_weight);

  if (THCTensor_(nDimension)(state, input) == 1) {
    cunn_ClassNLLCriterion_updateGradInput_kernel1<real>
      <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        gradInput_data,
        weights_data,
        target_data,
        total_weight_data,
        sizeAverage,
        n_classes,
        ignore_index
    );
  } else {
    cunn_ClassNLLCriterion_updateGradInput_kernel<real>
      <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
        gradInput_data,
        target_data,
        weights_data,
        total_weight_data,
        sizeAverage,
        THCTensor_(size)(state, input, 0),
        THCTensor_(size)(state, input, 1),
        n_classes,
        ignore_index
    );
  }
  THCudaCheck(cudaGetLastError());

  if (weights) {
    THCTensor_(free)(state, weights);
  }
  THCIndexTensor_(free)(state, target);
}

#endif
