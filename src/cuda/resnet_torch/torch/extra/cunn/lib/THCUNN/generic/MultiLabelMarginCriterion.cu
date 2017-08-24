#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MultiLabelMarginCriterion.cu"
#else

// TODO: improve error messages
void THNN_(MultiLabelMarginCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           THCTensor *istarget,
           bool sizeaverage)
{
  input = THCTensor_(newContiguous)(state, input);
  target = THCIndexTensor_(newContiguous)(state, target);
  istarget = THCTensor_(newContiguous)(state, istarget);
  THCTensor_(resizeAs)(state, istarget, input);

  if(input->nDimension == 1)
  {
    int dim = input->size[0];
    THArgCheck((target->nDimension == 1) && (target->size[0] == dim), 3,
        "inconsistent target size");
    THCTensor_(resize1d)(state, output, 1);

    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<real, accreal> <<<blocks,threads>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        1, dim,
        sizeaverage
        );
    THCudaCheck(cudaGetLastError());
  }
  else if(input->nDimension == 2)
  {
    int nframe = input->size[0];
    int dim = input->size[1];
    THArgCheck((target->nDimension == 2) && (target->size[0] == nframe)
               && (target->size[1] == dim), 3, "inconsistent target size");
    THCTensor *output_tmp = THCTensor_(newWithSize1d)(state, input->size[0]);

    dim3 blocks(input->size[0]);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<real, accreal> <<<blocks,threads>>>(
        THCTensor_(data)(state, output_tmp),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        nframe, dim,
        sizeaverage
        );
    THCudaCheck(cudaGetLastError());
    THCTensor_(resize1d)(state, output, 1);
    THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(THCTensor_(sumall)(state, output_tmp)));
    THCTensor_(free)(state, output_tmp);
  }
  else
    THError("vector or matrix expected");

  THCTensor_(free)(state, input);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, istarget);
}

void THNN_(MultiLabelMarginCriterion_updateGradInput)(
            THCState *state,
            THCTensor *input,
            THCIndexTensor *target,
            THCTensor *gradInput,
            THCTensor *istarget,
            bool sizeaverage)
{
  input = THCTensor_(newContiguous)(state, input);
  target = THCIndexTensor_(newContiguous)(state, target);
  istarget = THCTensor_(newContiguous)(state, istarget);
  THCTensor_(resizeAs)(state, gradInput, input);

  if(gradInput->nDimension == 1)
  {
    int dim = gradInput->size[0];
    THArgCheck((target->nDimension == 1) && (target->size[0] == dim), 3,
               "inconsistent target size");
    THArgCheck((istarget->nDimension == 1) && (istarget->size[0] == dim), 3,
               "inconsistent isTarget size");
    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<real, accreal> <<<blocks,threads>>>(THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        1, gradInput->size[0],
        sizeaverage);

  }
  else if(gradInput->nDimension == 2)
  {
    int nframe = gradInput->size[0];
    int dim = gradInput->size[1];
    THArgCheck((target->nDimension == 2) && (target->size[0] == nframe)
               && (target->size[1] == dim), 3, "inconsistent target size");
    THArgCheck((istarget->nDimension == 2) && (istarget->size[0] == nframe)
               && (istarget->size[1] == dim), 3, "inconsistent isTarget size");
    dim3 blocks(gradInput->size[0]);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<real, accreal> <<<blocks,threads>>>(THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        gradInput->size[0], gradInput->size[1],
        sizeaverage);
  }
  else
    THError("vector or matrix expected");

  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, istarget);
}

#endif
