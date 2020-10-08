/*
 * parse_gemm_layer.h
 *
 *  Created on: Apr 11, 2020
 *      Author: fernando
 */

#ifndef PARSE_GEMM_LAYER_H_
#define PARSE_GEMM_LAYER_H_

typedef enum {
    GENERATE_GOLDEN_LAYERS,
    COMPARING_CURRENT_TO_GOLDEN,
    INJECT_FAULT_IN_OUTPUT,
} LayerOperationType;

void set_layer_processing_parameters(
        const std::string& base_path,
        LayerOperationType current_operation
        );
void reset_counters();

#endif /* PARSE_GEMM_LAYER_H_ */
