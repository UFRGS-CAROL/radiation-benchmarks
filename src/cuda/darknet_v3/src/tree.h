#ifndef TREE_H
#define TREE_H
#include "darknet.h"

int hierarchy_top_prediction(real_t *predictions, tree *hier, real_t thresh,
		int stride);
real_t get_hierarchy_probability(real_t *x, tree *hier, int c, int stride);

#endif
