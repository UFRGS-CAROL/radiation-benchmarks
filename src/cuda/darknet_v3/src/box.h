#ifndef BOX_H
#define BOX_H
#include "darknet.h"
#include "type.h"

typedef struct {
	real_t dx, dy, dw, dh;
} dbox;

real_t box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
