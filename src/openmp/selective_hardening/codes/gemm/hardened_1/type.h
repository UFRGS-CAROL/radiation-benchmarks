/*
 * type.h
 *
 *  Created on: 14/01/2019
 *      Author: fernando
 */

#ifndef TYPE_H_
#define TYPE_H_

#if PRECISION == 32
typedef float real_t;
#define PRECISION_STR "Float"
#endif

#if PRECISION == 64
typedef double real_t;
#define PRECISION_STR "Double"
#endif



#endif /* TYPE_H_ */
