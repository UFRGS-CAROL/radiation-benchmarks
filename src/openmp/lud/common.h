#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Define the precision to float or double depending on compiling flags */
#if PRECISION == 32
typedef float FP;
#define PRECISION_STR "Float"
#endif
#if PRECISION == 64
typedef double FP;
#define PRECISION_STR "Double"
#endif
/* Default to double if no compiling flags are used */
#ifndef PRECISION_STR
typedef double FP;
#define PRECISION_STR "Double"
#endif

#define GET_RAND_FP ( (FP)rand() /   \
                     ((FP)(RAND_MAX)+(FP)(1)) )

#define MIN(i,j) ((i)<(j) ? (i) : (j))



typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
} func_ret_t;

typedef struct __stopwatch_t {
    struct timeval begin;
    struct timeval end;
} stopwatch;

void
stopwatch_start(stopwatch *sw);

void
stopwatch_stop (stopwatch *sw);

double
get_interval_by_sec(stopwatch *sw);

func_ret_t
create_matrix_from_file(FP **mp, const char *filename, int *size_p);

func_ret_t
create_matrix(FP **mp, int size);

func_ret_t
lud_verify(FP *m, FP *lu, int size);

void
matrix_duplicate(FP *src, FP **dst, int matrix_dim);

#ifdef __cplusplus
}
#endif

#endif
