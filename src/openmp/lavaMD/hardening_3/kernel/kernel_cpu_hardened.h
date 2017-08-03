#ifdef __cplusplus
extern "C" {
#endif

void  kernel_cpu(	par_str par,
                    dim_str dim,
                    box_str* box,
                    FOUR_VECTOR* rv_hardened_1,
	            FOUR_VECTOR* rv_hardened_2,
                    fp* qv,
                    FOUR_VECTOR* fv);

#ifdef __cplusplus
}
#endif
