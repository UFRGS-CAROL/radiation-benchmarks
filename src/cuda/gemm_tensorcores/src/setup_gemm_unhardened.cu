

#include "setup_template.h"
#include "GEMM.h"
//#include "GEMMWMMA.h"

/**
 * For WMMA without hardening
 */
template<class half_t, class real_t = half_t>
void hardware_unhardened_template(Log& log_obj) {
	//TODO
}

template<class real_t>
void software_unhardened_template(Log& log_obj) {
	HostVectors<real_t, real_t, real_t> hd(
			log_obj.size_matrices * log_obj.size_matrices);

	hd.load_matrices_files(log_obj);

	GEMM<real_t> mt(hd.host_matrix_a, hd.host_matrix_b, hd.host_matrix_c,
			hd.host_matrix_d, log_obj.size_matrices, real_t(log_obj.alpha),
			real_t(log_obj.beta));

	//For Unhardened THRESHOLD is 0
	setup_execute(mt, log_obj, hd);
}

void call_mxm_unhardened(Log& log_obj, GEMMTYPE gemm_t) {
	switch (gemm_t) {
	case NONDMRGEMM:
		if (log_obj.precision == "half") {
			software_unhardened_template<half>(log_obj);
		}

		if (log_obj.precision == "float") {
			software_unhardened_template<float>(log_obj);
		}

		if (log_obj.precision == "double") {
			software_unhardened_template<double>(log_obj);
		}
		break;
	case NONDMRWMMA:
		if (log_obj.precision == "half") {
			hardware_unhardened_template<half>(log_obj);
		}

		if (log_obj.precision == "float") {
			hardware_unhardened_template<half, float>(log_obj);
		}
		break;

	case DMRGEMM:
	case DMRGEMMMIXED:
	case DMRWMA:
		throw_line("Please, select the correct hardened option!");
	}

}
