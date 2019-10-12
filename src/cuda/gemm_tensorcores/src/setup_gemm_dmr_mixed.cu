#include "setup_template.h"
#include "GEMM.h"
#include "Log.h"
//#include "GEMMWMMA.h"

template<const uint32_t COUNT, const uint32_t THRESHOLD, class half_t,
		class real_t>
void software_dmr_mixed_template(Log& log_obj) {
	HostVectors<real_t, real_t, half_t> hd(
			log_obj.size_matrices * log_obj.size_matrices);

	hd.load_matrices_files(log_obj);

	GEMMDMRMIXED<COUNT, THRESHOLD, real_t, half_t> mt(hd.host_matrix_a,
			hd.host_matrix_b, hd.host_matrix_c, hd.host_matrix_d,
			log_obj.size_matrices, real_t(log_obj.alpha), real_t(log_obj.beta));

	//For the same type the threshold is also 0
	setup_execute<COUNT, THRESHOLD, real_t, real_t, half_t>(mt, log_obj, hd);
}

template<class half_t, class real_t>
void software_dmr_template_count_select(Log& log_obj) {
	switch (log_obj.check_block) {
	case 1:
		software_dmr_mixed_template<1, THRESHOLD_1, half_t, real_t>(log_obj);
		break;
	case 16:
		software_dmr_mixed_template<16, THRESHOLD_16, half_t, real_t>(log_obj);
		break;
	case BLOCK_SIZE:
		software_dmr_mixed_template<BLOCK_SIZE, THRESHOLD_32, half_t, real_t>(
				log_obj);
		break;
	default:
		throw_line(
				std::to_string(log_obj.check_block)
						+ " count operations not supported");
	}

}

void call_mxm_dmr_mixed(Log& log_obj, GEMMTYPE gemm_t) {
	switch (gemm_t) {
	case DMRGEMMMIXED:
		if (log_obj.precision == "float") {
//			software_dmr_template_count_select<float, half>(log_obj);
		}

		if (log_obj.precision == "double") {
			software_dmr_template_count_select<float, double>(log_obj);
		}
		break;
	case DMRWMA:
	case NONDMRGEMM:
	case DMRGEMM:
	case NONDMRWMMA:
		throw_line("Please, select the correct hardened option!");
	}
}
