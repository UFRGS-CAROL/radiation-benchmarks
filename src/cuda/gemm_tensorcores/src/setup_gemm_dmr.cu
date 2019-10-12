#include "setup_template.h"

template<class half_t, class real_t = half_t>
void hardware_dmr_template(Log& log_obj) {
	//TODO
//	HostVectors<real_t, real_t, real_t> hd(
//			log_obj.size_matrices * log_obj.size_matrices);
//
//	hd.load_matrices_files(log_obj);
//
//	std::shared_ptr<GEMMBase<real_t, real_t, real_t> > mt;
//	mt = std::make_shared<GEMMWMMADMR<real_t>>(hd.host_matrix_a,
//			hd.host_matrix_b, hd.host_matrix_c, hd.host_matrix_d,
//			log_obj.size_matrices, real_t(log_obj.alpha),
//			real_t(log_obj.beta));
//	setup_execute(mt, log_obj, hd);
}

template<const uint32_t COUNT, class real_t>
void software_dmr_template(Log& log_obj) {
	HostVectors<real_t, real_t, real_t> hd(
			log_obj.size_matrices * log_obj.size_matrices);

	hd.load_matrices_files(log_obj);

	GEMMDMR<COUNT, real_t> mt(hd.host_matrix_a, hd.host_matrix_b,
			hd.host_matrix_c, hd.host_matrix_d, log_obj.size_matrices,
			real_t(log_obj.alpha), real_t(log_obj.beta));

	//For the same type the threshold is also 0
	setup_execute<COUNT, 0>(mt, log_obj, hd);
}

template<class real_t>
void software_dmr_template_count_select(Log& log_obj) {
	switch (log_obj.check_block) {
	case 1:
		software_dmr_template<1, real_t>(log_obj);
		break;
	case 16:
		software_dmr_template<16, real_t>(log_obj);
		break;
	case BLOCK_SIZE:
		software_dmr_template<BLOCK_SIZE, real_t>(log_obj);
		break;
	default:
		throw_line(
				std::to_string(log_obj.check_block)
						+ " count operations not supported");
	}

}

void call_mxm_dmr(Log& log_obj, GEMMTYPE gemm_t) {
	switch (gemm_t) {
	case DMRGEMM:
		if (log_obj.precision == "half") {
			software_dmr_template_count_select<half>(log_obj);
		}

		if (log_obj.precision == "float") {
			software_dmr_template_count_select<float>(log_obj);
		}

		if (log_obj.precision == "double") {
			software_dmr_template_count_select<double>(log_obj);
		}
		break;
	case DMRWMA:
		if (log_obj.precision == "half") {
			hardware_dmr_template<half>(log_obj);
		}

		if (log_obj.precision == "float") {
			hardware_dmr_template<half, float>(log_obj);
		}
		break;

	case NONDMRGEMM:
	case DMRGEMMMIXED:
	case NONDMRWMMA:
		throw_line("Please, select the correct hardened option!");
	}
}
