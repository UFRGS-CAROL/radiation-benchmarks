

//template<class half_t, class real_t, class mixed_real_t>
//void call_mxm(Log& log_obj, GEMMTYPE gemm_t) {
//	HostVectors<real_t, real_t, mixed_real_t> hd(
//			log_obj.size_matrices * log_obj.size_matrices);
//
//	hd.load_matrices_files(log_obj);
//
//	std::shared_ptr<GEMMBase<real_t, real_t, mixed_real_t> > mt;
//	switch (gemm_t) {
//	case DMRWMA:
//	case NONDMRWMMA:
//	case NONDMRGEMM:
//	case DMRGEMM:
//		throw_line("Not implemented!");
//		break;
//	case DMRGEMMMIXED:
//		mt = std::make_shared<GEMMDMRMIXED<real_t, real_t, mixed_real_t>>(
//				hd.host_matrix_a, hd.host_matrix_b, hd.host_matrix_c,
//				hd.host_matrix_d, log_obj.size_matrices, real_t(log_obj.alpha),
//				real_t(log_obj.beta));
//		break;
//	}
//
//	setup_execute(mt, log_obj, hd);
//}
