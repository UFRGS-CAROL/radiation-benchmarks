#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

#include <iomanip>
#include <limits>
#include  <memory>
#include <stdexcept>

#ifdef OMP
#include <omp.h>
#endif

#include "Log.h"
#include "GEMM.h"
#include "GEMMWMMA.h"

#ifndef DEFAULT_INPUT_SIZE
#define DEFAULT_INPUT_SIZE 8192
#endif

#define GENERATOR_MAXABSVALUE 0.9
#define GENERATOR_MINABSVALUE 0.0001

typedef double BiggestPrecision;

template<class half_t, class real_t, class mixed_t>
struct HostVectors {
	// Matrices A and B
	std::vector<half_t> host_matrix_a;
	std::vector<half_t> host_matrix_b;

// C matrix
	std::vector<real_t> host_matrix_c;

// D Matrix
	std::vector<real_t> host_matrix_d;

	std::vector<real_t> host_gold;
	std::vector<mixed_t> host_matrix_smaller;

	HostVectors(int matrix_size) {
		// Matrices A and B
		this->host_matrix_a = std::vector < half_t > (matrix_size);
		this->host_matrix_b = std::vector < half_t > (matrix_size);

		// C matrix
		this->host_matrix_c = std::vector < real_t > (matrix_size);

		// D Matrix
		this->host_matrix_d = std::vector < real_t > (matrix_size, 0);

		this->host_gold = std::vector < real_t > (matrix_size);
		this->host_matrix_smaller = std::vector < mixed_t > (matrix_size, 0);
	}

	void load_matrices_files(Log& log) {
		if (!log.generate) {
			this->retrieve_matrices(log);
		} else {
			this->generate_matrices_files(log);
		}
	}

	void generate_matrices_files(Log& log) {

		std::ofstream f_a(log.a_input_path, std::ios::out | std::ios::binary);
		std::ofstream f_b(log.b_input_path, std::ios::out | std::ios::binary);
		std::ofstream f_c(log.c_input_path, std::ios::out | std::ios::binary);

		if (f_a.is_open() && f_b.is_open() && f_c.is_open()) {
			std::random_device rd; //Will be used to obtain a seed for the random number engine
			std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
			std::uniform_real_distribution<double> dis(GENERATOR_MINABSVALUE,
			GENERATOR_MAXABSVALUE);

			for (size_t i = 0; i < log.size_matrices; i++) {
				for (size_t j = 0; j < log.size_matrices; j++) {
					//Make sure that it is not 0
					host_matrix_a[i * log.size_matrices + j] = half_t(dis(gen));
					host_matrix_b[i * log.size_matrices + j] = half_t(dis(gen));
					host_matrix_c[i * log.size_matrices + j] = real_t(dis(gen));
				}
			}

			f_a.write(reinterpret_cast<char*>(host_matrix_a.data()),
					host_matrix_a.size() * sizeof(half_t));
			f_b.write(reinterpret_cast<char*>(host_matrix_b.data()),
					host_matrix_b.size() * sizeof(half_t));
			f_c.write(reinterpret_cast<char*>(host_matrix_c.data()),
					host_matrix_c.size() * sizeof(real_t));

			f_a.close();
			f_b.close();
			f_c.close();

		} else {
			throw_line("Some of the imput files could not be generated\n");
		}

	}

	void write_gold_to_file(std::string gold_path) {
		std::ofstream f_gold(gold_path,
				std::ofstream::out | std::ofstream::binary);

		if (f_gold.is_open()) {
			f_gold.write(reinterpret_cast<char*>(this->host_matrix_d.data()),
					sizeof(real_t) * host_gold.size());
			f_gold.close();
		} else {
			throw_line("Could not write gold file\n");
		}
	}

	void retrieve_matrices(Log& log) {

		double start = log.mysecond();
		std::ifstream f_a(log.a_input_path, std::ios::in | std::ios::binary);
		std::ifstream f_b(log.b_input_path, std::ios::in | std::ios::binary);
		std::ifstream f_c(log.c_input_path, std::ios::in | std::ios::binary);
		std::ifstream f_gold(log.gold_inout_path,
				std::ifstream::in | std::ifstream::binary);

		if (f_a.is_open() && f_b.is_open() && f_c.is_open() && f_gold) {

			f_a.seekg(0, std::ios::beg);
			f_a.read(reinterpret_cast<char*>(host_matrix_a.data()),
					sizeof(half_t) * host_matrix_a.size());

			f_b.seekg(0, std::ios::beg);
			f_b.read(reinterpret_cast<char*>(host_matrix_b.data()),
					sizeof(half_t) * host_matrix_b.size());

			f_c.seekg(0, std::ios::beg);
			f_c.read(reinterpret_cast<char*>(host_matrix_c.data()),
					sizeof(real_t) * host_matrix_c.size());

			f_gold.seekg(0, std::ios::beg);
			f_gold.read(reinterpret_cast<char*>(host_gold.data()),
					sizeof(real_t) * host_gold.size());

			f_a.close();
			f_b.close();
			f_c.close();
			f_gold.close();
		} else {
			log.log_error("Could not retrieve the matrices");
			throw_line("Could not retrieve the matrices\n");
		}

		std::cout << "Done with reading matrices " << log.mysecond() - start
				<< "s\n";
	}
};

bool cmp(const BiggestPrecision lhs, const BiggestPrecision rhs, Log& log) {
	const BiggestPrecision diff = abs(lhs - rhs);
	BiggestPrecision zero;

	if (log.use_tensor_cores) {
		zero = BiggestPrecision(ZERO_HALF);
	} else {
		if (log.precision == "float")
			zero = BiggestPrecision(ZERO_FLOAT);

		if (log.precision == "double")
			zero = BiggestPrecision(0.0002);
	}

	if (diff > zero) {
		return false;
	}
	return true;
}

template<class half_t, class real_t>
std::pair<int, int> check_output_errors_dmr(std::vector<real_t>& gold,
		std::vector<half_t>& d0, std::vector<real_t>& d1, Log& log) {
	int host_errors = 0;
	double threshold = -3222;
#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		BiggestPrecision gold_value = gold[i];
		BiggestPrecision half_precision = d0[i];
		BiggestPrecision full_precision = d1[i];
		threshold = std::fmax(threshold, fabs(half_precision - full_precision));
		if (gold_value != full_precision || !cmp(half_precision, full_precision, log)) {
#ifdef OMP
#pragma omp critical
			{
#endif

			std::stringstream error_detail("");
			error_detail << std::setprecision(20) << std::scientific;
			error_detail << "p: [" << int(floor(i / log.size_matrices)) << ", "
					<< i % log.size_matrices << "], r: " << full_precision
					<< ", e: " << gold_value << " smaller_precision: "
					<< half_precision;

			if (log.verbose && (host_errors < 10))
				std::cout << error_detail.str() << std::endl;

			log.log_error(error_detail.str());
			host_errors++;
#ifdef OMP
			}
#endif
		}
	}
	std::cout << "THRESHOLD 0 " << threshold << std::endl;
	auto dmr_err = dmr_errors();
	if (dmr_err != 0) {
		std::string error_detail;
		error_detail = "detected_dmr_errors: " + std::to_string(dmr_err);
		log.log_error(error_detail);
	}

	log.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	std::pair<int, int> res(dmr_err, host_errors);
	return res;
}

template<class half_t, class real_t, class mixed_t>
void setup_execute(
		std::shared_ptr<GEMMBase<half_t, real_t, mixed_t>> mult_enviroment,
		Log& log_obj, HostVectors<half_t, real_t, mixed_t>& hd) {
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	for (int it = 0; it < log_obj.iterations; it++) {
		double start_computation = log_obj.mysecond();
		log_obj.start_iteration_app();
		mult_enviroment->gemm();
		log_obj.end_iteration_app();
		double end_computation = log_obj.mysecond();

		mult_enviroment->pull_array(hd.host_matrix_d, hd.host_matrix_smaller);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time : %f ms\n", elapsedTime);

		if (!log_obj.generate) {
			std::pair<int, int> errors;
			double start, end;

			start = log_obj.mysecond();
			errors = check_output_errors_dmr(hd.host_gold,
					hd.host_matrix_smaller, hd.host_matrix_d, log_obj);
			end = log_obj.mysecond();

			std::cout << "Iteration: " << it << " dmr errors "
					<< errors.first << " radiation errors " << errors.second
					<< ". Time spent on computation "
					<< end_computation - start_computation
					<< "s. Time spent on comparing " << end - start << "s."
					<< std::endl;

			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				mult_enviroment->push_arrays(hd.host_matrix_a, hd.host_matrix_b,
						hd.host_matrix_c);
			}

		}

	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time : %f s\n", (elapsedTime / 1000));
	if (log_obj.generate) {
		hd.write_gold_to_file(log_obj.gold_inout_path);
	}
}

template<class half_t, class real_t>
void call_mxm(Log& log_obj, GEMMTYPE gemm_t) {
	HostVectors<half_t, real_t, half_t> hd(
			log_obj.size_matrices * log_obj.size_matrices);

	hd.load_matrices_files(log_obj);

	std::shared_ptr<GEMMBase<half_t, real_t, half_t>> mt;
	switch (gemm_t) {
	case NONDMRGEMM:
	case DMRWMA:
		throw_line("Not implemented!");
		break;
	case NONDMRWMMA:
		mt = std::make_shared<GEMMWMMAMIXED<half_t, real_t>>(hd.host_matrix_a,
				hd.host_matrix_b, hd.host_matrix_c, hd.host_matrix_d,
				log_obj.size_matrices, real_t(log_obj.alpha),
				real_t(log_obj.beta), gemm_t);
		break;
	}

	setup_execute(mt, log_obj, hd);
}

template<class real_t>
void call_mxm(Log& log_obj, GEMMTYPE gemm_t) {
	HostVectors<real_t, real_t, real_t> hd(
			log_obj.size_matrices * log_obj.size_matrices);

	hd.load_matrices_files(log_obj);

	std::shared_ptr<GEMMBase<real_t, real_t, real_t> > mt;
	switch (gemm_t) {
	case NONDMRGEMM:
		mt =
				std::make_shared < GEMM
						< real_t
								>> (hd.host_matrix_a, hd.host_matrix_b, hd.host_matrix_c, hd.host_matrix_d, log_obj.size_matrices, real_t(
										log_obj.alpha), real_t(log_obj.beta), gemm_t);
		break;

	case DMRGEMM:
		mt =
				std::make_shared < GEMMDMR
						< real_t
								>> (hd.host_matrix_a, hd.host_matrix_b, hd.host_matrix_c, hd.host_matrix_d, log_obj.size_matrices, real_t(
										log_obj.alpha), real_t(log_obj.beta), gemm_t);
		break;

	case DMRWMA:
		mt =
				std::make_shared < GEMMWMMADMR
						< real_t
								>> (hd.host_matrix_a, hd.host_matrix_b, hd.host_matrix_c, hd.host_matrix_d, log_obj.size_matrices, real_t(
										log_obj.alpha), real_t(log_obj.beta), gemm_t);
		break;
	case DMRGEMMMIXED:
	case NONDMRWMMA:
		throw_line( "Not implemented!");
	}

	setup_execute(mt, log_obj, hd);
}

template<class half_t, class real_t, class mixed_real_t>
void call_mxm(Log& log_obj, GEMMTYPE gemm_t) {
	HostVectors<real_t, real_t, mixed_real_t> hd(
			log_obj.size_matrices * log_obj.size_matrices);

	hd.load_matrices_files(log_obj);

	std::shared_ptr<GEMMBase<real_t, real_t, mixed_real_t> > mt;
	switch (gemm_t) {
	case DMRWMA:
	case NONDMRWMMA:
	case NONDMRGEMM:
	case DMRGEMM:
		throw_line( "Not implemented!");
		break;
	case DMRGEMMMIXED:
		mt = std::make_shared<GEMMDMRMIXED<real_t, real_t, mixed_real_t>>(
				hd.host_matrix_a, hd.host_matrix_b, hd.host_matrix_c,
				hd.host_matrix_d, log_obj.size_matrices, real_t(log_obj.alpha),
				real_t(log_obj.beta), gemm_t);
		break;
	}

	setup_execute(mt, log_obj, hd);
}

void usage(char **argv) {
	std::cout << "./" << argv[0]
			<< " --generate 0/1 --gold <gold file, DEFAULT=./gold.matrix > --size <matrix size, DEFAULT=8192> "
					"--iterations <how many iterations, optional> --input_a <input A, DEFAUL=./input_a.matrix> "
					"--input_b <input B, DEFAUL=./input_b.matrix> --input_c <input C, DEFAUL=./input_c.matrix>  --precision <float/double, DEFAULT=float>"
			<< std::endl;
}

int main(int argc, char** argv) {
	Log log_obj(argc, argv, DEFAULT_INPUT_SIZE);

	std::cout << "Generate: " << log_obj.generate << std::endl;
	std::cout << "A input path: " << log_obj.a_input_path << std::endl;
	std::cout << "B input path: " << log_obj.b_input_path << std::endl;
	std::cout << "C input path: " << log_obj.c_input_path << std::endl;
	std::cout << "Gold in/out path: " << log_obj.gold_inout_path << std::endl;
	std::cout << "Iterations: " << log_obj.iterations << std::endl;
	std::cout << "Matrix size: " << log_obj.size_matrices << std::endl;
	std::cout << "Precision: " << log_obj.precision << std::endl;
	std::cout << "Verbose: " << log_obj.verbose << std::endl;
	std::cout << "DMR type: " << log_obj.dmr << std::endl;
	std::cout << "Tensor cores: " << log_obj.use_tensor_cores << std::endl;

	GEMMTYPE gemm_type;
	//NONDMRGEMM, DMRGEMM, NONDMRWMMA, DMRWMA
	//DMR TYPES
	if (log_obj.dmr == "dmr") {
		gemm_type = DMRGEMM;
		//PRECISION
		//HALF
		if (log_obj.precision == "half") {
			if (log_obj.use_tensor_cores) {
				gemm_type = DMRWMA;
			}

			call_mxm<half>(log_obj, gemm_type);
		}

		//FLOAT
		if (log_obj.precision == "float") {
			call_mxm<float>(log_obj, gemm_type);

		}

		//DOUBLE
		if (log_obj.precision == "double") {
			call_mxm<double>(log_obj, gemm_type);
		}
	} else if (log_obj.dmr == "dmrmixed") {
		gemm_type = DMRGEMMMIXED;
		if (log_obj.precision == "float") {
//			call_mxm<half, float>(log_obj, gemm_type);
			throw_line( "Not implemented function!");
		}

		if (log_obj.precision == "double") {
			call_mxm<double, double, float>(log_obj, gemm_type);
		}
	} else if (log_obj.dmr == "nondmr") {
		gemm_type = NONDMRGEMM;
		if (log_obj.precision == "half") {
			call_mxm<half>(log_obj, gemm_type);
		}

		if (log_obj.precision == "float") {
			call_mxm<float>(log_obj, gemm_type);
		}

		if (log_obj.precision == "double") {
			call_mxm<double>(log_obj, gemm_type);
		}
	}
	std::cout << "Finished computation\n";
	return 0;
}
