/*
 * HostVectors.h
 *
 *  Created on: 11/10/2019
 *      Author: fernando
 */

#ifndef HOSTVECTORS_H_
#define HOSTVECTORS_H_

#include <fstream>      // std::ifstream
#include <random>
#include <vector>


#define GENERATOR_MAXABSVALUE 5.0
#define GENERATOR_MINABSVALUE -5.0

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
		this->host_matrix_a = std::vector<half_t>(matrix_size);
		this->host_matrix_b = std::vector<half_t>(matrix_size);

		// C matrix
		this->host_matrix_c = std::vector<real_t>(matrix_size);

		// D Matrix
		this->host_matrix_d = std::vector<real_t>(matrix_size, 0);

		this->host_gold = std::vector<real_t>(matrix_size);
		this->host_matrix_smaller = std::vector<mixed_t>(matrix_size, 0);
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
					sizeof(real_t) * this->host_matrix_d.size());
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


#endif /* HOSTVECTORS_H_ */
