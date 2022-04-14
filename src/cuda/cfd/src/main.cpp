#include <iostream>
#include <fstream>
#include <cuda.h>

#include "device_vector.h"
#include "cuda_utils.h"

#include "common.h"
#include "Parameters.h"
#include "generic_log.h"

#define CHAR_CAST(x) (reinterpret_cast<char*>(x))
#define ERROR_THRESHOLD 1e-6

//Radiation experiment
extern std::string get_multi_compiler_header();

template<class T>
using DevArray = std::vector<rad::DeviceVector<T>>;

extern void euler3D(int *elements_surrounding_elements, float *normals, float *variables,
                    float *fluxes, float *step_factors, float *areas, float *old_variables, int nelr,
                    cudaStream_t &stream);

extern void compute_flux_contribution(float &density, float3 &momentum, float &density_energy,
                                      float &pressure, float3 &velocity, float3 &fc_momentum_x, float3 &fc_momentum_y,
                                      float3 &fc_momentum_z, float3 &fc_density_energy);

extern void copy_to_symbol_variables(float h_ff_variable[NVAR],
                                     float3 h_ff_flux_contribution_momentum_x, float3 h_ff_flux_contribution_momentum_y,
                                     float3 h_ff_flux_contribution_momentum_z,
                                     float3 h_ff_flux_contribution_density_energy);

extern void initialize_variables(int nelr, float *variables, cudaStream_t &stream);

template<typename real_t>
void write_gold(std::vector<real_t> &gold_array, std::string &gold_path, int nel, int nelr) {
//    std::vector<float> h_variables = variables.to_vector(); //nelr * NVAR);
//	download(h_variables, variables, nelr * NVAR);
    std::ofstream gold_file(gold_path, std::ios::binary);
    if (gold_file.good()) {
        gold_file.write(CHAR_CAST(&nel), sizeof(int));
        gold_file.write(CHAR_CAST(&nelr), sizeof(int));
        gold_file.write(CHAR_CAST(gold_array.data()), sizeof(real_t) * gold_array.size());
    } else {
        throw_line("Impossible to write the file " + gold_path)
    }
//    std::ofstream file("density");
//    file << nel << " " << nelr << std::endl;
//    for (int i = 0; i < nel; i++)
//        file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
//
//    std::ofstream file_momentum("momentum");
//    file_momentum << nel << " " << nelr << std::endl;
//    for (int i = 0; i < nel; i++) {
//        for (int j = 0; j != NDIM; j++)
//            file_momentum << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
//        file_momentum << std::endl;
//    }
//    std::ofstream file_energy("density_energy");
//    file_energy << nel << " " << nelr << std::endl;
//    for (int i = 0; i < nel; i++)
//        file_energy << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;

//	delete[] h_variables;
}

template<typename real_t>
void read_gold(std::vector<real_t> &gold_array, std::string &gold_path) {
    std::ifstream gold_file(gold_path, std::ios::binary);
    if (gold_file.good()) {
        int nel, nelr;
        gold_file.read(CHAR_CAST(&nel), sizeof(int));
        gold_file.read(CHAR_CAST(&nelr), sizeof(int));
        gold_file.read(CHAR_CAST(gold_array.data()), sizeof(real_t) * gold_array.size());
    } else {
        throw_line("Impossible to read the file " + gold_path)
    }
}

template<typename real_t>
size_t compare_gold(const std::vector<real_t> &gold_array, const std::vector<real_t> &new_array, rad::Log &logger,
                    const int nel, const int nelr, const int stream) {
    auto cast_to_uint = [](const float *number) {
        uint32_t n_data;
        std::memcpy(&n_data, number, sizeof(float));
        return n_data;
    };
    size_t error_count = 0;
#ifndef FULL_COMPARISSON
    for (size_t i = 0; i < gold_array.size(); i++) {
        auto &g = gold_array[i];
        auto &n = new_array[i];
        auto diff = fabs(g - n);
        if (diff > ERROR_THRESHOLD) {
            std::string error_detail = "stream:" + std::to_string(stream) + " i:" + std::to_string(i);
            // It is better to write the raw data
            error_detail += " e:" + std::to_string(cast_to_uint(&g)) + " r:" + std::to_string(cast_to_uint(&n));
#pragma omp critical
            {
                logger.log_error_detail(error_detail);
            }
            error_count++;
            if (error_count < 10) {
                std::cout << error_detail << std::endl;
            }
        }
    }
#else
    //    std::ofstream file("density");
   for (int i = 0; i < nel; i++) {
//        file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
        auto index = i + VAR_DENSITY * nelr;
        auto g = gold_array[index];
        auto n = new_array[index];
        auto diff = fabs(g - n);
        if (diff > ERROR_THRESHOLD) {

            std::string error_detail = "stream:" + std::to_string(stream) + " density_i:" + std::to_string(index);
            // It is better to write the raw data
            error_detail += " e:" + std::to_string(cast_to_uint(&g));
            error_detail += " r:" + std::to_string(cast_to_uint(&n));
#pragma omp critical
            {
                logger.log_error_detail(error_detail);
            }
            error_count++;
            if (error_count < 10) {
                std::cout << error_detail << std::endl;
            }
        }
    }

//    std::ofstream file_momentum("momentum");
    for (int i = 0; i < nel; i++) {
        for (int j = 0; j != NDIM; j++) {
//      file_momentum << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
            auto index = i + (VAR_MOMENTUM + j) * nelr;
            auto g = gold_array[index], n = new_array[index];
            auto diff = fabs(g - n);
            if (diff > ERROR_THRESHOLD) {
                std::string error_detail = "stream:" + std::to_string(stream) + " momentum_ij:" + std::to_string(index);
                error_detail += "-" + std::to_string(i) + "-" + std::to_string(j);
                error_detail += " e:" + std::to_string(cast_to_uint(&g));
                error_detail += " r:" + std::to_string(cast_to_uint(&n));
#pragma omp critical
                {
                    logger.log_error_detail(error_detail);
                }
                error_count++;
                if (error_count < 10) {
                    std::cout << error_detail << std::endl;
                }
            }
        }

    }
//    std::ofstream file_energy("density_energy");
    for (int i = 0; i < nel; i++) {
//                file_energy << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
        auto index = i + VAR_DENSITY_ENERGY * nelr;
        auto g = gold_array[index], n = new_array[index];
        auto diff = fabs(g - n);
        if (diff > ERROR_THRESHOLD) {
            std::string error_detail =
                    "stream:" + std::to_string(stream) + " density_energy_i:" + std::to_string(index);
            // It is better to write the raw data
            error_detail += " e:" + std::to_string(cast_to_uint(&g));
            error_detail += " r:" + std::to_string(cast_to_uint(&n));
#pragma omp critical
            {
                logger.log_error_detail(error_detail);
            }
            error_count++;
            if (error_count < 10) {
                std::cout << error_detail << std::endl;
            }
        }
    }
#endif
    return error_count;
}


int main(int argc, char **argv) {

//	if (argc < 2) {
//		std::cout << "specify data file name" << std::endl;
//		return 0;
//	}
    Parameters parameters(argc, argv);

    // Start log helper
    auto test_name = "cudaCFD";
    auto test_info = "input:" + parameters.input + " gold:" + parameters.gold;
    test_info += " streams:" + std::to_string(parameters.stream_number);
    test_info += " iterations:" + std::to_string(parameters.iterations);
    test_info += get_multi_compiler_header();
    // print after each 10 iterations
    rad::Log logger(test_name, test_info, 10);
    std::string &data_file_name = parameters.input;

    cudaDeviceProp prop{};
    int dev;

    rad::checkFrameworkErrors(cudaSetDevice(DEVICE))
    rad::checkFrameworkErrors(cudaGetDevice(&dev))
    rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, dev))

    if (parameters.verbose) {
        std::cout << "WG size of kernel:initialize = " << BLOCK_SIZE_1 << ", WG size of kernel:compute_step_factor = "
                  << BLOCK_SIZE_2 << ", WG size of kernel:compute_flux = "
                  << BLOCK_SIZE_3 << ", WG size of kernel:time_step = " << BLOCK_SIZE_4 << "\n";
        std::cout << "Name:" << prop.name << std::endl;
        std::cout << parameters << std::endl;
        std::cout << logger << std::endl;
    }

    // set far field conditions and load them into constant memory on the gpu

    float h_ff_variable[NVAR];
    const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

    h_ff_variable[VAR_DENSITY] = float(1.4);

    float ff_pressure(1.0f);
    float ff_speed_of_sound = sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
    float ff_speed = float(ff_mach) * ff_speed_of_sound;

    float3 ff_velocity;
    ff_velocity.x = ff_speed * float(cos((float) angle_of_attack));
    ff_velocity.y = ff_speed * float(sin((float) angle_of_attack));
    ff_velocity.z = 0.0f;

    h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
    h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
    h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

    h_ff_variable[VAR_DENSITY_ENERGY] =
            h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) + (ff_pressure / float(GAMMA - 1.0f));

    float3 h_ff_momentum;
    h_ff_momentum.x = *(h_ff_variable + VAR_MOMENTUM + 0);
    h_ff_momentum.y = *(h_ff_variable + VAR_MOMENTUM + 1);
    h_ff_momentum.z = *(h_ff_variable + VAR_MOMENTUM + 2);
    float3 h_ff_flux_contribution_momentum_x;
    float3 h_ff_flux_contribution_momentum_y;
    float3 h_ff_flux_contribution_momentum_z;
    float3 h_ff_flux_contribution_density_energy;
    compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure,
                              ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y,
                              h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

    // copy far field conditions to the gpu
    copy_to_symbol_variables(h_ff_variable, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y,
                             h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);
    int nel;
    int nelr;

    std::ifstream file(data_file_name);

    file >> nel;
    nelr = BLOCK_SIZE_0 * ((nel / BLOCK_SIZE_0) + std::min(1, nel % BLOCK_SIZE_0));

    //RAD TEST
    std::vector<float> gold_array(nelr * NVAR);
    if (!parameters.generate) {
        read_gold(gold_array, parameters.gold);
    }
    //------------

    std::vector<float> h_areas(nelr);
    std::vector<int> h_elements_surrounding_elements(nelr * NNB);
    std::vector<float> h_normals(nelr * NDIM * NNB);

    // read in data
    for (int i = 0; i < nel; i++) {
        file >> h_areas[i];
        for (int j = 0; j < NNB; j++) {
            file >> h_elements_surrounding_elements[i + j * nelr];
            if (h_elements_surrounding_elements[i + j * nelr] < 0)
                h_elements_surrounding_elements[i + j * nelr] = -1;
            h_elements_surrounding_elements[i + j * nelr]--; //it's coming in with Fortran numbering

            for (int k = 0; k < NDIM; k++) {
                file >> h_normals[i + (j + k * NNB) * nelr];
                h_normals[i + (j + k * NNB) * nelr] = -h_normals[i + (j + k * NNB) * nelr];
            }
        }
    }

    // fill in remaining data
    int last = nel - 1;
    for (int i = nel; i < nelr; i++) {
        h_areas[i] = h_areas[last];
        for (int j = 0; j < NNB; j++) {
            // duplicate the last element
            h_elements_surrounding_elements[i + j * nelr] = h_elements_surrounding_elements[last + j * nelr];
            for (int k = 0; k < NDIM; k++)
                h_normals[last + (j + k * NNB) * nelr] = h_normals[last + (j + k * NNB) * nelr];
        }
    }

    // read in domain geometry
    DevArray<float> device_stream_areas(parameters.stream_number);
    DevArray<int> device_stream_elements_surrounding_elements(parameters.stream_number);
    DevArray<float> device_stream_normals(parameters.stream_number);

    // Create arrays and set initial conditions
    DevArray<float> device_stream_variables(parameters.stream_number);
    DevArray<float> device_stream_old_variables(parameters.stream_number);
    DevArray<float> device_stream_fluxes(parameters.stream_number);
    DevArray<float> device_stream_step_factors(parameters.stream_number);

    std::vector<cudaStream_t> streams(parameters.stream_number);

    for (auto i = 0; i < parameters.stream_number; i++) {
        rad::checkFrameworkErrors(cudaStreamCreate(&streams[i]))

        //		areas = alloc<float>(nelr);
        //		upload<float>(areas, h_areas, nelr);
        device_stream_areas[i] = h_areas;

        //		elements_surrounding_elements = alloc<int>(nelr * NNB);
        //		upload<int>(elements_surrounding_elements,
        //				h_elements_surrounding_elements, nelr * NNB);
        device_stream_elements_surrounding_elements[i] = h_elements_surrounding_elements;

        //		normals = alloc<float>(nelr * NDIM * NNB);
        //		upload<float>(normals, h_normals, nelr * NDIM * NNB);
        device_stream_normals[i] = h_normals;

        //		delete[] h_areas;
        //		delete[] h_elements_surrounding_elements;
        //		delete[] h_normals;

        auto &variables = device_stream_variables[i];
        auto &old_variables = device_stream_old_variables[i];
        auto &fluxes = device_stream_fluxes[i];
        auto &step_factors = device_stream_step_factors[i];
        variables.resize(nelr * NVAR);
        old_variables.resize(nelr * NVAR);
        fluxes.resize(nelr * NVAR);
        step_factors.resize(nelr);

        // make sure all memory is floatly allocated before we start timing
        initialize_variables(nelr, variables.data(), streams[i]);
        initialize_variables(nelr, old_variables.data(), streams[i]);
        initialize_variables(nelr, fluxes.data(), streams[i]);
        //	cudaMemset((void*) step_factors, 0, sizeof(float) * nelr);
        step_factors.clear();
    }

    // make sure CUDA isn't still doing something before we start timing
    rad::checkFrameworkErrors(cudaDeviceSynchronize())
    // Setup reload variables
    auto device_reload_variables = device_stream_variables;
    auto device_reload_old_variables = device_stream_old_variables;
    auto device_reload_fluxes = device_stream_fluxes;


    // these need to be computed the first time in order to compute time step
    std::cout << "Starting..." << std::endl;

//	StopWatchInterface *timer = 0;
    //	unsigned int timer = 0;

    // CUT_SAFE_CALL( cutCreateTimer( &timer));
    // CUT_SAFE_CALL( cutStartTimer( timer));
//	sdkCreateTimer(&timer);
//	sdkStartTimer(&timer);
    std::vector<std::vector<float>> host_variables(parameters.stream_number);
    // Begin iterations
    for (int i = 0; i < parameters.iterations; i++) {
        auto kernel_time = rad::mysecond();
        logger.start_iteration();

        for (auto stream = 0; stream < parameters.stream_number; stream++) {
            //		copy<float>(old_variables, variables, nelr * NVAR);
//			old_variables = variables;
            device_stream_old_variables[stream] = device_stream_variables[stream];

            auto variables = device_stream_variables[stream].data();
            auto old_variables = device_stream_old_variables[stream].data();
            auto fluxes = device_stream_fluxes[stream].data();
            auto step_factors = device_stream_step_factors[stream].data();
            auto elements_surrounding_elements = device_stream_elements_surrounding_elements[stream].data();
            auto normals = device_stream_normals[stream].data();
            auto areas = device_stream_areas[stream].data();

            euler3D(elements_surrounding_elements, normals, variables, fluxes, step_factors, areas, old_variables, nelr,
                    streams[stream]);
        }
        //if the device is sync no need to sync in the stream again
        rad::checkFrameworkErrors (cudaDeviceSynchronize())
        rad::checkFrameworkErrors(cudaGetLastError())
        logger.end_iteration();
        kernel_time = rad::mysecond() - kernel_time;

        auto copy_time = rad::mysecond();
        for (auto stream = 0; stream < parameters.stream_number; stream++) {
//            rad::checkFrameworkErrors (cudaStreamSynchronize(streams[stream]));;
            device_stream_variables[stream].to_vector(host_variables[stream]);
        }
        copy_time = rad::mysecond() - copy_time;

        // RAD setup
        size_t errors = 0;
        auto cmp_time = rad::mysecond();
        if (!parameters.generate) {
            std::vector<size_t> error_vector(parameters.stream_number);
#pragma omp parallel for shared(logger)
            for (int stream = 0; stream < parameters.stream_number; stream++) {
                error_vector[stream] = compare_gold(gold_array, host_variables[stream], logger, nel, nelr, stream);
            }
            logger.update_errors();

            for (auto err_i: error_vector) {
                errors += err_i;
            }
            // recopying to set the arrays to default
            device_stream_variables = device_reload_variables;
            device_stream_old_variables = device_reload_old_variables;
            device_stream_fluxes = device_reload_fluxes;
        }
        cmp_time = rad::mysecond() - cmp_time;

        if (parameters.verbose) {
            auto wasted_time = copy_time + cmp_time;
            auto full_time = wasted_time + kernel_time;
            std::cout << "Iteration:" << i << " Errors:" << errors << " Kernel time:" << kernel_time;
            std::cout << " Copy time:" << copy_time << "Compare time:" << cmp_time << " Wasted time: "
                      << int((wasted_time / full_time) * 100.0f) << "%" << std::endl;
            std::cout << "==========================================================================================\n";
        }
    }

    //	CUT_SAFE_CALL( cutStopTimer(timer) );
//	sdkStopTimer(&timer);
    if (parameters.generate) {
        if (parameters.verbose)
            std::cout << "Saving solution. Only stream 0 will be saved to gold" << std::endl;
        auto host_stream_variables_cmp = device_stream_variables[0].to_vector(); //nelr * NVAR);
        write_gold(host_stream_variables_cmp, parameters.gold, nel, nelr);
        if (parameters.verbose)
            std::cout << "Saved solution..." << std::endl;
    }

    std::cout << "Done..." << std::endl;
    for (auto &stream: streams) {
        rad::checkFrameworkErrors(cudaStreamDestroy(stream))
    }
    return 0;
}
