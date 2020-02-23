#include <iostream>
#include <fstream>
#include <cuda.h>

#include "device_vector.h"
#include "cuda_utils.h"

#include "common.h"
#include "Parameters.h"

extern void euler3D(int nelr,
		rad::DeviceVector<int>& elements_surrounding_elements,
		rad::DeviceVector<float>& normals, rad::DeviceVector<float>& variables,
		rad::DeviceVector<float>& fluxes,
		rad::DeviceVector<float>& step_factors, rad::DeviceVector<float>& areas,
		rad::DeviceVector<float>& old_variables);

extern void compute_flux_contribution(float& density, float3& momentum,
		float& density_energy, float& pressure, float3& velocity,
		float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z,
		float3& fc_density_energy);

extern void copy_to_symbol_variables(float h_ff_variable[NVAR],
		float3 h_ff_flux_contribution_momentum_x,
		float3 h_ff_flux_contribution_momentum_y,
		float3 h_ff_flux_contribution_momentum_z,
		float3 h_ff_flux_contribution_density_energy);

extern void initialize_variables(int nelr, float* variables);

void dump(rad::DeviceVector<float>& variables, int nel, int nelr) {
	std::vector<float> h_variables(variables.to_vector()); //nelr * NVAR);
//	download(h_variables, variables, nelr * NVAR);

	std::ofstream file("density");
	file << nel << " " << nelr << std::endl;
	for (int i = 0; i < nel; i++)
		file << h_variables[i + VAR_DENSITY * nelr] << std::endl;

	std::ofstream file_momentum("momentum");
	file_momentum << nel << " " << nelr << std::endl;
	for (int i = 0; i < nel; i++) {
		for (int j = 0; j != NDIM; j++)
			file_momentum << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
		file_momentum << std::endl;
	}

	std::ofstream file_energy("density_energy");
	file_energy << nel << " " << nelr << std::endl;
	for (int i = 0; i < nel; i++)
		file_energy << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;

//	delete[] h_variables;
}

int main(int argc, char** argv) {

//	if (argc < 2) {
//		std::cout << "specify data file name" << std::endl;
//		return 0;
//	}
	Parameters parameters(argc, argv);

	std::string& data_file_name = parameters.input;

	cudaDeviceProp prop;
	int dev;

	rad::checkFrameworkErrors(cudaSetDevice(DEVICE));
	rad::checkFrameworkErrors(cudaGetDevice(&dev));
	rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, dev));

	if (parameters.verbose) {
		std::cout << "WG size of kernel:initialize = " << BLOCK_SIZE_1
				<< ", WG size of kernel:compute_step_factor = " << BLOCK_SIZE_2
				<< ", "
						"WG size of kernel:compute_flux = " << BLOCK_SIZE_3
				<< ", WG size of kernel:time_step = " << BLOCK_SIZE_4 << "\n";
		std::cout << "Name:" << prop.name << std::endl;
		std::cout << parameters << std::endl;
	}

	// set far field conditions and load them into constant memory on the gpu

	float h_ff_variable[NVAR];
	const float angle_of_attack = float(3.1415926535897931 / 180.0f)
			* float(deg_angle_of_attack);

	h_ff_variable[VAR_DENSITY] = float(1.4);

	float ff_pressure = float(1.0f);
	float ff_speed_of_sound = sqrt(
	GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
	float ff_speed = float(ff_mach) * ff_speed_of_sound;

	float3 ff_velocity;
	ff_velocity.x = ff_speed * float(cos((float) angle_of_attack));
	ff_velocity.y = ff_speed * float(sin((float) angle_of_attack));
	ff_velocity.z = 0.0f;

	h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY]
			* ff_velocity.x;
	h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY]
			* ff_velocity.y;
	h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY]
			* ff_velocity.z;

	h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]
			* (float(0.5f) * (ff_speed * ff_speed))
			+ (ff_pressure / float(GAMMA - 1.0f));

	float3 h_ff_momentum;
	h_ff_momentum.x = *(h_ff_variable + VAR_MOMENTUM + 0);
	h_ff_momentum.y = *(h_ff_variable + VAR_MOMENTUM + 1);
	h_ff_momentum.z = *(h_ff_variable + VAR_MOMENTUM + 2);
	float3 h_ff_flux_contribution_momentum_x;
	float3 h_ff_flux_contribution_momentum_y;
	float3 h_ff_flux_contribution_momentum_z;
	float3 h_ff_flux_contribution_density_energy;
	compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum,
			h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity,
			h_ff_flux_contribution_momentum_x,
			h_ff_flux_contribution_momentum_y,
			h_ff_flux_contribution_momentum_z,
			h_ff_flux_contribution_density_energy);

	// copy far field conditions to the gpu
	copy_to_symbol_variables(h_ff_variable, h_ff_flux_contribution_momentum_x,
			h_ff_flux_contribution_momentum_y,
			h_ff_flux_contribution_momentum_z,
			h_ff_flux_contribution_density_energy);
	int nel;
	int nelr;

	// read in domain geometry
	rad::DeviceVector<float> areas;
	rad::DeviceVector<int> elements_surrounding_elements;
	rad::DeviceVector<float> normals;

	std::ifstream file(data_file_name);

	file >> nel;
	nelr = BLOCK_SIZE_0
			* ((nel / BLOCK_SIZE_0) + std::min(1, nel % BLOCK_SIZE_0));

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
				h_normals[i + (j + k * NNB) * nelr] = -h_normals[i
						+ (j + k * NNB) * nelr];
			}
		}
	}

	// fill in remaining data
	int last = nel - 1;
	for (int i = nel; i < nelr; i++) {
		h_areas[i] = h_areas[last];
		for (int j = 0; j < NNB; j++) {
			// duplicate the last element
			h_elements_surrounding_elements[i + j * nelr] =
					h_elements_surrounding_elements[last + j * nelr];
			for (int k = 0; k < NDIM; k++)
				h_normals[last + (j + k * NNB) * nelr] = h_normals[last
						+ (j + k * NNB) * nelr];
		}
	}

//		areas = alloc<float>(nelr);
//		upload<float>(areas, h_areas, nelr);
	areas = h_areas;

//		elements_surrounding_elements = alloc<int>(nelr * NNB);
//		upload<int>(elements_surrounding_elements,
//				h_elements_surrounding_elements, nelr * NNB);
	elements_surrounding_elements = h_elements_surrounding_elements;

//		normals = alloc<float>(nelr * NDIM * NNB);
//		upload<float>(normals, h_normals, nelr * NDIM * NNB);
	normals = h_normals;

//		delete[] h_areas;
//		delete[] h_elements_surrounding_elements;
//		delete[] h_normals;

	// Create arrays and set initial conditions
	rad::DeviceVector<float> variables(nelr * NVAR);
	initialize_variables(nelr, variables.data());

	rad::DeviceVector<float> old_variables(nelr * NVAR);
	rad::DeviceVector<float> fluxes(nelr * NVAR);
	rad::DeviceVector<float> step_factors(nelr);

	// make sure all memory is floatly allocated before we start timing
	initialize_variables(nelr, old_variables.data());
	initialize_variables(nelr, fluxes.data());
//	cudaMemset((void*) step_factors, 0, sizeof(float) * nelr);
	step_factors.clear();

	// make sure CUDA isn't still doing something before we start timing
	cudaDeviceSynchronize();

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

//	StopWatchInterface *timer = 0;
	//	unsigned int timer = 0;

	// CUT_SAFE_CALL( cutCreateTimer( &timer));
	// CUT_SAFE_CALL( cutStartTimer( timer));
//	sdkCreateTimer(&timer);
//	sdkStartTimer(&timer);
	auto begin = rad::mysecond();
	// Begin iterations
	for (int i = 0; i < parameters.iterations; i++) {
//		copy<float>(old_variables, variables, nelr * NVAR);
		old_variables = variables;

		euler3D(nelr, elements_surrounding_elements, normals, variables, fluxes,
				step_factors, areas, old_variables);

	}

	rad::checkFrameworkErrors(cudaDeviceSynchronize());
	;
	auto end = rad::mysecond();
	//	CUT_SAFE_CALL( cutStopTimer(timer) );
//	sdkStopTimer(&timer);

	std::cout << ((end - begin) / float(parameters.iterations))
			<< " seconds per iteration" << std::endl;

	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	std::cout << "Done..." << std::endl;

	return 0;
}
