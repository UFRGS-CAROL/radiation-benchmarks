// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

//#include <cutil.h>
//#include <helper_cuda.h>
//#include <helper_timer.h>

#include "cuda_utils.h"
#include "common.h"
#include "multi_compiler_analysis.h"

//Radiation experiments setup
std::string get_multi_compiler_header(){
    return rad::get_multi_compiler_header();
}

/*
 * Element-based Cell-centered FVM solver functions
 */
__constant__ float ff_variable[NVAR];
__constant__ float3 ff_flux_contribution_momentum_x[1];
__constant__ float3 ff_flux_contribution_momentum_y[1];
__constant__ float3 ff_flux_contribution_momentum_z[1];
__constant__ float3 ff_flux_contribution_density_energy[1];

__global__ void cuda_initialize_variables(int nelr, float *variables) {
    const unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int j = 0; j < NVAR; j++)
        variables[i + j * nelr] = ff_variable[j];
}

void initialize_variables(int nelr, float *variables, cudaStream_t &stream) {
    dim3 Dg(nelr / BLOCK_SIZE_1), Db(BLOCK_SIZE_1);
    cuda_initialize_variables<<<Dg, Db, 0, stream>>>(nelr, variables);
    //getLastCudaError("initialize_variables failed");
    rad::checkFrameworkErrors(cudaGetLastError())
//	;
}

__device__ __host__ void
compute_flux_contribution(float &density, float3 &momentum, float &density_energy, float &pressure, float3 &velocity,
                          float3 &fc_momentum_x, float3 &fc_momentum_y, float3 &fc_momentum_z,
                          float3 &fc_density_energy) {
    fc_momentum_x.x = velocity.x * momentum.x + pressure;
    fc_momentum_x.y = velocity.x * momentum.y;
    fc_momentum_x.z = velocity.x * momentum.z;

    fc_momentum_y.x = fc_momentum_x.y;
    fc_momentum_y.y = velocity.y * momentum.y + pressure;
    fc_momentum_y.z = velocity.y * momentum.z;

    fc_momentum_z.x = fc_momentum_x.z;
    fc_momentum_z.y = fc_momentum_y.z;
    fc_momentum_z.z = velocity.z * momentum.z + pressure;

    float de_p = density_energy + pressure;
    fc_density_energy.x = velocity.x * de_p;
    fc_density_energy.y = velocity.y * de_p;
    fc_density_energy.z = velocity.z * de_p;
}

__device__ inline void compute_velocity(float &density, float3 &momentum,
                                        float3 &velocity) {
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

__device__ inline float compute_speed_sqd(float3 &velocity) {
    return velocity.x * velocity.x + velocity.y * velocity.y
           + velocity.z * velocity.z;
}

__device__ inline float compute_pressure(float &density, float &density_energy,
                                         float &speed_sqd) {
    return (float(GAMMA) - float(1.0f))
           * (density_energy - float(0.5f) * density * speed_sqd);
}

__device__ inline float compute_speed_of_sound(float &density,
                                               float &pressure) {
    return sqrtf(float(GAMMA) * pressure / density);
}

__global__ void cuda_compute_step_factor(int nelr, const float *variables, float *areas, float *step_factors) {
    const unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

    float density = variables[i + VAR_DENSITY * nelr];
    float3 momentum;
    momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];

    float3 velocity;
    compute_velocity(density, momentum, velocity);
    float speed_sqd = compute_speed_sqd(velocity);
    float pressure = compute_pressure(density, density_energy, speed_sqd);
    float speed_of_sound = compute_speed_of_sound(density, pressure);

    // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c)....
    // but when we do time stepping, this later would need to
    // be divided by the area, so we just do it all at once
    step_factors[i] = float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}

void compute_step_factor(int nelr, float *variables, float *areas,
                         float *step_factors, cudaStream_t &stream) {
    dim3 Dg(nelr / BLOCK_SIZE_2), Db(BLOCK_SIZE_2);
    cuda_compute_step_factor<<<Dg, Db, 0, stream>>>(nelr, variables, areas,
                                                    step_factors);
//	getLastCudaError("compute_step_factor failed");
    rad::checkFrameworkErrors(cudaGetLastError())
//	;
}

/*
 */
__global__ void
cuda_compute_flux(int nelr, const int *elements_surrounding_elements, const float *normals, const float *variables,
                  float *fluxes) {
    const float smoothing_coefficient(0.2f);
    const unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

    int j, nb;
    float3 normal;
    float normal_len;
    float factor;

    float density_i = variables[i + VAR_DENSITY * nelr];
    float3 momentum_i;
    momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

    float3 velocity_i;
    compute_velocity(density_i, momentum_i, velocity_i);
    float speed_sqd_i = compute_speed_sqd(velocity_i);
    float speed_i = sqrtf(speed_sqd_i);
    float pressure_i = compute_pressure(density_i, density_energy_i,
                                        speed_sqd_i);
    float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
    float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
            flux_contribution_i_momentum_z;
    float3 flux_contribution_i_density_energy;
    compute_flux_contribution(density_i, momentum_i, density_energy_i,
                              pressure_i, velocity_i, flux_contribution_i_momentum_x,
                              flux_contribution_i_momentum_y, flux_contribution_i_momentum_z,
                              flux_contribution_i_density_energy);

    float flux_i_density(0.0f);
    float3 flux_i_momentum;
    flux_i_momentum.x = float(0.0f);
    flux_i_momentum.y = float(0.0f);
    flux_i_momentum.z = float(0.0f);
    float flux_i_density_energy(0.0f);

    float3 velocity_nb;
    float density_nb, density_energy_nb;
    float3 momentum_nb;
    float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y,
            flux_contribution_nb_momentum_z;
    float3 flux_contribution_nb_density_energy;
    float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

#pragma unroll
    for (j = 0; j < NNB; j++) {
        nb = elements_surrounding_elements[i + j * nelr];
        normal.x = normals[i + (j + 0 * NNB) * nelr];
        normal.y = normals[i + (j + 1 * NNB) * nelr];
        normal.z = normals[i + (j + 2 * NNB) * nelr];
        normal_len = sqrtf(
                normal.x * normal.x + normal.y * normal.y
                + normal.z * normal.z);

        if (nb >= 0)    // a legitimate neighbor
        {
            density_nb = variables[nb + VAR_DENSITY * nelr];
            momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
            momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
            momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
            density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
            compute_velocity(density_nb, momentum_nb, velocity_nb);
            speed_sqd_nb = compute_speed_sqd(velocity_nb);
            pressure_nb = compute_pressure(density_nb, density_energy_nb,
                                           speed_sqd_nb);
            speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
            compute_flux_contribution(density_nb, momentum_nb,
                                      density_energy_nb, pressure_nb, velocity_nb,
                                      flux_contribution_nb_momentum_x,
                                      flux_contribution_nb_momentum_y,
                                      flux_contribution_nb_momentum_z,
                                      flux_contribution_nb_density_energy);

            // artificial viscosity
            factor = -normal_len * smoothing_coefficient * float(0.5f)
                     * (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i
                        + speed_of_sound_nb);
            flux_i_density += factor * (density_i - density_nb);
            flux_i_density_energy += factor
                                     * (density_energy_i - density_energy_nb);
            flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
            flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
            flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

            // accumulate cell-centered fluxes
            factor = float(0.5f) * normal.x;
            flux_i_density += factor * (momentum_nb.x + momentum_i.x);
            flux_i_density_energy += factor
                                     * (flux_contribution_nb_density_energy.x
                                        + flux_contribution_i_density_energy.x);
            flux_i_momentum.x += factor
                                 * (flux_contribution_nb_momentum_x.x
                                    + flux_contribution_i_momentum_x.x);
            flux_i_momentum.y += factor
                                 * (flux_contribution_nb_momentum_y.x
                                    + flux_contribution_i_momentum_y.x);
            flux_i_momentum.z += factor
                                 * (flux_contribution_nb_momentum_z.x
                                    + flux_contribution_i_momentum_z.x);

            factor = float(0.5f) * normal.y;
            flux_i_density += factor * (momentum_nb.y + momentum_i.y);
            flux_i_density_energy += factor
                                     * (flux_contribution_nb_density_energy.y
                                        + flux_contribution_i_density_energy.y);
            flux_i_momentum.x += factor
                                 * (flux_contribution_nb_momentum_x.y
                                    + flux_contribution_i_momentum_x.y);
            flux_i_momentum.y += factor
                                 * (flux_contribution_nb_momentum_y.y
                                    + flux_contribution_i_momentum_y.y);
            flux_i_momentum.z += factor
                                 * (flux_contribution_nb_momentum_z.y
                                    + flux_contribution_i_momentum_z.y);

            factor = float(0.5f) * normal.z;
            flux_i_density += factor * (momentum_nb.z + momentum_i.z);
            flux_i_density_energy += factor
                                     * (flux_contribution_nb_density_energy.z
                                        + flux_contribution_i_density_energy.z);
            flux_i_momentum.x += factor
                                 * (flux_contribution_nb_momentum_x.z
                                    + flux_contribution_i_momentum_x.z);
            flux_i_momentum.y += factor
                                 * (flux_contribution_nb_momentum_y.z
                                    + flux_contribution_i_momentum_y.z);
            flux_i_momentum.z += factor
                                 * (flux_contribution_nb_momentum_z.z
                                    + flux_contribution_i_momentum_z.z);
        } else if (nb == -1)    // a wing boundary
        {
            flux_i_momentum.x += normal.x * pressure_i;
            flux_i_momentum.y += normal.y * pressure_i;
            flux_i_momentum.z += normal.z * pressure_i;
        } else if (nb == -2) // a far field boundary
        {
            factor = float(0.5f) * normal.x;
            flux_i_density += factor
                              * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
            flux_i_density_energy += factor
                                     * (ff_flux_contribution_density_energy[0].x
                                        + flux_contribution_i_density_energy.x);
            flux_i_momentum.x += factor
                                 * (ff_flux_contribution_momentum_x[0].x
                                    + flux_contribution_i_momentum_x.x);
            flux_i_momentum.y += factor
                                 * (ff_flux_contribution_momentum_y[0].x
                                    + flux_contribution_i_momentum_y.x);
            flux_i_momentum.z += factor
                                 * (ff_flux_contribution_momentum_z[0].x
                                    + flux_contribution_i_momentum_z.x);

            factor = float(0.5f) * normal.y;
            flux_i_density += factor
                              * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
            flux_i_density_energy += factor
                                     * (ff_flux_contribution_density_energy[0].y
                                        + flux_contribution_i_density_energy.y);
            flux_i_momentum.x += factor
                                 * (ff_flux_contribution_momentum_x[0].y
                                    + flux_contribution_i_momentum_x.y);
            flux_i_momentum.y += factor
                                 * (ff_flux_contribution_momentum_y[0].y
                                    + flux_contribution_i_momentum_y.y);
            flux_i_momentum.z += factor
                                 * (ff_flux_contribution_momentum_z[0].y
                                    + flux_contribution_i_momentum_z.y);

            factor = float(0.5f) * normal.z;
            flux_i_density += factor
                              * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
            flux_i_density_energy += factor
                                     * (ff_flux_contribution_density_energy[0].z
                                        + flux_contribution_i_density_energy.z);
            flux_i_momentum.x += factor
                                 * (ff_flux_contribution_momentum_x[0].z
                                    + flux_contribution_i_momentum_x.z);
            flux_i_momentum.y += factor
                                 * (ff_flux_contribution_momentum_y[0].z
                                    + flux_contribution_i_momentum_y.z);
            flux_i_momentum.z += factor
                                 * (ff_flux_contribution_momentum_z[0].z
                                    + flux_contribution_i_momentum_z.z);

        }
    }

    fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
    fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
    fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
    fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
    fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
}

void compute_flux(int nelr, int *elements_surrounding_elements, float *normals, float *variables, float *fluxes,
                  cudaStream_t &stream) {
    dim3 Dg(nelr / BLOCK_SIZE_3), Db(BLOCK_SIZE_3);
    cuda_compute_flux<<<Dg, Db, 0, stream>>>(nelr, elements_surrounding_elements, normals, variables, fluxes);
//	getLastCudaError("compute_flux failed");
    rad::checkFrameworkErrors(cudaGetLastError())
}

__global__ void cuda_time_step(int j, int nelr, const float *old_variables, float *variables, const float *step_factors,
                               const float *fluxes) {
    const unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

    float factor = step_factors[i] / float(RK + 1 - j);

    variables[i + VAR_DENSITY * nelr] = old_variables[i + VAR_DENSITY * nelr]
                                        + factor * fluxes[i + VAR_DENSITY * nelr];
    variables[i + VAR_DENSITY_ENERGY * nelr] = old_variables[i
                                                             + VAR_DENSITY_ENERGY * nelr]
                                               + factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
    variables[i + (VAR_MOMENTUM + 0) * nelr] = old_variables[i
                                                             + (VAR_MOMENTUM + 0) * nelr]
                                               + factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
    variables[i + (VAR_MOMENTUM + 1) * nelr] = old_variables[i
                                                             + (VAR_MOMENTUM + 1) * nelr]
                                               + factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
    variables[i + (VAR_MOMENTUM + 2) * nelr] = old_variables[i
                                                             + (VAR_MOMENTUM + 2) * nelr]
                                               + factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
}

void time_step(int j, int nelr, float *old_variables, float *variables, float *step_factors, float *fluxes,
               cudaStream_t &stream) {
    dim3 Dg(nelr / BLOCK_SIZE_4), Db(BLOCK_SIZE_4);
    cuda_time_step<<<Dg, Db, 0, stream>>>(j, nelr, old_variables, variables,
                                          step_factors, fluxes);
//	getLastCudaError("update failed");
    rad::checkFrameworkErrors(cudaGetLastError())
}

void copy_to_symbol_variables(float h_ff_variable[NVAR],
                              float3 h_ff_flux_contribution_momentum_x,
                              float3 h_ff_flux_contribution_momentum_y,
                              float3 h_ff_flux_contribution_momentum_z,
                              float3 h_ff_flux_contribution_density_energy) {
    // copy far field conditions to the gpu
    rad::checkFrameworkErrors(cudaMemcpyToSymbol(ff_variable, h_ff_variable, NVAR * sizeof(float)))
    rad::checkFrameworkErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_x,
                                                 &h_ff_flux_contribution_momentum_x, sizeof(float3)))
    rad::checkFrameworkErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_y,
                                                 &h_ff_flux_contribution_momentum_y, sizeof(float3)))
    rad::checkFrameworkErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_z,
                                                 &h_ff_flux_contribution_momentum_z, sizeof(float3)))
    rad::checkFrameworkErrors(cudaMemcpyToSymbol(ff_flux_contribution_density_energy,
                                                 &h_ff_flux_contribution_density_energy, sizeof(float3)))
}

/*
 * Main function
 */
void euler3D(int *elements_surrounding_elements, float *normals, float *variables, float *fluxes, float *step_factors,
             float *areas, float *old_variables, int nelr, cudaStream_t &stream) {

    // for the first iteration we compute the time step
    compute_step_factor(nelr, variables, areas, step_factors, stream);
    for (int j = 0; j < RK; j++) {
        compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes, stream);
        time_step(j, nelr, old_variables, variables, step_factors, fluxes, stream);
    }
    rad::checkFrameworkErrors(cudaGetLastError());
}
