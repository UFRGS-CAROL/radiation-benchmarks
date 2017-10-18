/*
 * logs_processing.cpp
 *
 *  Created on: 07/09/2017
 *      Author: fernando
 */

#include "logs_processing.h"
#include <string.h>
#include <cstdlib>
#include "hashTable.h"

#ifdef LOGS
#include "log_helper.h"
#endif

void start_count_app(char *gold_file, int iterations) {
#ifdef LOGS
	std::string header = "gold_file: " + std::string(gold_file) + " iterations: " + std::to_string(iterations);
	start_log_file("comd", header.c_std());
#endif
}

void finish_count_app() {
#ifdef LOGS
	end_log_file();
#endif
}

void start_iteration_app() {
#ifdef LOGS
	start_iteration();
#endif
}

void end_iteration_app() {
#ifdef LOGS
	end_iteration();
#endif
}

//TODO:
void save_gold(Gold *g) {

	//TODO: convert
	// atoms per cell
//	int atoms_per_cell_hist[size];
//	memset(atoms_per_cell_hist, 0, size * sizeof(int));
//	for (int iBox = 0; iBox < sim->boxes->nLocalBoxes; iBox++)
//		atoms_per_cell_hist[sim->boxes->nAtoms[iBox]]++;


	//TODO: convert
	// cell neighbors
//	int cell_neigh_hist[size];
//	memset(cell_neigh_hist, 0, size * sizeof(int));
//	for (int iBox = 0; iBox < sim->boxes->nLocalBoxes; iBox++) {
//		int neighbor_atoms = 0;
//		for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
//			int jBox = sim->host.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
//			neighbor_atoms += sim->boxes->nAtoms[jBox];
//		}
//		cell_neigh_hist[neighbor_atoms] += sim->boxes->nAtoms[iBox];
//	}


	//TODO: convert
	// find # of atoms in neighbor lists (cut-off + skin distance)
//	int neigh_lists_hist[size];
//	memset(neigh_lists_hist, 0, size * sizeof(int));


	//TODO: convert
	// find # of neighbors strictly under cut-off
//	int passed_cutoff_hist[size];
//	memset(passed_cutoff_hist, 0, size * sizeof(int));
//	for (int iBox = 0; iBox < sim->boxes->nLocalBoxes; iBox++) {
//		for (int iAtom = 0; iAtom < sim->boxes->nAtoms[iBox]; iAtom++) {
//			int passed_atoms = 0;
//			for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
//				int jBox = sim->host.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
//				for (int jAtom = 0; jAtom < sim->boxes->nAtoms[jBox]; jAtom++) {
//					int i_particle = iBox * MAXATOMS + iAtom;
//					int j_particle = jBox * MAXATOMS + jAtom;
//
//					real_t dx = sim->atoms->r.x[i_particle]
//							- sim->atoms->r.x[j_particle];
//					real_t dy = sim->atoms->r.y[i_particle]
//							- sim->atoms->r.y[j_particle];
//					real_t dz = sim->atoms->r.z[i_particle]
//							- sim->atoms->r.z[j_particle];
//
//					real_t r2 = dx * dx + dy * dy + dz * dz;
//
//					// TODO: this is only works for EAM potential
//					if (r2 < sim->gpu.eam_pot.cutoff * sim->gpu.eam_pot.cutoff)
//						passed_atoms++;
//				}
//			}
//			passed_cutoff_hist[passed_atoms]++;
//		}
//	}


	//TODO: convert
//	char fileName[100];
//	sprintf(fileName, "histogram_%i.csv", step);
//	FILE *file = fopen(fileName, "w");
//	fprintf(file,
//			"# of atoms,cell size = cutoff,neighbor cells,cutoff = 4.95 + 10%%,cutoff = 4.95,\n");
//	for (int i = 0; i < size; i++)
//		fprintf(file, "%i,%i,%i,%i,%i,\n", i, atoms_per_cell_hist[i],
//				cell_neigh_hist[i], neigh_lists_hist[i], passed_cutoff_hist[i]);



}

//TODO:
void load_gold(Gold *g){

}

//TODO
void init_gold(Gold *g, char *gold_path, int steps){
	strcpy(g->gold_path, gold_path);
	g->steps = steps;

	// alloc the memory
	g->gold_data = (SimFlat**) calloc(steps, sizeof(SimFlat*));

}
//TODO:
void compare_and_log(SimFlat *gold, SimFlat *found){

}

void destroy_gold(Gold *g){
	if(g->gold_data){
		free(g->gold_data);
	}
}


static inline void copy_haloexchange(HaloExchange *original, HaloExchange *copy){
	copy->bufCapacity = original->bufCapacity;
	ForceExchangeParms* parms = (ForceExchangeParms*) copy->parms;
	ForceExchangeParms* ori_parms = (ForceExchangeParms*) original->parms;

	//TODO: needs verification
	for(int i = 0; i < 6; i++){
		copy->nbrRank[i] = original->nbrRank[i];
		memcpy(parms->natoms_buf[i], ori_parms->natoms_buf[i], sizeof(int) * parms->nCells[i]);
		memcpy(parms->sendCells[i], ori_parms->sendCells[i], sizeof(int) * parms->nCells[i]);
		cudaMemcpy(parms->sendCellsGpu[i], ori_parms->sendCellsGpu[i], sizeof(int) * parms->nCells[i], cudaMemcpyDeviceToDevice);
		memcpy(parms->recvCells[i], ori_parms->recvCells[i], sizeof(int) * parms->nCells[i]);
		cudaMemcpy(parms->recvCellsGpu[i], ori_parms->recvCellsGpu[i], sizeof(int) * parms->nCells[i], cudaMemcpyDeviceToDevice);
		memcpy(parms->partial_sums[i], ori_parms->partial_sums[i], sizeof(int) * parms->nCells[i]);

	}


	copy->hashTable->nEntriesGet = original->hashTable->nEntriesGet;
	copy->hashTable->nEntriesPut = original->hashTable->nEntriesPut;
	copy->hashTable->nMaxEntries = original->hashTable->nMaxEntries;

	memcpy(copy->hashTable->offset, original->hashTable->offset, sizeof(int) * copy->hashTable->nMaxEntries);

	copy->destroy = original->destroy;
	copy->loadBuffer = original->loadBuffer;
	strcpy(copy->recvBufM, original->recvBufM);
	strcpy(copy->recvBufP, original->recvBufP);
	strcpy(copy->sendBufM, original->sendBufM);
	strcpy(copy->sendBufP, original->sendBufP);

	copy->type = original->type;
	copy->unloadBuffer = original->unloadBuffer;
}

static inline void copy_atoms(Atoms* original, Atoms* copy){

}


/**
 * int nSteps;            //<! number of time steps to run
   int printRate;         //<! number of steps between output
   double dt;             //<! time step
   real_t ePotential;     //!< the total potential energy of the system
   real_t eKinetic;       //!< the total kinetic energy of the system
   int method;
   int n_boundary_cells;	// note that this is 2-size boundary to allow overlap for atoms exchange
   int *boundary_cells;		//<! Two most outer rings of the local cells
   int *interior_cells;    //<! device array local cells excluding boundary_cells
   int n_boundary1_cells;	// 1-size boundary - necessary for sorting only
   int *boundary1_cells_d;	//<! boundary cells that are neighbors to halos (outer ring of the local Cells)
   int *boundary1_cells_h;	//<! boundary cells that are neighbors to halos (outer ring of the local Cells)
   cudaStream_t boundary_stream;
   cudaStream_t interior_stream;
   int gpuAsync;
   int gpuProfile;

   int *flags;			// flags for renumbering
   int *tmp_sort;		// temp array for merge sort
   char *gpu_atoms_buf;		// buffer for atoms exchange
   char *gpu_force_buf;		// buffer for forces exchange

   int ljInterpolation;     //<! compute Lennard-Jones potential using interpolation
   int spline;              //<! use splines for interpolation

   int usePairlist;         //<! use pairlists for cta_cell method in Lennard-Jones computation

   real_t skinDistance;



   Domain* domain;        //<! domain decomposition data
   LinkCell* boxes;       //<! link-cell data
   Atoms* atoms;          //<! atom data (positions, momenta, ...)
   SpeciesData* species;  //<! species data (per species, not per atom)
   BasePotential *pot;	  //!< the potential
   HaloExchange* atomExchange;
   SimGpu host;		// host data for staging
   SimGpu gpu;		// GPU data
 */

void copy_input_iteration(SimFlat* original, SimFlat* copy){
	copy_haloexchange(original->atomExchange, copy->atomExchange);
	copy_atoms(original->atoms, copy->atoms);

//	copy->boundary1_cells_d

}


