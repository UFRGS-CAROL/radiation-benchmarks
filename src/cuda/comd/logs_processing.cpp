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
#include "neighborList.h"

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
void load_gold(Gold *g) {

}

//TODO
void init_gold(Gold *g, char *gold_path, int steps) {
	strcpy(g->gold_path, gold_path);
	g->steps = steps;

	// alloc the memory
	g->gold_data = (SimFlat**) calloc(steps, sizeof(SimFlat*));

}
//TODO:
void compare_and_log(SimFlat *gold, SimFlat *found) {

}

void destroy_gold(Gold *g) {
	if (g->gold_data) {
		free(g->gold_data);
	}
}

static inline void copy_haloexchange(HaloExchange *original,
		HaloExchange *copy) {
	copy->bufCapacity = original->bufCapacity;
	ForceExchangeParms* parms = (ForceExchangeParms*) copy->parms;
	ForceExchangeParms* ori_parms = (ForceExchangeParms*) original->parms;

	//TODO: needs verification
	for (int i = 0; i < 6; i++) {
		copy->nbrRank[i] = original->nbrRank[i];
		memcpy(parms->natoms_buf[i], ori_parms->natoms_buf[i],
				sizeof(int) * parms->nCells[i]);
		memcpy(parms->sendCells[i], ori_parms->sendCells[i],
				sizeof(int) * parms->nCells[i]);
		cudaMemcpy(parms->sendCellsGpu[i], ori_parms->sendCellsGpu[i],
				sizeof(int) * parms->nCells[i], cudaMemcpyDeviceToDevice);
		memcpy(parms->recvCells[i], ori_parms->recvCells[i],
				sizeof(int) * parms->nCells[i]);
		cudaMemcpy(parms->recvCellsGpu[i], ori_parms->recvCellsGpu[i],
				sizeof(int) * parms->nCells[i], cudaMemcpyDeviceToDevice);
		memcpy(parms->partial_sums[i], ori_parms->partial_sums[i],
				sizeof(int) * parms->nCells[i]);

	}

	copy->hashTable->nEntriesGet = original->hashTable->nEntriesGet;
	copy->hashTable->nEntriesPut = original->hashTable->nEntriesPut;
	copy->hashTable->nMaxEntries = original->hashTable->nMaxEntries;

	memcpy(copy->hashTable->offset, original->hashTable->offset,
			sizeof(int) * copy->hashTable->nMaxEntries);

	copy->destroy = original->destroy;
	copy->loadBuffer = original->loadBuffer;
	strcpy(copy->recvBufM, original->recvBufM);
	strcpy(copy->recvBufP, original->recvBufP);
	strcpy(copy->sendBufM, original->sendBufM);
	strcpy(copy->sendBufP, original->sendBufP);

	copy->type = original->type;
	copy->unloadBuffer = original->unloadBuffer;
}

static inline void copy_vect(vec_t* original, vec_t* copy, int n) {
	memcpy(copy->x, original->x, sizeof(real_t) * n);
	memcpy(copy->y, original->y, sizeof(real_t) * n);
	memcpy(copy->z, original->z, sizeof(real_t) * n);
}

static inline void copy_neighborlistst(NeighborList* original,
		NeighborList* copy) {
	copy->nMaxLocal = original->nMaxLocal; // make this list a little larger to make room for migrated particles
	copy->maxNeighbors = original->maxNeighbors; //TODO: choose this value dynamically
	copy->skinDistance = original->skinDistance;
	copy->skinDistance2 = original->skinDistance2;
	copy->skinDistanceHalf2 = original->skinDistanceHalf2;
	copy->nStepsSinceLastBuild = original->nStepsSinceLastBuild;
	copy->updateNeighborListRequired = original->updateNeighborListRequired;
	copy->updateLinkCellsRequired = original->updateLinkCellsRequired;
	copy->forceRebuildFlag = original->forceRebuildFlag;

	memcpy(copy->list, original->list,
			sizeof(int) * original->nMaxLocal * original->maxNeighbors);
	memcpy(copy->nNeighbors, original->nNeighbors,
			sizeof(int) * original->nMaxLocal);
	//malloc_vec(&(neighborList->lastR), neighborList->nMaxLocal);
	copy_vect(&(original->lastR), &(copy->lastR), original->nMaxLocal);
}

static inline void copy_atoms(Atoms* original, Atoms* copy, int nTotalBoxes,
		int nLocalBoxes) {
	/*
	 *   // atom-specific data
	 int nLocal;    //!< total number of atoms on this processor
	 int* lid;      //!< A locally unique id for each atom (used for the neighborlist)
	 int nGlobal;   //!< total number of atoms in simulation

	 int* gid;      //!< A globally unique id for each atom
	 int* iSpecies; //!< the species index of the atom

	 struct NeighborListSt* neighborList;

	 vec_t r;     //!< positions
	 vec_t p;     //!< momenta of atoms
	 vec_t f;     //!< forces
	 real_t* U;     //!< potential energy per atom
	 */

	copy->nLocal = original->nLocal;
	copy->nGlobal = original->nGlobal;
	copy_neighborlistst(original->neighborList, copy->neighborList);
	int maxTotalAtoms = MAXATOMS * nTotalBoxes + 1; //one atom placed at infinity for neighborlist implementation

	copy_vect(&(original->r), &(copy->r), maxTotalAtoms);
	copy_vect(&(original->p), &(copy->p), maxTotalAtoms);
	copy_vect(&(original->f), &(copy->f), maxTotalAtoms);

	memcpy(copy->U, original->U, maxTotalAtoms * sizeof(real_t));
	memcpy(copy->gid, original->gid, maxTotalAtoms * sizeof(int));
	memcpy(copy->lid, original->lid, MAXATOMS * nLocalBoxes * sizeof(int));
	memcpy(copy->iSpecies, original->iSpecies, maxTotalAtoms * sizeof(int));

}

static inline void copy_domain(Domain* original, Domain* copy) {
	const int size = 3;
	memcpy(copy->globalExtent, original->globalExtent, sizeof(real_t) * size);
	memcpy(copy->globalMax, original->globalMax, sizeof(real_t) * size);
	memcpy(copy->globalMin, original->globalMin, sizeof(real_t) * size);
	memcpy(copy->localExtent, original->localExtent, sizeof(real_t) * size);
	memcpy(copy->localMax, original->localMax, sizeof(real_t) * size);
	memcpy(copy->localMin, original->localMin, sizeof(real_t) * size);
	memcpy(copy->procCoord, original->procCoord, sizeof(int) * size);
	memcpy(copy->procGrid, original->procGrid, sizeof(int) * size);
}

static inline void copy_linkcell(LinkCell* original, LinkCell* copy) {
	/*
	 *
	 int gridSize[3];     //!< number of boxes in each dimension on processor
	 int nLocalBoxes;     //!< total number of local boxes on processor
	 int nHaloBoxes;      //!< total number of remote halo/ghost boxes on processor
	 int nTotalBoxes;     //!< total number of boxes on processor
	 //!< nLocalBoxes + nHaloBoxes
	 real3_old localMin;      //!< minimum local bounds on processor
	 real3_old localMax;      //!< maximum local bounds on processor
	 real3_old boxSize;       //!< size of box in each dimension
	 real3_old invBoxSize;    //!< inverse size of box in each dimension

	 int *boxIDLookUp; //!< 3D array storing the box IDs
	 int3_t *boxIDLookUpReverse; //!< 1D array storing the tuple for a given box ID

	 int* nAtoms;         //!< total number of atoms in each box
	 */

	const int size = 3;
	memcpy(copy->gridSize, original->gridSize, sizeof(int) * size);
	copy->nLocalBoxes = original->nLocalBoxes;
	copy->nHaloBoxes = original->nHaloBoxes;
	copy->nTotalBoxes = original->nTotalBoxes;
	memcpy(copy->localMax, original->localMax, sizeof(real_t) * size);
	memcpy(copy->localMin, original->localMin, sizeof(real_t) * size);
	memcpy(copy->boxSize, original->boxSize, sizeof(real_t) * size);
	memcpy(copy->invBoxSize, original->invBoxSize, sizeof(real_t) * size);
	memcpy(copy->boxIDLookUp, original->boxIDLookUp,
			original->gridSize[0] * original->gridSize[1]
					* original->gridSize[2] * sizeof(int));
	memcpy(copy->boxIDLookUpReverse, original->boxIDLookUpReverse,
			original->nLocalBoxes * sizeof(int3_t));
	memcpy(copy->nAtoms, original->nAtoms, original->nTotalBoxes * sizeof(int));
}

static inline void copy_speciesdata(SpeciesData* original, SpeciesData *copy) {
	/*
	 *    char  name[3];   //!< element name
	 int	 atomicNo;  //!< atomic number
	 real_t mass;
	 */
	memcpy(copy->name, original->name, 3);
	copy->atomicNo = original->atomicNo;
	copy->mass = original->mass;
}

static inline void copy_basepotential(BasePotential* original,
		BasePotential* copy) {
	/*
	 *    real_t cutoff;          //!< potential cutoff distance in Angstroms
	 real_t mass;            //!< mass of atoms in intenal units
	 real_t lat;             //!< lattice spacing (angs) of unit cell
	 char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
	 char  name[3];	   //!< element name
	 int	 atomicNo;	   //!< atomic number
	 int  (*force)(struct SimFlatSt* s); //!< function pointer to force routine
	 void (*print)(FILE* file, struct BasePotentialSt* pot);
	 void (*destroy)(struct BasePotentialSt** pot); //!< destruction of the potential
	 */
	copy->cutoff = original->cutoff;
	copy->mass = original->mass;
	copy->lat = original->lat;
	copy->atomicNo = original->atomicNo;
	copy->force = original->force;
	copy->print = original->print;
	copy->destroy = original->destroy;
	memcpy(copy->latticeType, original->latticeType, 8);
	memcpy(copy->name, original->name, 3);
}

static inline void copy_atomsgpu_host(AtomsGpu* original, AtomsGpu* copy){
	/*
	 *   vec_t r;			// atoms positions
  vec_t p;			// atoms momentum
  vec_t f;			// atoms forces
  real_t *e;		// atoms energies
  int *iSpecies;  // atoms species id
  int *gid;			// atoms global id
  NeighborListGpu neighborList;
	 */
	memcpy(&(copy->r), &(original->r), sizeof(vec_t));
	memcpy(&(copy->f), &(original->f), sizeof(vec_t));
	memcpy(&(copy->p), &(original->p), sizeof(vec_t));

}

static inline void copy_simgpu(SimGpu* original, SimGpu* copy) {
	/*
	 *   int max_atoms_cell;		// max atoms per cell (usually < 32)

	 AtomsGpu atoms;

	 int *neighbor_cells;		// neighbor cells indices
	 int *neighbor_atoms;		// neighbor atom offsets
	 int *num_neigh_atoms;		// number of neighbor atoms per cell

	 // species data
	 real_t *species_mass;		// masses of species

	 LinkCellGpu boxes;

	 HashTableGpu d_hashTable;
	 int *d_updateLinkCellsRequired; //flag that indicates that another linkCell() call is required //TODO this should be haloExchangeRequired()
	 int *cell_type;		// type of cell: 0 - interior, 1 - boundary

	 AtomListGpu a_list;		// all local cells
	 AtomListGpu i_list;		// interior cells
	 AtomListGpu b_list;		// boundary cells

	 // potentials
	 LjPotentialGpu lj_pot;
	 EamPotentialGpu eam_pot;

	 int * pairlist;

	 int genPairlist;
	 int usePairlist;
	 *
	 */

	copy->max_atoms_cell = original->max_atoms_cell;


}

void copy_input_iteration(SimFlat* original, SimFlat* copy) {
	/*
	 * Domain* domain;        //<! domain decomposition data
	 LinkCell* boxes;       //<! link-cell data
	 Atoms* atoms;          //<! atom data (positions, momenta, ...)
	 SpeciesData* species;  //<! species data (per species, not per atom)
	 BasePotential *pot;	  //!< the potential
	 HaloExchange* atomExchange;
	 SimGpu host;		// host data for staging
	 SimGpu gpu;		// GPU data
	 */
	copy_domain(original->domain, copy->domain);
	copy_linkcell(original->boxes, copy->boxes);
	copy_atoms(original->atoms, copy->atoms, original->boxes->nTotalBoxes,
			original->boxes->nLocalBoxes);
	copy_speciesdata(original->species, copy->species);
	copy_basepotential(original->pot, copy->pot);
	copy_haloexchange(original->atomExchange, copy->atomExchange);
	copy_simgpu(&(original->host), &(copy->host));
	copy_simgpu(&(original->gpu), &(copy->gpu));
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
	 */
	copy->nSteps = original->nSteps;
	copy->printRate = original->printRate;
	copy->dt = original->dt;
	memcpy(&(copy->eKinetic), &(original->eKinetic), sizeof(real_t));
	memcpy(&(copy->ePotential), &(original->ePotential), sizeof(real_t));
	copy->method = original->method;
	copy->n_boundary_cells = original->n_boundary_cells;
	copy->n_boundary1_cells = original->n_boundary1_cells;
	memcpy(copy->boundary_cells, original->boundary_cells,
			sizeof(int) * copy->n_boundary_cells);
	memcpy(copy->boundary1_cells_h, original->boundary1_cells_h,
			sizeof(int) * copy->n_boundary1_cells);
	cudaMemcpy(copy->boundary1_cells_d, original->boundary1_cells_d,
			sizeof(int) * copy->n_boundary1_cells, cudaMemcpyDeviceToDevice);
	memcpy(&(copy->boundary_stream), &(original->boundary_stream),
			sizeof(cudaStream_t));
	memcpy(&(copy->interior_stream), &(original->interior_stream),
			sizeof(cudaStream_t));
	copy->gpuAsync = original->gpuAsync;
	copy->gpuProfile = original->gpuProfile;
	copy->ljInterpolation = original->ljInterpolation;
	copy->spline = original->spline;
	memcpy(&(copy->skinDistance), &(original->skinDistance), sizeof(real_t));
	copy->usePairlist = original->usePairlist;
	memcpy(&(copy->skinDistance), &(original->skinDistance), sizeof(real_t));

	cudaMemcpy(copy->flags, original->flags,
			original->boxes->nTotalBoxes * MAXATOMS * sizeof(int),
			cudaMemcpyDeviceToDevice);

	cudaMemcpy(copy->tmp_sort, original->tmp_sort,
			original->boxes->nTotalBoxes * MAXATOMS * sizeof(int),
			cudaMemcpyDeviceToDevice);
	cudaMemcpy(copy->gpu_atoms_buf, original->gpu_atoms_buf,
			original->boxes->nTotalBoxes * MAXATOMS * sizeof(AtomMsg),
			cudaMemcpyDeviceToDevice);
	cudaMemcpy(copy->gpu_force_buf, original->gpu_force_buf,
			original->boxes->nTotalBoxes * MAXATOMS * sizeof(ForceMsg),
			cudaMemcpyDeviceToDevice);

//	copy->boundary1_cells_d

}

