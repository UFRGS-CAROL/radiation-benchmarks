/*
 * logs_processing.cpp
 *
 *  Created on: 07/09/2017
 *      Author: fernando
 */

#include "logs_processing.h"
#include <string.h>
#include <cstdlib>

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


