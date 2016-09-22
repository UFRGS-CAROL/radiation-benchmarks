/*
 * log_processing.h
 *
 *  Created on: 22/09/2016
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

#include "list.h"

typedef struct gold_input{
	list *input_names;
	list **input_data;
	int classes;
}GoldInput;

void make_gold_input(GoldInput *gold){
	gold->input_names = make_list();
}

void add_new_name(GoldInput *gold, char *new_name){
	list_insert(&gold->input_names, new_name);
}

void load_input_files(GoldInput *gold){
	gold->input_data = calloc(gold->input_names->size, sizeof(list));

	int i = 0;
	FILE *pf;
	for(i = 0; i < gold->input_names->size; i++){
		char *file_name = get(gold->input_names, i);
		if(!(pf = fopen(file_name, "r"))){
			exit(EXIT_FAILURE);
		}
		//for each input_name I need open and put all data in a list
		gold->input_data[i] = make_list();
		int j =0;
		for(j = 0; j < gold->classes; j++){
			char *id = calloc (100, sizeof(char));;
			float *pp = calloc(5, sizeof(float)); //probality and positions
			fscanf(pf, "%s %f %f %f %f %f\n", id, pp[0],pp[1], pp[2],pp[3],pp[4]);
			list_insert(gold->input_data[i], id);
			list_insert(gold->input_data[i], pp);
		}
	}

}

void free_gold_input(GoldInput *gold){
	int i;
	for(i = 0; i < gold->input_names->size; i++)
		free_all(gold->input_data[i]);

	free_all(gold->input_data);
	free_all(gold->input_names);
	gold->classes = 0;
}

#endif /* LOG_PROCESSING_H_ */
