/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "kmeans.h"

extern double wtime(void);

void usage(char *argv0) {
    printf("Usage: %s <#points> <#features> <input_filename> <output_filename>\n", argv0);
    printf("  #points: total number of points to be clustered\n");
    printf("  #features: number of features of each point\n");
}

void generateInput(char *input_file, int npoints, int nfeatures)
{
	int i, j;
	FILE *finput;
	if ((finput = fopen(input_file, "wb")) == NULL) {
	    fprintf(stderr, "Error: cannot open file (%s)\n", input_file);
	    exit(1);
	}
	fwrite(&npoints, 1, sizeof(int), finput);
	fwrite(&nfeatures, 1, sizeof(int), finput);

	float buf;

	for ( i = 0; i < npoints; i++ )
	{   
	    for ( j = 0; j < nfeatures; j++ ) { 
	        buf = ( (float)rand() / (float)RAND_MAX );
	        fwrite(&buf, 1, sizeof(float), finput);
	    }
	}
	fclose(finput);
}

void readSize(char *input_file, int *ptrnpoints, int *ptrnfeatures)
{
	FILE *infile;
	int i, ret;

    if ((infile = fopen(input_file, "rb")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", input_file);
        exit(1);
    }
    ret=fread(ptrnpoints,   1, sizeof(int), infile);
    ret=fread(ptrnfeatures, 1, sizeof(int), infile);

    fclose(infile);
}

void readInput(char *input_file, int npoints, int nfeatures, float **features, float *buf)
{
	FILE *infile;
	int i, ret;

    if ((infile = fopen(input_file, "rb")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", input_file);
        exit(1);
    }

    for (i=1; i<npoints; i++)
        features[i] = features[i-1] + nfeatures;
    ret=fread(buf, 1, npoints*nfeatures*sizeof(float), infile);

    fclose(infile);
}

void writeOutput(char *output_file, int min_nclusters, int max_nclusters, int nfeatures, float **cluster_centres)
{
	int i, j;
	FILE *foutput;
	if ((foutput=fopen(output_file, "wb"))==NULL)
	{
		fprintf(stderr, "Error: cannot open file (%s)\n", output_file);
	    exit(1);
	}
	if(min_nclusters == max_nclusters) {
		//printf("\n================= Centroid Coordinates =================\n");
		for(i = 0; i < max_nclusters; i++){
			//printf("%d:", i);
			for(j = 0; j < nfeatures; j++){
				//printf(" %.2f", cluster_centres[i][j]);
				fwrite(&cluster_centres[i][j], 1, sizeof(float), foutput);
			}
			//printf("\n\n");
		}
	}
	fclose(foutput);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
	float	threshold = 0.001;		/* default value */
	int		max_nclusters=5;		/* default value */
	int		min_nclusters=5;		/* default value */
	int		best_nclusters = 0;
	int		nfeatures = 0;
	int		npoints = 0;
	float	len;
	float 	*buf;
	int		i, j, index;
	         
	float **features;
	float **cluster_centres=NULL;
	int		nloops = 1;				/* default value */
			
	int		isRMSE = 0;		
	float	rmse;

	char *input_file, *output_file;

	int enable_perfmeasure=1;

	if (argc!=5)
	{	usage(argv[0]);
		exit(-1);	}
	npoints = atoi(argv[1]);
	nfeatures = atoi(argv[2]);
	input_file = argv[3];
	output_file = argv[4];

	printf("cudaKmeans. npoints=%d nfeatures=%d threshold=%f clusters=%d\n", npoints, nfeatures, threshold, max_nclusters);

/* ============== I/O begin ==============*/
srand(7);												/* seed for future random number generator */	
    FILE *finput;
	if ((finput=fopen(input_file, "rb")) == NULL)
	{
		printf("Generator: input file %s does not exist. Generating a new one...", input_file);fflush(stdout);
		generateInput(input_file, npoints, nfeatures);
		printf("Done.\n");
	}
	else
	{	fclose(finput);		}
	printf("Input %s exists. Reading...\n", input_file);fflush(stdout);

	readSize(input_file, &npoints, &nfeatures);

/* allocate space for features[][] and read attributes of all objects */
    buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
    features    = (float**)malloc(npoints*          sizeof(float*));
    features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
	readInput(input_file, npoints, nfeatures, features, buf);

	// error check for clusters
	if (npoints < min_nclusters)
	{
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
		exit(0);
	}

	memcpy(features[0], buf, npoints*nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
	free(buf);

	/* ======================= core of the clustering ===================*/

    //cluster_timing = omp_get_wtime();		/* Total clustering time */
	cluster_centres = NULL;
    index = cluster(npoints,				/* number of data points */
					nfeatures,				/* number of features for each point */
					features,				/* array: [npoints][nfeatures] */
					min_nclusters,			/* range of min to max number of clusters */
					max_nclusters,
					threshold,				/* loop termination factor */
				   &best_nclusters,			/* return: number between min and max */
				   &cluster_centres,		/* return: [best_nclusters][nfeatures] */  
				   &rmse,					/* Root Mean Squared Error */
					isRMSE,					/* calculate RMSE */
					nloops,					/* number of iteration for each number of clusters */		
					enable_perfmeasure);
    
	//cluster_timing = omp_get_wtime() - cluster_timing;

	writeOutput(output_file, min_nclusters, max_nclusters, nfeatures, cluster_centres);
	
	len = (float) ((max_nclusters - min_nclusters + 1)*nloops);

	printf("Number of Iteration: %d\n", nloops);
	//printf("Time for I/O: %.5fsec\n", io_timing);
	//printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);
	
	if(min_nclusters != max_nclusters){
		if(nloops != 1){									//range of k, multiple iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
		else{												//range of k, single iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
	}
	else{
		if(nloops != 1){									// single k, multiple iteration
			//printf("Average Clustering Time: %.5fsec\n",
			//		cluster_timing / nloops);
			if(isRMSE)										// if calculated RMSE
				printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
		}
		else{												// single k, single iteration				
			if(isRMSE)										// if calculated RMSE
				printf("Root Mean Squared Error: %.3f\n", rmse);
		}
	}
	

	/* free up memory */
	free(features[0]);
	free(features);    
    return(0);
}

