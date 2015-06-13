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
    printf("Usage: %s <input_filename> <output_filename> <#iteractions>\n", argv0);
    printf("  #points: total number of points to be clustered\n");
    printf("  #features: number of features of each point\n");
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

void readGold(char *gold_file, int max_nclusters, int nfeatures, float **gold_cluster_centres)
{
	int i, j;
	FILE *fgold;
	if ((fgold=fopen(gold_file, "rb"))==NULL)
	{	fprintf(stderr, "Error: no such file (%s)\n", gold_file);
        exit(1);
    }	
	for (i=1; i<max_nclusters; i++)
		gold_cluster_centres[i]=gold_cluster_centres[i-1]+nfeatures;

	int ret = fread(gold_cluster_centres[0], 1, max_nclusters*nfeatures*sizeof(float), fgold);
	printf("Got %d gold entries.\n", ret);
	fclose(fgold);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
	int		opt;
 	extern char   *optarg;
	char   *filename = 0;
	float  *buf;
	char	line[1024];
	int		isBinaryFile = 0;

	float	threshold = 0.001;		/* default value */
	int		max_nclusters=5;		/* default value */
	int		min_nclusters=5;		/* default value */
	int		best_nclusters = 0;
	int		nfeatures = 0;
	int		npoints = 0;
	float	len;
	         
	float **features;
	float **cluster_centres=NULL;
	float **gold_cluster_centres;
	int		i, j, index;
	int		nloops = 1;				/* default value */
			
	int		isRMSE = 0;		
	float	rmse;
	int		loop1;
	
	char *input_file, *output_file;
	int iteractions;

	int enable_perfmeasure = 0;

	if (argc!=4)
	{	usage(argv[0]);
		exit(-1);	}
	input_file = argv[1];
	output_file = argv[2];
	iteractions = atoi(argv[3]);
	

	/* ============== I/O begin ==============*/
	readSize(input_file, &npoints, &nfeatures);
	gold_cluster_centres    = (float**)malloc(max_nclusters*          sizeof(float*));
	gold_cluster_centres[0] = (float*) malloc(max_nclusters*nfeatures*sizeof(float));
	readGold(output_file, max_nclusters, nfeatures, gold_cluster_centres);

	printf("cudaKmeans. npoints=%d nfeatures=%d threshold=%f clusters=%d ITERACTIONS=%d\n", npoints, nfeatures, threshold, max_nclusters, iteractions);fflush(stdout);
	for (loop1=0; loop1<iteractions; loop1++)
	{
		/* ============== I/O begin ==============*/
		/* allocate space for features[][] and read attributes of all objects */
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		features    = (float**)malloc(npoints*          sizeof(float*));
		features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
		readInput(input_file, npoints, nfeatures, features, buf);
	
		/* ============== I/O end ==============*/

		// error check for clusters
		if (npoints < min_nclusters)
		{
			printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
			exit(0);
		}

		srand(7);												/* seed for future random number generator */	
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


		/* =============== Command Line Output =============== */

		/* cluster center coordinates
		   :displayed only for when k=1*/
		//printf("\n================= Centroid Coordinates =================\n");
		for(i = 0; i < max_nclusters; i++){
			//printf("%d:", i);
			for(j = 0; j < nfeatures; j++){
				//printf(" %.2f", cluster_centres[i][j]);
				if (gold_cluster_centres[i][j]!=cluster_centres[i][j])
					printf("ERROR(e:%f r:%f)", gold_cluster_centres[i][j], cluster_centres[i][j]);
			}
			//printf("\n\n");
		}

		printf(".");fflush(stdout);

		/* free up memory */
		free(features[0]);
		free(features);    
	}
    return(0);
}

