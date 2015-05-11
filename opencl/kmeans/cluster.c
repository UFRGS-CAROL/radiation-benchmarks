#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "kmeans.h"


int cluster(int      npoints,				/* number of data points */
            int      nfeatures,				/* number of attributes for each point */
            float  **features,			/* array: [npoints][nfeatures] */                  
            int      min_nclusters,			/* range of min to max number of clusters */
	    int		 max_nclusters,
            float    threshold,				/* loop terminating factor */
            float ***cluster_centres,		/* out: [best_nclusters][nfeatures] */
	    int		 nloops					/* number of iteration for each number of clusters */
			)
{    
	int	nclusters; /* number of clusters k */	
	int    *membership; /* which cluster a data point belongs to */
	float **tmp_cluster_centres; /* hold coordinates of cluster centers */
	int		i;
	/* allocate memory for membership */
	membership = (int*) malloc(npoints * sizeof(int));

	/* sweep k from min to max_nclusters to find the best number of clusters */
	for(nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
	{
		if (nclusters > npoints) break;	/* cannot have more clusters than points */

		/* allocate device memory, invert data array (@ kmeans_cuda.cu) */
		allocate(npoints, nfeatures, nclusters, features);

		/* iterate nloops times for each number of clusters */
		for(i = 0; i < nloops; i++)
		{
			/* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
			tmp_cluster_centres = kmeans_clustering(features, nfeatures, npoints, nclusters, threshold, membership);
			if (*cluster_centres) {
				free((*cluster_centres)[0]);
				free(*cluster_centres);
			}
			*cluster_centres = tmp_cluster_centres;
		}
		deallocateMemory(); /* free device memory (@ kmeans_cuda.cu) */
	}
    free(membership);
    return 0;
}

