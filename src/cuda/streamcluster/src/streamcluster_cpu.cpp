/***********************************************
 streamcluster.cpp
 : original source code of streamcluster with minor
 modification regarding function calls

 - original code from PARSEC Benchmark Suite
 - parallelization with CUDA API has been applied by

 Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
 University of Virginia
 Department of Electrical and Computer Engineering
 Department of Computer Science

 ***********************************************/

#include <vector>
#include <tuple>
#include <string>

#include "streamcluster.h"
#include "cuda_utils.h"

//using namespace std;

#define MAXNAMESIZE 1024 			// max filename length#define SEED 1#define SP 1 						// number of repetitions of speedy must be >=1#define ITER 3 						// iterate ITER* k log k times; ITER >= 1//#define INSERT_WASTE				// Enables waste computation in dist function#define CACHE_LINE 512				// cache line in byte// GLOBALstatic bool *switch_membership;		//whether to switch membership in pgainstatic bool *is_center;				//whether a point is a centerstatic int *center_table;			//index table of centersstatic int nproc; 					//# of threadsbool isCoordChanged;
// GPU Timing Info
double serial_t;
double cpu_to_gpu_t;
double gpu_to_cpu_t;
double alloc_t;
double kernel_t;
double free_t;

void inttofile(int data, char *filename) {
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "%d ", data);
	fclose(fp);
}

int isIdentical(float *i, float *j, int D) {
// tells whether two points of D dimensions are identical

	int a = 0;
	int equal = 1;

	while (equal && a < D) {
		if (i[a] != j[a])
			equal = 0;
		else
			a++;
	}
	if (equal)
		return 1;
	else
		return 0;

}

/* shuffle points into random order */
void shuffle(Points *points) {
	long i, j;
	Point temp;
	for (i = 0; i < points->num - 1; i++) {
		j = (lrand48() % (points->num - i)) + i;
		temp = points->p[i];
		points->p[i] = points->p[j];
		points->p[j] = temp;
	}
}

/* shuffle an array of integers */
void intshuffle(int *intarray, int length) {
	long i, j;
	int temp;
	for (i = 0; i < length; i++) {
		j = (lrand48() % (length - i)) + i;
		temp = intarray[i];
		intarray[i] = intarray[j];
		intarray[j] = temp;
	}
}

#ifdef INSERT_WASTE
float waste(float s )
{
	for( int i =0; i< 4; i++ ) {
		s += pow(s,0.78);
	}
	return s;
}
#endif

/* compute Euclidean distance squared between two points */
float dist(Point p1, Point p2, int dim) {
	int i;
	float result = 0.0;
	for (i = 0; i < dim; i++)
		result += (p1.coord[i] - p2.coord[i]) * (p1.coord[i] - p2.coord[i]);
#ifdef INSERT_WASTE
	float s = waste(result);
	result += s;
	result -= s;
#endif
	return (result);
}

/* run speedy on the points, return total cost of solution */
float pspeedy(Points *points, float z, long *kcenter, int pid,
		pthread_barrier_t* barrier) {
	//my block
	long bsize = points->num / nproc;
	long k1 = bsize * pid;
	long k2 = k1 + bsize;
	if (pid == nproc - 1)
		k2 = points->num;

	static float totalcost;

	static bool open = false;
	static float* costs; //cost for each thread.
	static int i;

	/* create center at first point, send it to itself */
	for (int k = k1; k < k2; k++) {
		float distance = dist(points->p[k], points->p[0], points->dim);
		points->p[k].cost = distance * points->p[k].weight;
		points->p[k].assign = 0;
	}

	if (pid == 0) {
		*kcenter = 1;
		costs = (float*) malloc(sizeof(float) * nproc);
	}

	if (pid != 0) { // we are not the master threads. we wait until a center is opened.
		while (1) {
			if (i >= points->num)
				break;
			for (int k = k1; k < k2; k++) {
				float distance = dist(points->p[i], points->p[k], points->dim);
				if (distance * points->p[k].weight < points->p[k].cost) {
					points->p[k].cost = distance * points->p[k].weight;
					points->p[k].assign = i;
				}
			}
		}
	} else { // I am the master thread. I decide whether to open a center and notify others if so.
		for (i = 1; i < points->num; i++) {
			bool to_open = ((float) lrand48() / (float) INT_MAX)
					< (points->p[i].cost / z);
			if (to_open) {
				(*kcenter)++;
				open = true;
				for (int k = k1; k < k2; k++) {
					float distance = dist(points->p[i], points->p[k],
							points->dim);
					if (distance * points->p[k].weight < points->p[k].cost) {
						points->p[k].cost = distance * points->p[k].weight;
						points->p[k].assign = i;
					}
				}
				open = false;
			}
		}
		open = true;
	}

	open = false;
	float mytotal = 0;
	for (int k = k1; k < k2; k++) {
		mytotal += points->p[k].cost;
	}
	costs[pid] = mytotal;

	// aggregate costs from each thread
	if (pid == 0) {
		totalcost = z * (*kcenter);
		for (int i = 0; i < nproc; i++) {
			totalcost += costs[i];
		}
		free(costs);
	}

	return (totalcost);
}

/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */

float pFL(Points *points, int *feasible, int numfeasible, float z, long *k,
		int kmax, float cost, long iter, float e, int pid,
		pthread_barrier_t* barrier) {
	long i;
	long x;
	float change;
	long numberOfPoints;

	change = cost;
	/* continue until we run iter iterations without improvement */
	/* stop instead if improvement is less than e */
	while (change / cost > 1.0 * e) {
		change = 0.0;
		numberOfPoints = points->num;
		/* randomize order in which centers are considered */

		if (pid == 0) {
			intshuffle(feasible, numfeasible);
		}

		for (i = 0; i < iter; i++) {
			x = i % numfeasible;
			change += pgain(feasible[x], points, z, k, kmax, is_center,
					center_table, switch_membership, isCoordChanged, &serial_t,
					&cpu_to_gpu_t, &gpu_to_cpu_t, &alloc_t, &kernel_t, &free_t);
		}

		cost -= change;
	}
	return (cost);
}

int selectfeasible_fast(Points *points, int **feasible, int kmin, int pid,
		pthread_barrier_t* barrier) {

	int numfeasible = points->num;
	if (numfeasible > (ITER * kmin * log((float) kmin)))
		numfeasible = (int) (ITER * kmin * log((float) kmin));
	*feasible = (int *) malloc(numfeasible * sizeof(int));

	float* accumweight;
	float totalweight;

	/*
	 Calcuate my block.
	 For now this routine does not seem to be the bottleneck, so it is not parallelized.
	 When necessary, this can be parallelized by setting k1 and k2 to
	 proper values and calling this routine from all threads ( it is called only
	 by thread 0 for now ).
	 Note that when parallelized, the randomization might not be the same and it might
	 not be difficult to measure the parallel speed-up for the whole program.
	 */
	//  long bsize = numfeasible;
	long k1 = 0;
	long k2 = numfeasible;

	float w;
	int l, r, k;

	/* not many points, all will be feasible */
	if (numfeasible == points->num) {
		for (int i = k1; i < k2; i++)
			(*feasible)[i] = i;
		return numfeasible;
	}

	accumweight = (float*) malloc(sizeof(float) * points->num);
	accumweight[0] = points->p[0].weight;
	totalweight = 0;
	for (int i = 1; i < points->num; i++) {
		accumweight[i] = accumweight[i - 1] + points->p[i].weight;
	}
	totalweight = accumweight[points->num - 1];

	for (int i = k1; i < k2; i++) {
		w = (lrand48() / (float) INT_MAX) * totalweight;
		//binary search
		l = 0;
		r = points->num - 1;
		if (accumweight[0] > w) {
			(*feasible)[i] = 0;
			continue;
		}
		while (l + 1 < r) {
			k = (l + r) / 2;
			if (accumweight[k] > w) {
				r = k;
			} else {
				l = k;
			}
		}
		(*feasible)[i] = r;
	}

	free(accumweight);

	return numfeasible;
}

/* compute approximate kmedian on the points */
float pkmedian(Points *points, long kmin, long kmax, long* kfinal, int pid,
		pthread_barrier_t* barrier) {
	int i;
	float cost;
	float lastcost;
	float hiz, loz, z;

	static long k;
	static int *feasible;
	static int numfeasible;
	static float* hizs;

	if (pid == 0)
		hizs = (float*) calloc(nproc, sizeof(float));
	hiz = loz = 0.0;
	long numberOfPoints = points->num;
	long ptDimension = points->dim;

	//my block
	long bsize = points->num / nproc;
	long k1 = bsize * pid;
	long k2 = k1 + bsize;
	if (pid == nproc - 1)
		k2 = points->num;

	float myhiz = 0;
	for (long kk = k1; kk < k2; kk++) {
		myhiz += dist(points->p[kk], points->p[0], ptDimension)
				* points->p[kk].weight;
	}
	hizs[pid] = myhiz;

	for (int i = 0; i < nproc; i++) {
		hiz += hizs[i];
	}

	loz = 0.0;
	z = (hiz + loz) / 2.0;
	/* NEW: Check whether more centers than points! */
	if (points->num <= kmax) {
		/* just return all points as facilities */
		for (long kk = k1; kk < k2; kk++) {
			points->p[kk].assign = kk;
			points->p[kk].cost = 0;
		}
		cost = 0;
		if (pid == 0) {
			free(hizs);
			*kfinal = k;
		}
		return cost;
	}

	if (pid == 0)
		shuffle(points);
	cost = pspeedy(points, z, &k, pid, barrier);

	i = 0;
	/* give speedy SP chances to get at least kmin/2 facilities */
	while ((k < kmin) && (i < SP)) {
		cost = pspeedy(points, z, &k, pid, barrier);
		i++;
	}

	/* if still not enough facilities, assume z is too high */
	while (k < kmin) {
		if (i >= SP) {
			hiz = z;
			z = (hiz + loz) / 2.0;
			i = 0;
		}
		if (pid == 0)
			shuffle(points);
		cost = pspeedy(points, z, &k, pid, barrier);
		i++;
	}

	/* now we begin the binary search for real */
	/* must designate some points as feasible centers */
	/* this creates more consistancy between FL runs */
	/* helps to guarantee correct # of centers at the end */

	if (pid == 0) {
		numfeasible = selectfeasible_fast(points, &feasible, kmin, pid,
				barrier);
		for (int i = 0; i < points->num; i++) {
			is_center[points->p[i].assign] = true;
		}
	}

	while (1) {

		/* first get a rough estimate on the FL solution */
		//    pthread_barrier_wait(barrier);
		lastcost = cost;
		cost = pFL(points, feasible, numfeasible, z, &k, kmax, cost,
				(long) (ITER * kmax * log((float) kmax)), 0.1, pid, barrier);

		/* if number of centers seems good, try a more accurate FL */
		if (((k <= (1.1) * kmax) && (k >= (0.9) * kmin))
				|| ((k <= kmax + 2) && (k >= kmin - 2))) {

			/* may need to run a little longer here before halting without
			 improvement */

			cost = pFL(points, feasible, numfeasible, z, &k, kmax, cost,
					(long) (ITER * kmax * log((float) kmax)), 0.001, pid,
					barrier);
		}

		if (k > kmax) {
			/* facilities too cheap */
			/* increase facility cost and up the cost accordingly */
			loz = z;
			z = (hiz + loz) / 2.0;
			cost += (z - loz) * k;
		}
		if (k < kmin) {
			/* facilities too expensive */
			/* decrease facility cost and reduce the cost accordingly */
			hiz = z;
			z = (hiz + loz) / 2.0;
			cost += (z - hiz) * k;
		}

		/* if k is good, return the result */
		/* if we're stuck, just give up and return what we have */
		if (((k <= kmax) && (k >= kmin)) || ((loz >= (0.999) * hiz))) {
			break;
		}
	}

	//clean up...
	if (pid == 0) {
		free(feasible);
		free(hizs);
		*kfinal = k;
	}

	return cost;
}

/* compute the means for the k clusters */
int contcenters(Points *points) {
	long i, ii;
	float relweight;

	for (i = 0; i < points->num; i++) {
		/* compute relative weight of this point to the cluster */
		if (points->p[i].assign != i) {
			relweight = points->p[points->p[i].assign].weight
					+ points->p[i].weight;
			relweight = points->p[i].weight / relweight;
			for (ii = 0; ii < points->dim; ii++) {
				points->p[points->p[i].assign].coord[ii] *= 1.0 - relweight;
				points->p[points->p[i].assign].coord[ii] +=
						points->p[i].coord[ii] * relweight;
			}
			points->p[points->p[i].assign].weight += points->p[i].weight;
		}
	}

	return 0;
}

/* copy centers from points to centers */
void copycenters(Points *points, Points* centers, long* centerIDs,
		long offset) {
	long i;
	long k;

	bool *is_a_median = (bool *) calloc(points->num, sizeof(bool));

	/* mark the centers */
	for (i = 0; i < points->num; i++) {
		is_a_median[points->p[i].assign] = 1;
	}

	k = centers->num;

	/* count how many  */
	for (i = 0; i < points->num; i++) {
		if (is_a_median[i]) {
			memcpy(centers->p[k].coord, points->p[i].coord,
					points->dim * sizeof(float));
			centers->p[k].weight = points->p[i].weight;
			centerIDs[k] = i + offset;
			k++;
		}
	}

	centers->num = k;

	free(is_a_median);
}

void* localSearchSub(void* arg_) {
	pkmedian_arg_t* arg = (pkmedian_arg_t*) arg_;
	pkmedian(arg->points, arg->kmin, arg->kmax, arg->kfinal, arg->pid,
			arg->barrier);

	return NULL;
}

void localSearch(Points* points, long kmin, long kmax, long* kfinal) {

	pthread_barrier_t barrier;
	pthread_t* threads = new pthread_t[nproc];
	pkmedian_arg_t* arg = new pkmedian_arg_t[nproc];

	for (int i = 0; i < nproc; i++) {
		arg[i].points = points;
		arg[i].kmin = kmin;
		arg[i].kmax = kmax;
		arg[i].pid = i;
		arg[i].kfinal = kfinal;

		arg[i].barrier = &barrier;
		localSearchSub(&arg[0]);
	}

	for (int i = 0; i < nproc; i++) {
	}

	delete[] threads;
	delete[] arg;
}

void outcenterIDs(Points* centers, long* centerIDs, char* outfile) {
	FILE* fp = fopen(outfile, "w");
	if (fp == NULL) {
		fprintf(stderr, "error opening %s\n", outfile);
		exit(1);
	}
	std::vector<int> is_a_median(centers->num, 0);
	for (int i = 0; i < centers->num; i++) {
		is_a_median[centers->p[i].assign] = 1;
	}

	for (int i = 0; i < centers->num; i++) {
		if (is_a_median[i]) {
			fprintf(fp, "%ld\n", centerIDs[i]);
			fprintf(fp, "%lf\n", centers->p[i].weight);
			for (int k = 0; k < centers->dim; k++) {
				fprintf(fp, "%lf ", centers->p[i].coord[k]);
			}
			fprintf(fp, "\n\n");
		}
	}
	fclose(fp);
}

void freePoints(Points& pts) {
	if (pts.p) {
		if (pts.p->coord)
			free(pts.p->coord);
		free(pts.p);
	}
}

std::tuple<Points, long*> streamCluster(PStream* stream, long kmin, long kmax,
		int dim, long chunksize, long centersize, char* outfile) {
	float* block = (float*) malloc(chunksize * dim * sizeof(float));
	float* centerBlock = (float*) malloc(centersize * dim * sizeof(float));
	long* centerIDs = (long*) malloc(centersize * dim * sizeof(long));

	if (block == NULL) {
		fprintf(stderr, "not enough memory for a chunk!\n");
		exit(1);
	}

	Points points;
	points.dim = dim;
	points.num = chunksize;
	points.p = (Point *) malloc(chunksize * sizeof(Point));
	for (int i = 0; i < chunksize; i++) {
		points.p[i].coord = &block[i * dim];
	}

	Points centers;
	centers.dim = dim;
	centers.p = (Point *) malloc(centersize * sizeof(Point));
	centers.num = 0;

	for (int i = 0; i < centersize; i++) {
		centers.p[i].coord = &centerBlock[i * dim];
		centers.p[i].weight = 1.0;
	}

	long IDoffset = 0;
	long kfinal;
	while (1) {

		size_t numRead = stream->read(block, dim, chunksize);
		fprintf(stderr, "read %lu points\n", numRead);

		if (stream->ferror()
				|| numRead < (unsigned int) chunksize && !stream->feof()) {
			fprintf(stderr, "error reading data!\n");
			exit(1);
		}

		points.num = numRead;
		for (int i = 0; i < points.num; i++) {
			points.p[i].weight = 1.0;
		}

		switch_membership = (bool*) malloc(points.num * sizeof(bool));
		is_center = (bool*) calloc(points.num, sizeof(bool));
		center_table = (int*) malloc(points.num * sizeof(int));

		localSearch(&points, kmin, kmax, &kfinal);

		fprintf(stderr, "finish local search\n");

		contcenters(&points);
		isCoordChanged = true;

		if (kfinal + centers.num > centersize) {
			//here we don't handle the situation where # of centers gets too large.
			fprintf(stderr, "oops! no more space for centers\n");
			exit(1);
		}

		copycenters(&points, &centers, centerIDs, IDoffset);
		IDoffset += numRead;

		free(is_center);
		free(switch_membership);
		free(center_table);

		if (stream->feof()) {
			break;
		}
	}

	//finally cluster all temp centers
	switch_membership = (bool*) malloc(centers.num * sizeof(bool));
	is_center = (bool*) calloc(centers.num, sizeof(bool));
	center_table = (int*) malloc(centers.num * sizeof(int));

	localSearch(&centers, kmin, kmax, &kfinal);
	contcenters(&centers);
//	outcenterIDs(&centers, centerIDs, outfile);
	if (block)
		free(block);

	freePoints(points);
	return {centers, centerIDs};
}

int main(int argc, char **argv) {
//	char *outfilename = new char[MAXNAMESIZE];
//	char *infilename = new char[MAXNAMESIZE];
	long kmin, kmax, n, chunksize, clustersize;
	int dim;
	printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
	if (argc < 10) {
		fprintf(stderr,
				"usage: %s k1 k2 d n chunksize clustersize infile outfile nproc\n",
				argv[0]);
		fprintf(stderr, "  k1:          Min. number of centers allowed\n");
		fprintf(stderr, "  k2:          Max. number of centers allowed\n");
		fprintf(stderr, "  d:           Dimension of each data point\n");
		fprintf(stderr, "  n:           Number of data points\n");
		fprintf(stderr,
				"  chunksize:   Number of data points to handle per step\n");
		fprintf(stderr,
				"  clustersize: Maximum number of intermediate centers\n");
		fprintf(stderr, "  infile:      Input file (if n<=0)\n");
		fprintf(stderr, "  outfile:     Output file\n");
		fprintf(stderr, "  nproc:       Number of threads to use\n");
		fprintf(stderr, "\n");
		fprintf(stderr,
				"if n > 0, points will be randomly generated instead of reading from infile.\n");
		exit(1);
	}
	kmin = atoi(argv[1]);
	kmax = atoi(argv[2]);
	dim = atoi(argv[3]);
	n = atoi(argv[4]);
	chunksize = atoi(argv[5]);
	clustersize = atoi(argv[6]);
	std::string infilename(argv[7]);
	std::string outfilename(argv[8]);
	nproc = atoi(argv[9]);

	srand48(SEED);
	PStream* stream;
	if (n > 0) {
		stream = new SimStream(n);
	} else {
		stream = new FileStream(infilename.c_str());
	}

	for (auto i = 0; i < 1000000; i++) {

		double t1 = rad::mysecond();

		serial_t = 0.0;
		cpu_to_gpu_t = 0.0;
		gpu_to_cpu_t = 0.0;
		alloc_t = 0.0;
		free_t = 0.0;
		kernel_t = 0.0;

		isCoordChanged = false;

		Points pts;

		long *centerIDs;
		std::tie(pts, centerIDs) = streamCluster(stream, kmin, kmax, dim,
				chunksize, clustersize, const_cast<char*>(outfilename.c_str()));

		double t2 = rad::mysecond();

		outcenterIDs(&pts, centerIDs, const_cast<char*>(outfilename.c_str()));

		if (centerIDs) {
			free(centerIDs);
		}

		freePoints(pts);

		if (switch_membership)
			free(switch_membership);	//whether to switch membership in pgain
		if (is_center)
			free(is_center);				//whether a point is a center
		if (center_table)
			free(center_table);			//index table of centers

		std::cout << "Iteration " << i << " time = " <<  t2 - t1 << std::endl;
	}
	delete stream;

	return 0;
}
