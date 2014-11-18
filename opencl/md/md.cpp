#include "md.h"


using namespace std;

// Forward Declarations
inline double distance(const double4* position, const int i, const int j);

inline void insertInOrder(std::list<double>& currDist, std::list<int>& currList,
        const int j, const double distIJ, const int maxNeighbors);

inline int buildNeighborList(const int nAtom, const double4* position,
        int* neighborList);

inline int populateNeighborList(std::list<double>& currDist,
        std::list<int>& currList, const int j, const int nAtom,
        int* neighborList);


bool checkResults(double4* d_force, double4 *position,
                  int *neighList, int nAtom)
{
    for (int i = 0; i < nAtom; i++)
    {
        double4 ipos = position[i];
        double4 f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j*nAtom + i];
            double4 jpos = position[jidx];
            // Calculate distance
            double delx = ipos.x - jpos.x;
            double dely = ipos.y - jpos.y;
            double delz = ipos.z - jpos.z;
            double r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {

                r2inv = 1.0f/r2inv;
                double r6inv = r2inv * r2inv * r2inv;
                double force = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * force;
                f.y += dely * force;
                f.z += delz * force;
            }
            j++;
        }
        // Check the results
        double diffx = (d_force[i].x - f.x) / d_force[i].x;
        double diffy = (d_force[i].y - f.y) / d_force[i].y;
        double diffz = (d_force[i].z - f.z) / d_force[i].z;
        double err = sqrt(diffx*diffx) + sqrt(diffy*diffy) + sqrt(diffz*diffz);
        if (err > (3.0 * EPSILON))
        {
            cout << "Test Failed, idx: " << i << " diff: " << err << "\n";
            cout << "f.x: " << f.x << " df.x: " << d_force[i].x << "\n";
            cout << "f.y: " << f.y << " df.y: " << d_force[i].y << "\n";
            cout << "f.z: " << f.z << " df.z: " << d_force[i].z << "\n";
            cout << "Test FAILED\n";
            return false;
        }
    }
    cout << "Test Passed\n";
    return true;
}


extern const char *cl_source_md;


int main() {

    // Problem Parameters
    const int probSizes[4] = { 12288, 24576, 36864, 73728 };
    int sizeClass = 1;
    int nAtom = probSizes[sizeClass - 1];

	cout << "#Atons = " << nAtom << "\n";
    double4 *position;
    double4 *force;
    int *neighborList;

    position = (double4*)malloc(sizeof(double4)*nAtom);
    force = (double4*)malloc(sizeof(double4)*nAtom);
    neighborList = (int*)malloc(sizeof(int) * nAtom * maxNeighbors);

	initOpenCL();

	ocl_alloc_buffers(nAtom, maxNeighbors);

    size_t localSize  = 128;
    size_t globalSize = nAtom;

    cout << "Initializing test problem (this can take several "
            "minutes for large problems).\n                   ";

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (drand48() * domainEdge);
        position[i].y = (drand48() * domainEdge);
        position[i].z = (drand48() * domainEdge);
    }

    ocl_write_position_buffer(nAtom, position);

	int totalPairs = buildNeighborList(nAtom, position,
            neighborList);

    cout << "Finished.\n";
    cout << totalPairs << " of " << nAtom*maxNeighbors <<
            " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %" << endl;

	
	ocl_write_neighborList_buffer(maxNeighbors, nAtom, neighborList);
 
	ocl_set_kernel_args(maxNeighbors, nAtom);

	ocl_exec_kernel(globalSize, localSize);

	ocl_read_force_buffer(nAtom, force);
    
    cout << "Performing Correctness Check (can take several minutes)\n";

    // If results are correct, skip the performance tests
    if (!checkResults(force, position, neighborList, nAtom)) {
		cout << "Results does not check\n";
        return 1;
    }

	cout << "Results check\n";

    ocl_release_buffers();
    deinitOpenCL();

}


inline double distance(const double4* position, const int i, const int j)
{
    double4 ipos = position[i];
    double4 jpos = position[j];
    double delx = ipos.x - jpos.x;
    double dely = ipos.y - jpos.y;
    double delz = ipos.z - jpos.z;
    double r2inv = delx * delx + dely * dely + delz * delz;
    return r2inv;
}


inline void insertInOrder(list<double>& currDist, list<int>& currList,
        const int j, const double distIJ, const int maxNeighbors)
{

    typename list<double>::iterator   it;
    typename list<int>::iterator it2;

    it2 = currList.begin();

    double currMax = currDist.back();

    if (distIJ > currMax) return;

    for (it=currDist.begin(); it!=currDist.end(); it++)
    {
        if (distIJ < (*it))
        {
            // Insert into appropriate place in list
            currDist.insert(it,distIJ);
            currList.insert(it2, j);

            // Trim end of list
            currList.resize(maxNeighbors);
            currDist.resize(maxNeighbors);
            return;
        }
        it2++;
    }
}



inline int buildNeighborList(const int nAtom, const double4* position,
        int* neighborList)
{
    int totalPairs = 0;
    // Build Neighbor List
    // Find the nearest N atoms to each other atom, where N = maxNeighbors
    for (int i = 0; i < nAtom; i++)
    {
        // Current neighbor list for atom i, initialized to -1
        list<int>   currList(maxNeighbors, -1);
        // Distance to those neighbors.  We're populating this with the
        // closest neighbors, so initialize to FLT_MAX
        list<double> currDist(maxNeighbors, FLT_MAX);

        for (int j = 0; j < nAtom; j++)
        {
            if (i == j) continue; // An atom cannot be its own neighbor

            // Calculate distance and insert in order into the current lists
            double distIJ = distance(position, i, j);
            insertInOrder(currDist, currList, j, distIJ, maxNeighbors);
        }
        // We should now have the closest maxNeighbors neighbors and their
        // distances to atom i. Populate the neighbor list data structure
        // for GPU coalesced reads.
        // The populate method returns how many of the maxNeighbors closest
        // neighbors are within the cutoff distance.  This will be used to
        // calculate GFLOPS later.
        totalPairs += populateNeighborList(currDist, currList, i, nAtom,
                neighborList);
    }
    return totalPairs;
}


inline int populateNeighborList(list<double>& currDist,
        list<int>& currList, const int i, const int nAtom,
        int* neighborList)
{
    int idx = 0;
    int validPairs = 0; // Pairs of atoms closer together than the cutoff

    // Iterate across distance and neighbor list
    typename list<double>::iterator distanceIter = currDist.begin();
    for (list<int>::iterator neighborIter = currList.begin();
            neighborIter != currList.end(); neighborIter++)
    {
        // Populate packed neighbor list
        neighborList[(idx * nAtom) + i] = *neighborIter;

        // If the distance is less than cutoff, increment valid counter
        if (*distanceIter < cutsq)
            validPairs++;

        // Increment idx and distance iterator
        idx++;
        distanceIter++;
    }
    return validPairs;
}
