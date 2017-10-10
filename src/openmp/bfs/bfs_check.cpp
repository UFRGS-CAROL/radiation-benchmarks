#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
//#define NUM_THREAD 4
#define OPEN


#ifdef LOGS
#include "../../include/log_helper.h"
#endif

#ifdef TIMING
#include <sys/time.h>
long long timing_get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

long long setup_start, setup_end;
long long loop_start, loop_end;
long long kernel_start, kernel_end;
long long check_start, check_end;
#endif

FILE *fp;

//Structure to hold a node information
struct Node
{
    int starting;
    int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv) {

    fprintf(stderr,"Usage: %s <num_threads> <input_file> <gold file> <# iterations>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    BFSGraph( argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv)
{
#ifdef TIMING
    setup_start = timing_get_time();
#endif
    int no_of_nodes = 0;
    int edge_list_size = 0;
    char *input_f, *gold_f;
    int	 num_omp_threads, loop_iterations=1;

    if(argc!=5) {
        Usage(argc, argv);
        exit(0);
    }

    num_omp_threads = atoi(argv[1]);
    input_f = argv[2];
    gold_f = argv[3];
    loop_iterations = atoi(argv[4]);

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[200];
    snprintf(test_info, 200, "filename:%s threads:%d", input_f, num_omp_threads);
    start_log_file("openmpBFS", test_info);
#endif


    printf("Reading File\n");
    //Read in Graph from a file
    fp = fopen(input_f,"r");
    if(!fp)
    {
        printf("Error Reading graph file\n");
        return;
    }
    int source = 0;
    fscanf(fp,"%d",&no_of_nodes);
    // allocate host memory
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
    int start, edgeno;
    // initalize the memory
    for( unsigned int i = 0; i < no_of_nodes; i++)
    {
        fscanf(fp,"%d %d",&start,&edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i]=false;
        h_updating_graph_mask[i]=false;
        h_graph_visited[i]=false;
    }
    //read the source node from the file
    fscanf(fp,"%d",&source);
    // source=0; //tesing code line
    //set the source node as true in the mask
    h_graph_mask[source]=true;
    h_graph_visited[source]=true;
    fscanf(fp,"%d",&edge_list_size);
    int id,cost;
    int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    for(int i=0; i < edge_list_size ; i++)
    {
        fscanf(fp,"%d",&id);
        fscanf(fp,"%d",&cost);
        h_graph_edges[i] = id;
    }
    if(fp)
        fclose(fp);


    // read gold
    printf("Reading gold\n");
    int* h_cost_gold = (int*) malloc( sizeof(int)*no_of_nodes);
    FILE *fpo = fopen(gold_f,"rb");
    for(int i=0; i<no_of_nodes; i++) {
        fread(&h_cost_gold[i], sizeof(int), 1, fpo);
    }
    fclose(fpo);


    printf("allocating memory\n");
    // allocate mem for the result on host side
    int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);

    printf("Start traversing the tree\n");

#ifdef TIMING
    setup_end = timing_get_time();
#endif

    int loop;
    for(loop=0; loop<loop_iterations; loop++) {
#ifdef TIMING
        loop_start = timing_get_time();
#endif
        int k=0;
#ifdef TIMING
        kernel_start = timing_get_time();
#endif
#ifdef LOGS
        start_iteration();
#endif
        bool stop;
        do
        {
            //if no thread changes this value then the loop stops
            stop=false;

#ifdef OPEN
            //omp_set_num_threads(num_omp_threads);
            #pragma omp parallel for
#endif
            for(int tid = 0; tid < no_of_nodes; tid++ )
            {
                if (h_graph_mask[tid] == true) {
                    h_graph_mask[tid]=false;
                    for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++)
                    {
                        int id = h_graph_edges[i];
                        if(!h_graph_visited[id])
                        {
                            h_cost[id]=h_cost[tid]+1;
                            h_updating_graph_mask[id]=true;
                        }
                    }
                }
            }

#ifdef OPEN
            #pragma omp parallel for
#endif
            for(int tid=0; tid< no_of_nodes ; tid++ )
            {
                if (h_updating_graph_mask[tid] == true) {
                    h_graph_mask[tid]=true;
                    h_graph_visited[tid]=true;
                    stop=true;
                    h_updating_graph_mask[tid]=false;
                }
            }
            k++;
        }
        while(stop);
#ifdef LOGS
        end_iteration();
#endif
#ifdef TIMING
        kernel_end = timing_get_time();
#endif


#ifdef TIMING
        check_start = timing_get_time();
#endif

        // check output with gold
        int errors = 0;
        for(int i=0; i<no_of_nodes; i++) {
            if(h_cost[i] != h_cost_gold[i]) {
                errors++;
                char error_detail[200];
                sprintf(error_detail," p: [%d], r: %d, e: %d", i, h_cost[i], h_cost_gold[i]);
#ifdef LOGS
                log_error_detail(error_detail);
#endif
            }
        }
#ifdef LOGS
        log_error_count(errors);
#endif
#ifdef TIMING
        check_end = timing_get_time();
#endif
        if(errors > 0) {
            printf("Errors: %d\n",errors);
        } else {
            printf(".");
            fflush(stdout);
        }
        /******* Reload memory *********/
        fp = fopen(input_f,"r");
        if(!fp)
        {
            printf("Error Reading graph file\n");
            return;
        }
        fscanf(fp,"%d",&no_of_nodes);
        // initalize the memory
        for( unsigned int i = 0; i < no_of_nodes; i++)
        {
            //fscanf(fp,"%d %d",&start,&edgeno);
            //h_graph_nodes[i].starting = start;
            //h_graph_nodes[i].no_of_edges = edgeno;
            h_graph_mask[i]=false;
            h_updating_graph_mask[i]=false;
            h_graph_visited[i]=false;
        }
        //read the source node from the file
        //fscanf(fp,"%d",&source);
        //set the source node as true in the mask
        h_graph_mask[source]=true;
        h_graph_visited[source]=true;
        if(fp)
            fclose(fp);
        /******* Reload memory end *****/

#ifdef TIMING
        loop_end = timing_get_time();
        {
            double setup_timing = (double) (setup_end - setup_start) / 1000000;
            double loop_timing = (double) (loop_end - loop_start) / 1000000;
            double kernel_timing = (double) (kernel_end - kernel_start) / 1000000;
            double check_timing = (double) (check_end - check_start) / 1000000;
            printf("\n\tTIMING:\n");
            printf("setup: %f\n",setup_timing);
            printf("loop: %f\n",loop_timing);
            printf("kernel: %f\n",kernel_timing);
            printf("check: %f\n",check_timing);
        }
#endif
    }

#ifdef LOGS
    end_log_file();
#endif
    // cleanup memory
    free( h_graph_nodes);
    free( h_graph_edges);
    free( h_graph_mask);
    free( h_updating_graph_mask);
    free( h_graph_visited);
    free( h_cost);

}

