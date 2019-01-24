#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#ifdef LOGS
#include "../../include/log_helper.h"
#endif /* LOGS */

#define MAX_ERR_ITER_LOG 500

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
#define OPEN
//#define NUM_THREAD 4

#ifdef TIMING
inline long long timing_get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

long long setup_start, setup_end;
long long loop_start, loop_end;
long long kernel_start, kernel_end;
long long check_start, check_end;
long long int flops=0;
#endif

typedef float FLOAT;

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

int num_omp_threads;


/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */
void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row, int col,
                      FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1,
                      FLOAT step)
{
    FLOAT delta;
    int r, c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;

#ifdef OPEN
    omp_set_num_threads(num_omp_threads);
    #pragma omp parallel for shared(power, temp, result) private(chunk, r, c, delta) firstprivate(row, col, num_chunk, chunks_in_row) schedule(static)
#endif
    for ( chunk = 0; chunk < num_chunk; ++chunk )
    {
        int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row);
        int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;

        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
        {
            for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
                for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
                    /* Corner 1 */
                    if ( (r == 0) && (c == 0) ) {
                        delta = (Cap_1) * (power[0] +
                                           (temp[1] - temp[0]) * Rx_1 +
                                           (temp[col] - temp[0]) * Ry_1 +
                                           (amb_temp - temp[0]) * Rz_1);
                    }	/* Corner 2 */
                    else if ((r == 0) && (c == col-1)) {
                        delta = (Cap_1) * (power[c] +
                                           (temp[c-1] - temp[c]) * Rx_1 +
                                           (temp[c+col] - temp[c]) * Ry_1 +
                                           (   amb_temp - temp[c]) * Rz_1);
                    }	/* Corner 3 */
                    else if ((r == row-1) && (c == col-1)) {
                        delta = (Cap_1) * (power[r*col+c] +
                                           (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 +
                                           (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
                                           (   amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Corner 4	*/
                    else if ((r == row-1) && (c == 0)) {
                        delta = (Cap_1) * (power[r*col] +
                                           (temp[r*col+1] - temp[r*col]) * Rx_1 +
                                           (temp[(r-1)*col] - temp[r*col]) * Ry_1 +
                                           (amb_temp - temp[r*col]) * Rz_1);
                    }	/* Edge 1 */
                    else if (r == 0) {
                        delta = (Cap_1) * (power[c] +
                                           (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 +
                                           (temp[col+c] - temp[c]) * Ry_1 +
                                           (amb_temp - temp[c]) * Rz_1);
                    }	/* Edge 2 */
                    else if (c == col-1) {
                        delta = (Cap_1) * (power[r*col+c] +
                                           (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 +
                                           (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 +
                                           (amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Edge 3 */
                    else if (r == row-1) {
                        delta = (Cap_1) * (power[r*col+c] +
                                           (temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 +
                                           (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
                                           (amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Edge 4 */
                    else if (c == 0) {
                        delta = (Cap_1) * (power[r*col] +
                                           (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 +
                                           (temp[r*col+1] - temp[r*col]) * Rx_1 +
                                           (amb_temp - temp[r*col]) * Rz_1);
                    }
                    result[r*col+c] =temp[r*col+c]+ delta;
                }
            }
            continue;
        }

        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
            #pragma omp simd
            for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
                /* Update Temperatures */
                result[r*col+c] =temp[r*col+c]+
                                 ( Cap_1 * (power[r*col+c] +
                                            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.f*temp[r*col+c]) * Ry_1 +
                                            (temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c]) * Rx_1 +
                                            (amb_temp - temp[r*col+c]) * Rz_1));
            }
        }
    }
}

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int row, int col)
{

    FLOAT grid_height = chip_height / row;
    FLOAT grid_width = chip_width / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1=1.f/Rx;
    FLOAT Ry_1=1.f/Ry;
    FLOAT Rz_1=1.f/Rz;
    FLOAT Cap_1 = step/Cap;

    FLOAT* r = result;
    FLOAT* t = temp;
    int i = 0;
#ifdef TIMING
    kernel_start = timing_get_time();
#endif
#ifdef LOGS
    start_iteration();
#endif /* LOGS */
    for (i = 0; i < num_iterations ; i++)
    {
        single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        FLOAT* tmp = t;
        t = r;
        r = tmp;
    }
#ifdef LOGS
    end_iteration();
#endif /* LOGS */
#ifdef TIMING
    kernel_end = timing_get_time();
#endif
}

void fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}


void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file)
{
    int i, index;
    FILE *fp;
    char str[STR_SIZE];
    FLOAT val;

    fp = fopen (file, "r");
    if (!fp)
        fatal ("file could not be opened for reading");

    for (i=0; i < grid_rows * grid_cols; i++) {
        if (fgets(str, STR_SIZE, fp) == NULL) {
            fatal("fgets error");
        }
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1) )
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file> <output_file> <# iterations>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char **argv)
{
#ifdef TIMING
    setup_start = timing_get_time();
#endif
    int grid_rows, grid_cols, sim_time, i;
    FLOAT *temp, *power, *result, *final_result, *gold;
    char *tfile, *pfile, *ofile;
    int tot_iterations = 1;

    /* check validity of inputs	*/
    if (argc != 9)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 ||
            (grid_cols = atoi(argv[2])) <= 0 ||
            (sim_time = atoi(argv[3])) <= 0 ||
            (num_omp_threads = atoi(argv[4])) <= 0 ||
            (tot_iterations = atoi(argv[8])) <= 0
       )
        usage(argc, argv);

    /* allocate memory for the temperature and power arrays	*/
    temp = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
    power = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
    gold = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
    if(!temp || !power)
        fatal("unable to allocate memory");

#ifdef LOGS
    char test_info[100];
    snprintf(test_info, 100, "simIter:%d gridSize:%dx%d",sim_time, grid_rows, grid_cols);
    start_log_file((char *)"openMPHotspot", test_info);
    set_max_errors_iter(MAX_ERR_ITER_LOG);
    set_iter_interval_print(10);
#endif /* LOGS */

    /* read initial temperatures and input power	*/
    tfile = argv[5];
    pfile = argv[6];
    ofile = argv[7];

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);
    read_input(gold, grid_rows, grid_cols, ofile);

#ifdef TIMING
    setup_end = timing_get_time();
#endif
    printf("Starting hotspot loop\n");
    int loop1;
    for (loop1=0; loop1 < tot_iterations; loop1++)
    {
#ifdef TIMING
        loop_start = timing_get_time();
#endif
#ifdef ERR_INJ
        if(loop1 == 2) {
            printf("injecting error, changing input!\n");
            temp[0] =  temp[0]*2;
            power[0] = power[0]*5;
        } else if (loop1 == 3) {
            printf("get ready, infinite loop...\n");
            fflush(stdout);
            while(1) {
                sleep(1000);
            }
        }
#endif


        compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);


        final_result = (1&sim_time) ? result : temp;
        int errors = 0;
#ifdef TIMING
        check_start = timing_get_time();
#endif
        #pragma omp parallel for reduction(+:errors)
        for (i=0; i < grid_rows; i++) {
            int j;
            for (j=0; j < grid_cols; j++) {
                if ((fabs((final_result[i*grid_cols+j] - gold[i*grid_cols+j]) / final_result[i*grid_cols+j]) > 0.0000000001) || (fabs((final_result[i*grid_cols+j] - gold[i*grid_cols+j]) / gold[i*grid_cols+j]) > 0.0000000001)) {
                //if(final_result[i*grid_cols+j] != gold[i*grid_cols+j] ) {
                    errors++;
                }
            }
        }
#ifdef TIMING
        check_end = timing_get_time();
#endif

        if (errors!=0)
        {
            printf("kernel errors: %d\n", errors);
#ifdef LOGS
            int err_loged=0;
            for (i=0; i < grid_rows && err_loged < MAX_ERR_ITER_LOG && err_loged < errors; i++) {
                int j;
                for (j=0; j < grid_cols && err_loged < MAX_ERR_ITER_LOG && err_loged < errors; j++) {
                    if ((fabs((final_result[i*grid_cols+j] - gold[i*grid_cols+j]) / final_result[i*grid_cols+j]) > 0.0000000001) || (fabs((final_result[i*grid_cols+j] - gold[i*grid_cols+j]) / gold[i*grid_cols+j]) > 0.0000000001)) {
                    //if(final_result[i*grid_cols+j] != gold[i*grid_cols+j] ) {
                        err_loged++;
                        char error_detail[150];
                        snprintf(error_detail, 150, "r:%f e:%f [%d,%d]\n", final_result[i*grid_cols+j], gold[i*grid_cols+j], i, j);
                        log_error_detail(error_detail);
                    }
                }
            }
            log_error_count(errors);
#endif /* LOGS */
            read_input(power, grid_rows, grid_cols, pfile);
            read_input(gold, grid_rows, grid_cols, ofile);
        }
        else
        {
            printf(".");
        }
        fflush(stdout);
        read_input(temp, grid_rows, grid_cols, tfile);
#ifdef TIMING
        loop_end = timing_get_time();
        double setup_timing = (double) (setup_end - setup_start) / 1000000;
        double loop_timing = (double) (loop_end - loop_start) / 1000000;
        double kernel_timing = (double) (kernel_end - kernel_start) / 1000000;
        double check_timing = (double) (check_end - check_start) / 1000000;
        printf("\n\tTIMING:\n");
        printf("setup: %f\n",setup_timing);
        printf("loop: %f\n",loop_timing);
        printf("kernel: %f\n",kernel_timing);
        printf("check: %f\n",check_timing);
#endif
    }

#ifdef LOGS
    end_log_file();
#endif /* LOGS */
    /* cleanup	*/
    free(temp);
    free(power);

    return 0;
}
