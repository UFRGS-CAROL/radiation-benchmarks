#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Params ---------------------------------------------------------------------
struct Params {

    const char *file_name;
	int 		n_frames;

    Params(int argc, char **argv) {

        file_name       = "input/peppa/";
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:t:w:r:a:f:c:xl:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'f': file_name       = optarg; break;
            case 'r': n_frames        = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }

    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./cedd [options]"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    folder containing input video files (default=input/peppa/)"
                "\n");
    }
};

void read_input(unsigned char** all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {


	int erro = 0;
        char FileName[100];
        sprintf(FileName, "%s%d.txt", p.file_name, 5); // Escolhe o  5to Frame

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

        FILE *fpo = fopen("Novo_Frame", "w");
        if(fpo == NULL)
            exit(EXIT_FAILURE);

        fscanf(fp, "%d\n", &rowsc);
        fscanf(fp, "%d\n", &colsc);

	// Coloca no novo frame #linhas e # colunas
	fprintf(fpo,"%d\n",rowsc);
	fprintf(fpo,"%d\n",colsc);

        in_size = rowsc * colsc * sizeof(unsigned char);
        all_gray_frames[0]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
				// Leitura do Pixel do Frame
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[0][i * colsc + j]);
				if(i==10 && j ==50){ // Dentro do quadrado
					printf("Injetando Erro no quadrado central\n");
					printf("a %d\n",colsc);
					fprintf(fpo,"%u ",erro);				
				}
				else{
					fprintf(fpo,"%u ",all_gray_frames[0][i * colsc + j]);
				}
				// Testar se esta dentro do quadrado interior e colocar um numero 0
            }
        }
        fclose(fp);
		fclose(fpo);
}

int main(int argc, char **argv){

    Params      p(argc, argv);
    const int n_frames =p.n_frames;
    unsigned char **all_gray_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);
	
}
