
#include <string>

void BFSGraph(std::string input_f, std::string output_f);

void Usage(int argc, char**argv) {

	fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);

}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	if (argc != 2) {
		Usage(argc, argv);
		exit(0);
	}

	BFSGraph(std::string(argv[1]), "result.txt");
}
