//new classes
#include "App.h"
//for radiation test verification
#ifdef LOGS
#include "log_helper.h"
#endif
#include <sys/time.h>

/**
 Mysecond function
 for time verification
 */

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

int main(int argc, char** argv) {
	try {
		Args args;
		if (argc < 2)
			args.printHelp();
		args = Args::read(argc, argv);
		if (args.help_showed)
			return -1;
		App app(args);
		app.run();
	} catch (const Exception& e) {
		return cout << "error: " << e.what() << endl, 1;
	} catch (const exception& e) {
		return cout << "error: " << e.what() << endl, 1;
	} catch (...) {
		return cout << "unknown exception" << endl, 1;
	}
	return 0;
}
