//new classes
#include "App.h"
//for radiation test verification


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
	} catch (std::exception& e) {
		return cout << "error: " << e.what() << endl, 1;
	} catch (...) {
		return cout << "unknown exception" << endl, 1;
	}
	return 0;
}
