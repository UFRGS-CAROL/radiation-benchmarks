#include <stdio.h>
#include <stdlib.h>
#include "GoldGenerator.h"

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
