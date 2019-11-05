
#include "common.h"


void exception(std::string msg, std::string file, int line) {
	throw std::runtime_error(msg + " at " + file + ":" + std::to_string(line));
}
