#ifndef SELECTIVE_HARDENING_H
#define SELECTIVE_HARDENING_H

int throw_error_int(int var1, int var2, const char* variable_name, const char* executable_name);

#define SELECTIVE_HARDENING_INT(a,b,var,exec_name) (((a ^ b) == 0) ? (a) : (throw_error_int(a,b,var,exec_name)))

#endif
