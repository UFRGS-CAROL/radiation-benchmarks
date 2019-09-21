#ifndef SELECTIVE_HARDENING_H
#define SELECTIVE_HARDENING_H

int throw_error_int(int var1, int var2, const char* variable_name);
void write_got_sdc(int got_sdc);

#define SELECTIVE_HARDENING_INT(a,b,var) (((a ^ b) == 0) ? (a) : (throw_error_int(a,b,var)))

#endif
