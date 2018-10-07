#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct {
	char *key;
	char *val;
	int used;
} kvp;

int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
real_t option_find_real_t(list *l, char *key, real_t def);
real_t option_find_real_t_quiet(list *l, char *key, real_t def);
void option_unused(list *l);

#endif
