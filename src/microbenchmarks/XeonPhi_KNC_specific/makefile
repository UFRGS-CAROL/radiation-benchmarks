# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
RM = rm -f
CACHE = cache/

SCALAR_AND = scalar/and/
SCALAR_OR = scalar/or/
SCALAR_REG = scalar/reg/
SCALAR_SLR = scalar/slr/
SCALAR_ROL = scalar/rol/

VECTOR_AND = vector/and/
VECTOR_OR = vector/or/
VECTOR_REG = vector/reg/
VECTOR_SLR = vector/slr/


MY_TARGETS = $(CACHE) \
			$(SCALAR_AND) $(SCALAR_OR) $(SCALAR_REG) $(SCALAR_SLR) $(SCALAR_ROL) \
			$(VECTOR_AND) $(VECTOR_OR) $(VECTOR_REG) $(VECTOR_SLR)

.PHONY: all $(MY_TARGETS)

all: $(MY_TARGETS)

$(MY_TARGETS):
			@$(MAKE) -B -C $@;

