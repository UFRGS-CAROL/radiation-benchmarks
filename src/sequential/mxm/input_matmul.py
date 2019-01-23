#!/usr/bin/python

import math
import sys
from random import uniform

if len(sys.argv) < 2:
    print("USAGE: {} <input> <size>".format(sys.argv[0]))
    sys.exit(1)

with open(sys.argv[1], "w") as fp:
    for i in range(int(sys.argv[2])):
        a = uniform(1.0, i * math.pi + 4.966228736338716478)
        b = uniform(1.0, i / math.pi + 2.726218736218716238)
        fp.write("{} {}".format(a, b))

