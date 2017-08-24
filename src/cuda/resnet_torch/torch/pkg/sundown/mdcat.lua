#!/usr/bin/env lua

local ascii = require 'sundown.ascii'
assert(#arg == 1, 'usage: mdcat <file.md>')
print(ascii.render(io.open(arg[1]):read('*all')))
