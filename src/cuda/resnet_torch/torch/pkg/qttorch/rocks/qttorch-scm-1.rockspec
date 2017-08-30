package = "qttorch"
version = "scm-1"

source = {
   url = "git://github.com/torch/qttorch.git",
}

description = {
   summary = "QT interface to Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/qttorch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "qtlua >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
