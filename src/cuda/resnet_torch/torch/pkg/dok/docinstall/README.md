<a name="install.dok"/>
# Torch Installation Manual #

Currently Torch7 installation can be done only from the
sources. Binary releaseses will be distributed soon.

<a name="install.sources"/>
# Installing from sources #

`Torch7` is mainly made out of `ANSI C` and `Lua`, which makes
it easy to compile everywhere. The graphical interface is based on QT
and requires a `C++` compiler.

The installation process became easily portable on most platforms,
thanks to [CMake](http://www.cmake.org), a tool which replace the
aging `configure/automake` tools. CMake allows us to detect and
configure Torch properly.

You will find here step-by-step instructions for each system we are supporting.

You are also strongly encouraged to read the [CMake hints](#CMakeHints)
section for more details on CMake (and before reporting a problem).

If you are a programmer, you might want to produce your own
[development package](#DevPackages).

<a name="install.linux"/>
## Linux ##

### A. Requirements ###

Torch compilation requires a number of standard packages described below:
  * __Mandatory:__
    * A `C/C++` compiler. [CLang](http:_clang.llvm.org) is great. The [GNU compiler](http:_gcc.gnu.org) or Intel compiler work fine.
    * [CMake](http://www.cmake.org) version 2.6 or later is required.
    * [Gnuplot](http://gnuplot.info), version `4.4` or later is recommended for best experience.

  * __Recommended:__
    * [GNU Readline](http://tiswww.case.edu/php/chet/readline/rltop.html)
    * [Git](http://git-scm.com/) to keep up-to-date sources
    * [QT 4.4](http://trolltech.com/products) or newer development libraries
    * BLAS. [OpenBLAS](https://github.com/xianyi/OpenBLAS) is recommended for that purpose on Intel computers.
    * LAPACK. [OpenBLAS](https://github.com/xianyi/OpenBLAS) is recommended for that purpose on Intel computers.

The installation of most of these packages should be rather
straightforward. For `Ubuntu 10.04 LTS` system we use the
`apt-get` magic:

For GCC:
```
sudo apt-get install gcc g++
```
If you prefer to use CLang:
```
sudo apt-get install clang
```

CMake reads CC and CXX variables. If you do not want to use the default compiler, just do
```
export CC=clang
export CXX=clang++
```

To install the additional packages, do:
```
sudo apt-get install cmake
sudo apt-get install libreadline5-dev
sudo apt-get install git-core
sudo apt-get install gnuplot
```

Please adapt according to your distribution.

Note: readline library is helpful for better command line interaction,
but it is not required. It is only used when QT is installed.

We require `QT 4.4` for handling graphics (_beware_ not installing QT 4.3
or older). If it is not found at compile time, Torch will still compile but
no graphics will be available. On `Ubuntu 10.04 LTS` distribution you can
install it with
```
sudo apt-get install libqt4-core libqt4-gui libqt4-dev
```

An excellent BLAS/LAPACK implementation is also recommended for speed. See
our [BLAS recommendations](blas).

<a name="install.sources"/>
### B. Getting Torch sources ###

Torch7 is being developed on [github](http://github.com).

```
git clone git://github.com/andresy/torch.git
```


<a name="install.config"/>
### C. Configuring Torch ###

We use `CMake` for configuring `Torch`. We _highly_ recommend to create
first a dedicated build directory. This eases cleaning up built objects,
but also allow you to build Torch with _various configurations_
(e.g. Release and Debug in two different build directories).

```
cd torch
mkdir build
cd build
cmake ..
```

The `..` given to `cmake` indicates the directory where the
sources are. We chose here to have a `build` directory inside
`torch`, but it could be anywhere else. In that latter case, go
instead in your build directory and then do: 

```
cmake /path/to/torch/sources
```

CMake detects external libraries or tools necessary for Torch, and
produces Makefiles such that Torch is then easily compilable on your
platform. If you prefer the GUI version of CMake, you can replace
`cmake` by `ccmake` in the above command lines. In particular, it
is _strongly encouraged_ to use `ccmake` for finer configuration
of Torch.

The most common Torch configuration step you might want to perform is
changing the installation path. By default, Torch will be installed in
`/usr/local`. You will need super-user rights to perform that. If
you are not root on your computer, you can instead specifying a
install directory to `CMake` on the above `cmake` command:

```
cmake .. -DCMAKE_INSTALL_PREFIX=/my/install/path
```

Equivalently you can set the variable `CMAKE_INSTALL_PREFIX` if you
use `ccmake` GUI.  Please, see [[http://www.cmake.org|CMake
documentation]] or _at least_ [[#CMakeHints|some of our CMake
hints]] for more details on configuration.

<a name="install.compile"/>
### D. Compiling and installing ###

If the configuration was successful, Makefiles should have appeared in
your build directory.  Compile Torch with:

then compile and install with:
```
make install
```

This last command might possibly be prefixed by `sudo` if you are
installing Torch in `/usr/local`.

<a name="install.run"/>
### E. Running Torch ###

Now Torch should be installed in `/usr/local` or in
`/my/install/path` if you chose to use the `CMAKE_INSTALL_PREFIX`
when configuring with CMake.  Lua executables (`torch-lua`, 
`torch-qlua` and `torch`) are found in the `bin` sub-directory of 
these installation directories.

```
/usr/local/bin/torch-lua
Lua 5.1.4  Copyright (C) 1994-2008 Lua.org, PUC-Rio
> require 'torch'
> = torch.Tensor(5):zero()

0
0
0
0
0
[torch.Tensor of dimension 5]

> 
```

For convenience, you might want to add to your `PATH` the path to
lua binaries. The executable `torch-lua` is a simple Lua interpreter 
(as provided on [Lua website](http://www.lua.org)), while `torch-qlua` 
has enhanced interactivity (like completion) and is able to handle
graphics and QT widgets.

For best experience we suggest using the `torch` executable, which
preloads the most commonly used libraries into the global namespace.

```
/usr/local/bin/torch
Try the IDE: torch -ide
Type help() for more info
Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU
Lua 5.1  Copyright (C) 1994-2008 Lua.org, PUC-Rio
torch> =torch.randn(10,10)
 1.3862  1.5983 -2.0216 -0.1502  1.9467 -1.2322  0.1628 -2.6253  1.3255 -0.5784
 0.1363 -1.2638 -1.0661  0.0233  1.3064 -0.8817  1.1424  1.0952 -0.2147  0.7712
 1.1348 -0.8596 -0.6102  0.9137 -1.1582 -0.3301  0.5250  1.3631 -0.4051 -0.9549
-0.2734 -0.0914  0.9728  1.3272 -0.4126 -0.1264 -1.2936 -0.2120  1.3040 -1.9991
-0.9642  0.2367 -0.5246 -0.0476 -0.6586  1.8705  0.8482 -1.2768 -0.0782  0.5403
 0.4551 -1.4549 -0.7079 -1.6308 -0.2086 -1.7208 -1.4915  0.9703  0.3661  0.5051
 0.3082  0.3188 -1.1247  0.1343 -0.2671 -0.4596 -0.2618  1.7482  0.4714  0.5217
-0.8406 -0.2372 -0.1504  0.6982 -0.5437  0.7447  0.0229 -2.4998  0.7367 -1.2721
-0.3993  1.5155 -0.3685 -0.0534 -0.0495 -0.1002 -0.3652  0.1248 -0.2693  0.9159
-1.5035  0.7326 -0.6262  0.2715  0.0543 -0.7419 -0.6758 -0.0221  0.5342 -0.4262
[torch.DoubleTensor of dimension 10x10]

torch> 

```

You can get more help about `torch`:

```
/usr/local/bin/torch -h
Torch7 Shell

Usage: torch [options] [script [args]]

General options:
  -b|-bare         start a bare environment (no libs preloaded)
  -e string        execute string
  -l lib           require lib
  -i               enter interactive mode after executing script [false]
  -v|-version      show version information [false]
  -h|-help         this help [false]

Qt options:
  -nographics|-ng  disable all the graphical capabilities [false]
  -ide             enable IDE (graphical console) [false]
  -onethread       run lua in the main thread (might be safer) [false] 
```

## MacOS X ##

### A. Requirements ###

Torch compilation requires a number of standard packages described below:
  * __Mandatory:__
    * A `C/C++` compiler. [CLang](http:_clang.llvm.org) is great. The [GNU compiler](http:_gcc.gnu.org) or Intel compiler work fine.
    * [CMake](http://www.cmake.org) version 2.6 or later is required.
    * [Gnuplot](http://gnuplot.info), version `4.4` or later is recommended for best experience.

  * __Recommended:__
    * [GNU Readline](http://tiswww.case.edu/php/chet/readline/rltop.html)
    * [Git](http://git-scm.com/) to keep up-to-date sources
    * [QT 4.4](http://trolltech.com/products) or newer development libraries
    * BLAS. [OpenBLAS](https://github.com/xianyi/OpenBLAS) is recommended for that purpose on Intel computers.
    * LAPACK. [OpenBLAS](https://github.com/xianyi/OpenBLAS) is recommended for that purpose on Intel computers.

Installation of gcc should be done by installing the
[[http://developer.apple.com/tools/xcode|the Apple developer
tools]]. These tools should also be available on you MacOS X
installation DVD.

CMake can be retrieved from
[CMake website](http://www.cmake.org/HTML/Download.html) (you can
take the __DMG__ installer). However, we found it was as simple to use
[Homebrew](http:_mxcl.github.com/homebrew/), or [MacPorts](http:_www.macports.org/)
which are necessary anyway for git and the Readline library. We recommend to avoid
[Fink](http://finkproject.org/), which tends to be always
outdated. Assuming you installed Homebrew, just do:

```
brew install readline
brew install cmake
brew install git
brew install gnuplot
```

For installing QT, one can use Homebrew, but it might take too long to
compile.  Instead, you can
[download](http://trolltech.com/downloads/opensource/appdev/mac-os-cpp)
the binary __DMG__ file available on [[http://trolltech.com|Trolltech
website]] and install it.

An excellent BLAS/LAPACK implementation is also recommended for speed. See
our [BLAS recommendations](blas).

Last but not least, GCC >= 4.6 is *required* to enable OpenMP on MacOS X. This
is a bit crazy, but compiling against OpenMP with previous versions of GCC
will give you random segfaults and trap errors (a known issue on the web).
We strongly recommend you to install GCC 4.6, to fully benefit from Torch's
fast numeric routines. A very simple way of doing so is to install the 
[GFortran](http://gcc.gnu.org/wiki/GFortranBinaries) libraries, which are
packaged as a simple dmg, ready to install. That'll automatically install gcc
and g++. Once this is done, set your CC and CXX before building Torch:

```
export CC=/usr/local/gfortran/bin/gcc
export CXX=/usr/local/gfortran/bin/g++
```

### B. Getting Torch sources ###

Same as [getting sources](#install.sources) for linux.

### C. Configuring Torch ###

Same as [configuring](#install.config) for linux.

### D. Compiling and Installing ###

Same as [compiling](#install.compile) for linux.

### E. Running Torch ###

Same as [runnning](#install.run) for linux.

<a name="install.freebsd"/>
## FreeBSD ##

### A. Requirements ###

Torch compilation requires a number of standard packages described below:
  * __Mandatory:__
    * A `C/C++` compiler. [CLang](http:_clang.llvm.org) is great. The [GNU compiler](http:_gcc.gnu.org) or Intel compiler work fine.
    * [CMake](http://www.cmake.org) version 2.6 or later is required.
    * [Gnuplot](http://gnuplot.info), version `4.4` or later is recommended for best experience.

  * __Recommended:__
    * [GNU Readline](http://tiswww.case.edu/php/chet/readline/rltop.html)
    * [Git](http://git-scm.com/) to keep up-to-date sources
    * [QT 4.4](http://trolltech.com/products) or newer development libraries
    * BLAS. [OpenBLAS](https://github.com/xianyi/OpenBLAS) is recommended for that purpose on Intel computers.
    * LAPACK. [OpenBLAS](https://github.com/xianyi/OpenBLAS) is recommended for that purpose on Intel computers.

GCC and CLang come with FreeBSD install. However, only GCC 4.2 is installed by default (for licensing reasons).
We prefer to use CLang. If you want to stick with GCC, we recommend installing GCC 4.4 or GCC 4.6 instead of using
GCC 4.2 (poor performance on recent CPUs).
```
pkg_add -r gcc46
```

CMake reads CC and CXX variables. If you do not want to use the default compiler, just do
```
export CC=clang
export CXX=clang++
```

Additional packages can be easily installed with:
```
pkg_add -r readline
pkg_add -r cmake
pkg_add -r git
pkg_add -r gnuplot
```

Note: on FreeBSD 9.0, it seems `pdflib` (a dependency of gnuplot) is not available as binary. Please,
install gnuplot instead in the port tree:
```
cd /usr/ports/math/gnuplot
make install clean
```

For installing QT, use also `pkg_add -r qt4`, followed by `pkg_add -r qt4-XXX`, where
XXX is one of the components (or tools) listed on [Qt FreeBSD page](http://www.freebsd.org/doc/en/books/porters-handbook/using-qt.html).
Be sure to install all components and tools listed there.

An excellent BLAS/LAPACK implementation is also recommended for speed. See
our [BLAS recommendations](blas).

### B. Getting Torch sources ###

Same as [getting sources](#install.sources) for linux.

### C. Configuring Torch ###

Same as [configuring](#install.config) for linux. Note that dynamic RPATH (related to `$ORIGIN`) do not work properly
on my FreeBSD 9. You can deactivate this with the `WITH_DYNAMIC_RPATH` option.
```
cmake .. -DCMAKE_INSTALL_PREFIX=/my/install/path -DWITH_DYNAMIC_RPATH=OFF
```

### D. Compiling and Installing ###

Same as [compiling](#install.compile) for linux.

### E. Running Torch ###

Same as [runnning](#install.run) for linux.

## Cygwin ##

_We do not recommend_ Cygwin installation. Cygwin is pretty slow, and we
could not manage to make QT 4.4 work under Cygwin. Instead prefer
[native windows](#Windows) installation.

<a name="Windows"/>
## Windows ##

___ Torch7 is not yet Windows compatible, coming soon ___


<a name="CMakeHints"/>
## CMake hints ##

CMake is well documented on [http:_www.cmake.org](http:_www.cmake.org).

### CMake and CLang ###

If you like to use [CLang](http://clang.llvm.org) for compiling Torch7, assuming a proper
CLang installation, you only have to do
```
export CC=clang
export CXX=clang++
```
before calling cmake command line.

### CMake GUI ###

Under Windows, CMake comes by default with a GUI. Under Unix system it is
quite handy to use the _text GUI_ available through `ccmake`.
`ccmake` works in the same way than `cmake`: go in your build directory and
```
ccmake /path/to/torch/source
```

Windows and Unix GUI works in the same way: you `configure`, _possibly several times_,
until CMake has detected everything and proposes to `generate` the configuration.

After each configuration step, you can modify CMake variables to suit your needs.

### CMake variables ###

CMake is highly configurable thanks to _variables_ you can set when
executing it. It is really easy to change these variables with CMake GUI. If you want
to stick with the command line you can also change a variable by doing:
```
cmake /path/to/torch/source -DMY_VARIABLE=MY_VALUE
```
where `MY_VARIABLE` is the name of the variable you want to set and
`MY_VALUE` is its corresponding value.

#### Interesting standard CMake variables ####

  * `CMAKE_INSTALL_PREFIX`: directory where Torch is going to be installed
  * `CMAKE_BUILD_TYPE`: `Release` for optimized compilation, `Debug` for debug compilation.
  * `CMAKE_C_FLAGS`: add here the flags you want to pass to the C compiler (like `-Wall` for e.g.)

#### Notable Torch7 CMake variables ####

  * `WITH_BLAS`: specify which BLAS you want to use (if you have several on your computers). Can be mkl/open/goto/acml/atlas/accelerate/veclib/generic.
  * `WITH_LUA_JIT`: say to CMake to compile Torch7 against LuaJIT instead of Lua. (default is OFF)
  * `WITH_QTLUA`: compile QtLua if Qt is found (default is ON)
  * `WITH_QTLUA_IDE`: compile QtLua IDE if Qt is found (default is ON)
  * `WITH_RPATH`: use RPATH such that you do not need to add Torch7 install library path in LD_LIBRARY_PATH. (default is ON)
  * `WITH_DYNAMIC_RPATH`: if used together with WITH_RPATH, will make library paths relative to the Torch7 executable. If you move the install directory, things will still work. This flag does not work on FreeBSD. (default is ON).

### CMake caches everything ###

As soon as CMake performed a test to detect an external library, it saves
the result of this test in a cache and will not test it again.

If you forgot to install a library (like QT or Readline), and install it
after having performed a CMake configuration, it will not be used by Torch
when compiling.

_In doubt_, if you changed, updated, added some libraries that should be used by Torch, you should
_erase your build directory and perform CMake configuration again_.


<a name="DevPackages"/>
## Development Torch packages ##

If you want to develop your own package, you can put it in the `dev`
sub-directory. Packages in `dev` are all compiled in the same way that the
ones in `packages` sub-directory. We prefer to have this directory to make a
clear difference between official packages and development packages.

Alternatively, you can use [Torch package manager](#PackageManager) 
to build and distribute your packages.

<a name="PackageManager"/>
## The Torch Package Management System ##

Torch7 has a built-in package management system that makes it very easy 
for anyone to get extra (experimental) packages, and create and distribute
yours.

Calling `torch-pkg` without arguments will give you some help:

```
/usr/local/bin/torch-pkg
Torch7 Package Manager

Usage: torch-pkg [options] <command> [argument]

Supported Commands:
  help            print this help
  install         install a package (download + build + deploy)
  download        download a package locally
  build           build a local package
  deploy          deploy a package (build it if necessary)
  list            list available packages
  search          search for a package
  add             add a server address to local config ($HOME/.torchpkg/config)

Arguments for install|download:
  <pkg-name>      a package name (to be found in one of the configured repos)
  <pkg-repo>      the full address of a GIT repository
  <pkg-url>       the URL of a simple package (should be a tar/tgz/tbz)

Arguments for add:
  <base-url>      a base URL where GIT repos or tars can be found

Options:
  -v|-verbose     be more verbose
  -l|-local       local install
  -n|-nodeps      do not install dependencies (when installing)
  -d|-dry         dry run 
```

It's fairly self-explanatory. You can easily get a list of the available
packages:

```
/usr/local/bin/torch-pkg list

--> retrieving package lists from servers

--> parallel
    A parallel computing framework for Torch7, with serialization facilities
    hosted at: https://github.com/clementfarabet/lua---parallel

--> image
    An image-processing toolbox for Torch7
    hosted at: https://github.com/clementfarabet/lua---image

--> optim
    An optimization toolbox for Torch7
    hosted at: https://github.com/koraykv/optim

...
```

To install a new package, simply do:

```
/usr/local/bin/torch-pkg install pkgname
```

The sources of the packages are downloaded and kept in a hidden
directory in your home:

```
torch-pkg install image
ls ~/.torch/torch-pkg/image/
```

If you just want to get the sources of a package, without
installing it, you can get it like this:

```
/usr/local/bin/torch-pkg download pkgname
```

And then build it and install it:

```
cd pkgname
/usr/local/bin/torch-pkg build
/usr/local/bin/torch-pkg deploy
```

If you need to distribute your own packages, you just have
to create a package file, which contains one entry per package,
and then make it available online. Users can then easily add
that file to their repository by doing:

```
/usr/local/bin/torch-pkg add http://url/to/config
```

The config typically looks like:

```
pkg = pkg or {}

pkg.image = {
  git = 'https://github.com/clementfarabet/lua---image',
  description = 'An image-processing toolbox for Torch7',
  dependencies = {'sys', 'xlua'},
  commit = 'master'
}

pkg.optim = {
  git = 'https://github.com/koraykv/optim',
  description = 'An optimization toolbox for Torch7',
  dependencies = {},
  commit = 'newpack'
}

pkg.parallel = {
  git = 'https://github.com/clementfarabet/lua---parallel',
  description = 'A parallel computing framework for Torch7, with serialization facilities',
  dependencies = {'sys'},
  commit = 'newpack'
}
```

<a name="install.binary"/>
# Installing from binaries #

__This section is not applicable now as we have not produced binaries yet.__

__Please [install from sources](#install.sources).__



