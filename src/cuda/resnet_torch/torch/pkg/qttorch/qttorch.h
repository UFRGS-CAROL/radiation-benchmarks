// -*- C++ -*-

#ifndef QTTORCH_H
#define QTTORCH_H

extern "C" {
#include "lua.h"
#include "lauxlib.h"
}
#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef _WIN32
# ifdef libqttorch_EXPORTS
#  define QTTORCH_API __declspec(dllexport)
# else
#  define QTTORCH_API __declspec(dllimport)
# endif
#else
# define QTTORCH_API /**/
#endif

#define LUA_EXTERNC extern "C"

LUA_EXTERNC QTTORCH_API int luaopen_libqttorch(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


