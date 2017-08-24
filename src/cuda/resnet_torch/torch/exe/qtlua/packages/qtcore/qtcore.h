// -*- C++ -*-

#ifndef QTCORE_H
#define QTCORE_H

extern "C" {
#include "lua.h"
#include "lauxlib.h"
}
#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef _WIN32
# ifdef libqtcore_EXPORTS
#  define QTCORE_API __declspec(dllexport)
# else
#  define QTCORE_API __declspec(dllimport)
# endif
#else
# define QTCORE_API /**/
#endif

#define LUA_EXTERNC extern "C"

LUA_EXTERNC QTCORE_API int luaopen_libqtcore(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


