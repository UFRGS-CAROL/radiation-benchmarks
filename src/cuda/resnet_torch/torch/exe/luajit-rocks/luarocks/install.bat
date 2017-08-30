rem=rem --[[
@setlocal&  set luafile="%~f0" & if exist "%~f0.bat" set luafile="%~f0.bat"
@win32\lua5.1\bin\lua5.1.exe %luafile% %*&  exit /b ]]

local vars = {}


vars.PREFIX = nil
vars.VERSION = "2.3"
vars.SYSCONFDIR = nil
vars.SYSCONFFORCE = nil
vars.CONFBACKUPDIR = nil
vars.SYSCONFFILENAME = nil
vars.CONFIG_FILE = nil
vars.TREE_ROOT = nil
vars.TREE_BIN = nil
vars.TREE_LMODULE = nil
vars.TREE_CMODULE = nil
vars.LUA_INTERPRETER = nil
vars.LUA_PREFIX = nil
vars.LUA_BINDIR = nil
vars.LUA_INCDIR = nil
vars.LUA_LIBDIR = nil
vars.LUA_LIBNAME = nil
vars.LUA_VERSION = "5.1"
vars.LUA_SHORTV = nil   -- "51"
-- MinGW does not generate .lib, nor needs it to link, but MSVC does
-- so .lib must be listed first to ensure they are found first if present.
-- To prevent MSVC trying to link to a .dll, which won't work.
vars.LUA_LIB_NAMES = "lua5.1.lib lua51.lib lua5.1.dll lua51.dll liblua.dll.a"
vars.LUA_RUNTIME = nil
vars.UNAME_M = nil
vars.COMPILER_ENV_CMD = nil

local FORCE = false
local FORCE_CONFIG = false
local INSTALL_LUA = false
local USE_MINGW = false
local USE_MSVC_MANUAL = false
local REGISTRY = true
local NOADMIN = false
local PROMPT = true
local SELFCONTAINED = false

---
-- Some helpers
-- 

local pe = assert(loadfile(".\\win32\\pe-parser.lua"))()

local function die(message)
	if message then print(message) end
	print()
	print("Failed installing LuaRocks. Run with /? for help.")
	os.exit(1)
end

function split_string(str, delim, maxNb)
	-- Eliminate bad cases...
	if string.find(str, delim) == nil then
		return { str }
	end
	if maxNb == nil or maxNb < 1 then
		maxNb = 0	 -- No limit
	end
	local result = {}
	local pat = "(.-)" .. delim .. "()"
	local nb = 0
	local lastPos
	for part, pos in string.gmatch(str, pat) do
		nb = nb + 1
		result[nb] = part
		lastPos = pos
		if nb == maxNb then break end
	end
	-- Handle the last field
	if nb ~= maxNb then
		result[nb + 1] = string.sub(str, lastPos)
	end
	return result
end


local function exec(cmd)
	--print(cmd)
	local status = os.execute("type NUL && "..cmd)
	return (status == 0 or status == true) -- compat 5.1/5.2
end

local function exists(filename)
	local cmd = [[.\win32\tools\test -e "]]..filename..[["]]
	return exec(cmd)
end

local function mkdir (dir)
	return exec([[.\win32\tools\mkdir -p "]]..dir..[[" >NUL]])
end

-- does the current user have admin privileges ( = elevated)
local function permission()
	return exec("net session >NUL 2>&1") -- fails if not admin
end

-- rename filename (full path) to backupname (name only), appending number if required
-- returns the new name (name only)
local function backup(filename, backupname)
	local path = filename:match("(.+)%\\.-$").."\\"
	local nname = backupname
	local i = 0
	while exists(path..nname) do
		i = i + 1
		nname = backupname..tostring(i)
	end
	exec([[REN "]]..filename..[[" "]]..nname..[[" > NUL]])
	return nname
end

-- interpolate string with values from 'vars' table
local function S (tmpl)
	return (tmpl:gsub('%$([%a_][%w_]*)', vars))
end

local function print_help()
	print(S[[
Installs LuaRocks.

/P [dir]       Where to install LuaRocks. 
               Default is %PROGRAMFILES%\LuaRocks

Configuring the destinations:
/TREE [dir]    Root of the local system tree of installed rocks.
               Default is {BIN}\..\ if {BIN} ends with '\bin'
               otherwise it is {BIN}\systree. 
/SCRIPTS [dir] Where to install commandline scripts installed by
               rocks. Default is {TREE}\bin.
/LUAMOD [dir]  Where to install Lua modules installed by rocks.
               Default is {TREE}\share\lua\{LV}.
/CMOD [dir]    Where to install c modules installed by rocks.
               Default is {TREE}\lib\lua\{LV}.
/CONFIG [dir]  Location where the config file should be installed.
               Default is to follow /P option
/SELFCONTAINED Creates a self contained installation in a single
               directory given by /P.
               Sets the /TREE and /CONFIG options to the same 
               location as /P. And does not load registry info
               with option /NOREG. The only option NOT self
               contained is the user rock tree, so don't use that
               if you create a self contained installation.
               
Configuring the Lua interpreter:
/LV [version]  Lua version to use; either 5.1, 5.2, or 5.3.
               Default is 5.1
/LUA [dir]     Location where Lua is installed - e.g. c:\lua\5.1\
               If not provided, the installer will search the system
               path and some default locations for a valid Lua
               installation.
               This is the base directory, the installer will look
               for subdirectories bin, lib, include. Alternatively
               these can be specified explicitly using the /INC,
               /LIB, and /BIN options.
/INC [dir]     Location of Lua includes - e.g. c:\lua\5.1\include
               If provided overrides sub directory found using /LUA.
/LIB [dir]     Location of Lua libraries (.dll/.lib) - e.g. c:\lua\5.1\lib
               If provided overrides sub directory found using /LUA.
/BIN [dir]     Location of Lua executables - e.g. c:\lua\5.1\bin
               If provided overrides sub directory found using /LUA.
/L             Install LuaRocks' own copy of Lua even if detected,
               this will always be a 5.1 installation.
               (/LUA, /INC, /LIB, /BIN cannot be used with /L)

Compiler configuration:
               By default the installer will try to determine the 
               Microsoft toolchain to use. And will automatically use 
               a setup command to initialize that toolchain when 
               LuaRocks is run. If it cannot find it, it will default 
               to the /MSVC switch.
/MSVC          Use MS toolchain, without a setup command (tools must
               be in your path)
/MW            Use mingw as build system (tools must be in your path)

Other options:
/FORCECONFIG   Use a single config location. Do not use the
               LUAROCKS_CONFIG variable or the user's home directory.
               Useful to avoid conflicts when LuaRocks
               is embedded within an application.
/F             Remove installation directory if it already exists.
/NOREG         Do not load registry info to register '.rockspec'
               extension with LuaRocks commands (right-click).
/NOADMIN       The installer requires admin privileges. If not
               available it will elevate a new process. Use this
               switch to prevent elevation, but make sure the
               destination paths are all accessible for the current
               user.
/Q             Do not prompt for confirmation of settings

]])
end

-- ***********************************************************
-- Option parser
-- ***********************************************************
local function parse_options(args)
	for _, option in ipairs(args) do
		local name = option.name:upper()
		if name == "/?" then
			print_help()
			os.exit(0)
		elseif name == "/P" then
			vars.PREFIX = option.value
		elseif name == "/CONFIG" then
			vars.SYSCONFDIR = option.value
			vars.SYSCONFFORCE = true
		elseif name == "/TREE" then
			vars.TREE_ROOT = option.value
		elseif name == "/SCRIPTS" then
			vars.TREE_BIN = option.value
		elseif name == "/LUAMOD" then
			vars.TREE_LMODULE = option.value
		elseif name == "/CMOD" then
			vars.TREE_CMODULE = option.value
		elseif name == "/LV" then
			vars.LUA_VERSION = option.value
		elseif name == "/L" then
			INSTALL_LUA = true
		elseif name == "/MW" then
			USE_MINGW = true
		elseif name == "/MSVC" then
			USE_MSVC_MANUAL = true
		elseif name == "/LUA" then
			vars.LUA_PREFIX = option.value
		elseif name == "/LIB" then
			vars.LUA_LIBDIR = option.value
		elseif name == "/INC" then
			vars.LUA_INCDIR = option.value
		elseif name == "/BIN" then
			vars.LUA_BINDIR = option.value
		elseif name == "/FORCECONFIG" then
			FORCE_CONFIG = true
		elseif name == "/F" then
			FORCE = true
		elseif name == "/SELFCONTAINED" then
			SELFCONTAINED = true
		elseif name == "/NOREG" then
			REGISTRY = false
		elseif name == "/NOADMIN" then
			NOADMIN = true
		elseif name == "/Q" then
			PROMPT = false
		else
			die("Unrecognized option: " .. name)
		end
	end
end

-- check for combination/required flags
local function check_flags()
	if SELFCONTAINED then
		if not vars.PREFIX then
			die("Option /P is required when using /SELFCONTAINED")
		end
		if vars.SYSCONFDIR or vars.TREE_ROOT or vars.TREE_BIN or vars.TREE_LMODULE or vars.TREE_CMODULE then
			die("Cannot combine /TREE, /SCRIPTS, /LUAMOD, /CMOD, or /CONFIG with /SELFCONTAINED")
		end
	end
	if INSTALL_LUA then
		if vars.LUA_INCDIR or vars.LUA_BINDIR or vars.LUA_LIBDIR or vars.LUA_PREFIX then
			die("Cannot combine option /L with any of /LUA /BIN /LIB /INC")
		end
		if vars.LUA_VERSION ~= "5.1" then
			die("Bundled Lua version is 5.1, cannot install "..vars.LUA_VERSION)
		end
	end
	if vars.LUA_VERSION ~= "5.1" then
		if vars.LUA_VERSION == "5.2" then
			vars.LUA_LIB_NAMES = vars.LUA_LIB_NAMES:gsub("5([%.]?)1", "5%12")
		elseif vars.LUA_VERSION == "5.3" then
			vars.LUA_LIB_NAMES = vars.LUA_LIB_NAMES:gsub("5([%.]?)1", "5%13")
		else
			die("Bad argument: /LV must either be 5.1, 5.2, or 5.3")
		end
	end
  if USE_MSVC_MANUAL and USE_MINGW then
    die("Cannot combine option /MSVC and /MW")
  end
end

-- ***********************************************************
-- Detect Lua
-- ***********************************************************
local function look_for_interpreter (directory)
	if vars.LUA_BINDIR then
        -- if LUA_BINDIR is specified, it must be there, otherwise we fail
		if exists( S"$LUA_BINDIR\\lua$LUA_VERSION.exe" ) then
			vars.LUA_INTERPRETER = S"lua$LUA_VERSION.exe"
			print(S"       Found $LUA_BINDIR\\$LUA_INTERPRETER")
			return true
		elseif exists( S"$LUA_BINDIR\\lua$LUA_SHORTV.exe" ) then
			vars.LUA_INTERPRETER = S"lua$LUA_SHORTV.exe"
			print(S"       Found $LUA_BINDIR\\$LUA_INTERPRETER")
			return true
		elseif exists(S"$LUA_BINDIR\\lua.exe") then
			vars.LUA_INTERPRETER = "lua.exe"
			print(S"       Found $LUA_BINDIR\\$LUA_INTERPRETER")
			return true
		elseif exists(S"$LUA_BINDIR\\luajit.exe") then
			vars.LUA_INTERPRETER = "luajit.exe"
			print(S"       Found $LUA_BINDIR\\$LUA_INTERPRETER")
			return true
		end
		die(S"Lua executable lua.exe, luajit.exe, lua$LUA_SHORTV.exe or lua$LUA_VERSION.exe not found in $LUA_BINDIR")
	end

	for _, e in ipairs{ [[\]], [[\bin\]] } do
		if exists(directory..e.."\\lua"..vars.LUA_VERSION..".exe") then
			vars.LUA_INTERPRETER = S"lua$LUA_VERSION.exe"
			vars.LUA_BINDIR = directory .. e
			print("       Found ."..e..vars.LUA_INTERPRETER)
			return true

		elseif exists(directory..e.."\\lua"..vars.LUA_SHORTV..".exe") then
			vars.LUA_INTERPRETER = S"lua$LUA_SHORTV.exe"
			vars.LUA_BINDIR = directory .. e
			print("       Found ."..e..vars.LUA_INTERPRETER)
			return true

		elseif exists(directory..e.."\\lua.exe") then
			vars.LUA_INTERPRETER = "lua.exe"
			vars.LUA_BINDIR = directory..e
			print("       Found ."..e..vars.LUA_INTERPRETER)
			return true

		elseif exists(directory..e.."\\luajit.exe") then
			vars.LUA_INTERPRETER = "luajit.exe"
			vars.LUA_BINDIR = directory..e
			print("       Found ."..e..vars.LUA_INTERPRETER)
			return true
		end
	end
	--print("      No Lua interpreter found")
	return false
end

local function look_for_link_libraries (directory)
	if vars.LUA_LIBDIR then
		for name in vars.LUA_LIB_NAMES:gmatch("[^%s]+") do
			print(S"    checking for $LUA_LIBDIR\\"..name)
			if exists(vars.LUA_LIBDIR.."\\"..name) then
				vars.LUA_LIBNAME = name
				print("       Found "..name)
				return true
			end
		end
		die(S"link library (one of; $LUA_LIB_NAMES) not found in $LUA_LIBDIR")
	end

	for _, e in ipairs{ [[\]], [[\lib\]], [[\bin\]]} do
		for name in vars.LUA_LIB_NAMES:gmatch("[^%s]+") do
			print("    checking for "..directory..e.."\\"..name)
			if exists(directory..e.."\\"..name) then
				vars.LUA_LIBDIR = directory .. e
				vars.LUA_LIBNAME = name
				print("       Found "..name)
				return true
			end
		end
	end
	return false
end

local function look_for_headers (directory)
	if vars.LUA_INCDIR then
		print(S"    checking for $LUA_INCDIR\\lua.h")
		if exists(S"$LUA_INCDIR\\lua.h") then
			print("       Found lua.h")
			return true
		end
		die(S"lua.h not found in $LUA_INCDIR")
	end

	for _, e in ipairs{ 
        S([[\include\lua\$LUA_VERSION]]), 
        S([[\include\lua$LUA_SHORTV]]), 
        S([[\include\lua$LUA_VERSION]]), 
        S([[\include\$LUA_VERSION]]), 
        [[\include\]],
        [[\]], 
      } do
		print("    checking for "..directory..e.."\\lua.h")
		if exists(directory..e.."\\lua.h") then
			vars.LUA_INCDIR = directory..e
			print("       Found lua.h")
			return true
		end
	end
	return false
end


local function get_runtime()
	local f
	vars.LUA_RUNTIME, f = pe.msvcrt(vars.LUA_BINDIR.."\\"..vars.LUA_INTERPRETER)
	if type(vars.LUA_RUNTIME) ~= "string" then
		-- analysis failed, issue a warning
		vars.LUA_RUNTIME = "MSVCR80"
		print("*** WARNING ***: could not analyse the runtime used, defaulting to "..vars.LUA_RUNTIME)
	else
		print("    "..f.." uses "..vars.LUA_RUNTIME..".DLL as runtime")
	end
	return true
end

local function get_architecture()
	-- detect processor arch interpreter was compiled for
	local proc = (pe.parse(vars.LUA_BINDIR.."\\"..vars.LUA_INTERPRETER) or {}).Machine
	if not proc then
		die("Could not detect processor architecture used in "..vars.LUA_INTERPRETER)
	end
	proc = pe.const.Machine[proc]  -- collect name from constant value
	if proc == "IMAGE_FILE_MACHINE_I386" then
		proc = "x86"
	else
		proc = "x86_64"
	end
	return proc
end

-- get a string value from windows registry.
local function get_registry(key, value)
	local keys = {key}
	local key64, replaced = key:gsub("(%u+\\Software\\)", "\1Wow6432Node\\", 1)

	if replaced == 1 then
		keys = {key64, key}
	end

	for _, k in ipairs(keys) do
		local h = io.popen('reg query "'..k..'" /v '..value..' 2>NUL')
		local output = h:read("*a")
		h:close()

		local v = output:match("REG_SZ%s+([^\n]+)")
		if v then
			return v
		end
	end
	return nil
end

local function get_visual_studio_directory()
	assert(type(vars.LUA_RUNTIME)=="string", "requires vars.LUA_RUNTIME to be set before calling this function.")
	local major, minor = vars.LUA_RUNTIME:match('VCR%u*(%d+)(%d)$') -- MSVCR<x><y> or VCRUNTIME<x><y>
	if not major then 
    print(S[[    Cannot auto-detect Visual Studio version from $LUA_RUNTIME]])
    return nil 
  end
	local keys = {
		"HKLM\\Software\\Microsoft\\VisualStudio\\%d.%d\\Setup\\VC",
		"HKLM\\Software\\Microsoft\\VCExpress\\%d.%d\\Setup\\VS"
	}
	for _, key in ipairs(keys) do
    local versionedkey = key:format(major, minor)
		local vcdir = get_registry(versionedkey, "ProductDir")
    print("    checking: "..versionedkey)
		if vcdir then 
      print("        Found: "..vcdir)
      return vcdir 
    end
	end
	return nil
end

local function get_windows_sdk_directory()
	assert(type(vars.LUA_RUNTIME) == "string", "requires vars.LUA_RUNTIME to be set before calling this function.")
	-- Only v7.1 and v6.1 shipped with compilers
	-- Other versions requires a separate  installation of Visual Studio.
	-- see https://github.com/keplerproject/luarocks/pull/443#issuecomment-152792516
	local wsdks = {
		["MSVCR100"] = "v7.1", -- shipped with Visual Studio 2010 compilers.
		["MSVCR100D"] = "v7.1", -- shipped with Visual Studio 2010 compilers.
		["MSVCR90"] = "v6.1", -- shipped with Visual Studio 2008 compilers.
		["MSVCR90D"] = "v6.1", -- shipped with Visual Studio 2008 compilers.
	}
	local wsdkver = wsdks[vars.LUA_RUNTIME]
	if not wsdkver then
    print(S[[    Cannot auto-detect Windows SDK version from $LUA_RUNTIME]])
		return nil
	end

	local key = "HKLM\\Software\\Microsoft\\Microsoft SDKs\\Windows\\"..wsdkver
  print("   checking: "..key)
  local dir = get_registry(key, "InstallationFolder")
  if dir then
    print("        Found: "..dir)
    return dir
  end
  print("        No SDK found")
	return nil
end

-- returns the batch command to setup msvc compiler path.
-- or an empty string (eg. "") if not found
local function get_msvc_env_setup_cmd()
  print(S[[Looking for Microsoft toolchain matching runtime $LUA_RUNTIME and architecture $UNAME_M]])

	assert(type(vars.UNAME_M) == "string", "requires vars.UNAME_M to be set before calling this function.")
	local x64 = vars.UNAME_M=="x86_64"

	-- 1. try visual studio command line tools
	local vcdir = get_visual_studio_directory()
	if vcdir then
		-- 1.1. try vcvarsall.bat
		local vcvarsall = vcdir .. 'vcvarsall.bat'
		if exists(vcvarsall) then
			return ('call "%s"%s'):format(vcvarsall, x64 and ' amd64' or '')
		end

		-- 1.2. try vcvars32.bat / vcvars64.bat
		local relative_path = x64 and "bin\\amd64\\vcvars64.bat" or "bin\\vcvars32.bat"
		local full_path = vcdir .. relative_path
		if exists(full_path) then
			return ('call "%s"'):format(full_path)
		end
	end

	-- 2. try for Windows SDKs command line tools.
	local wsdkdir = get_windows_sdk_directory()
	if wsdkdir then
		local setenv = wsdkdir.."Bin\\SetEnv.cmd"
		if exists(setenv) then
			return ('call "%s" /%s'):format(setenv, x64 and "x64" or "x86")
		end
	end

	-- finally, we can't detect more, just don't setup the msvc compiler in luarocks.bat.
	return ""
end

local function look_for_lua_install ()
	print("Looking for Lua interpreter")
	local directories
	if vars.LUA_PREFIX then
		directories = { vars.LUA_PREFIX }
	else
		-- no prefix given, so use path
		directories = (os.getenv("PATH",";") or "")
		local i = 1
        while i ~= 0 do directories, i = directories:gsub(";;",";") end  --remove all doubles
		directories = split_string(directories,";")
		-- if a path element ends with "\bin\" then remove it, as the searcher will check there anyway
		for i, val in ipairs(directories) do
			-- remove trailing backslash
			while val:sub(-1,-1) == "\\" and val:sub(-2,-1) ~= ":\\" do 
				val = val:sub(1,-2)
			end
			-- remove trailing 'bin'
			if val:upper():sub(-4,-1) == "\\BIN" or val:upper():sub(-4,-1) == ":BIN" then
				val = val:sub(1,-5)
			end
			directories[i] = val
		end
		-- finaly add some other default paths
		table.insert(directories, [[c:\lua5.1.2]])
		table.insert(directories, [[c:\lua]])
		table.insert(directories, [[c:\kepler\1.1]])
	end
	if vars.LUA_BINDIR and vars.LUA_LIBDIR and vars.LUA_INCDIR then
		if look_for_interpreter(vars.LUA_BINDIR) and 
			look_for_link_libraries(vars.LUA_LIBDIR) and
			look_for_headers(vars.LUA_INCDIR)
		then
			if get_runtime() then
				print("Runtime check completed, now testing interpreter...")
				if exec(S[["$LUA_BINDIR\$LUA_INTERPRETER" -v 2>NUL]]) then
					print("    Ok")
					return true
				end
				print("   Interpreter returned an error, not ok")
			end
		end
		return false
	end
	
	for _, directory in ipairs(directories) do
		print("    checking " .. directory)
		if exists(directory) then
			if look_for_interpreter(directory) then
				print("Interpreter found, now looking for link libraries...")
				if look_for_link_libraries(directory) then
					print("Link library found, now looking for headers...")
					if look_for_headers(directory) then
						print("Headers found, checking runtime to use...")
						if get_runtime() then
							print("Runtime check completed, now testing interpreter...")
							if exec(S[["$LUA_BINDIR\$LUA_INTERPRETER" -v 2>NUL]]) then
								print("    Ok")
								return true
							end
							print("   Interpreter returned an error, not ok")
						end
					end
				end
			end
		end
	end
	return false
end

-- backup config[x.x].lua[.bak] and site_config[_x_x].lua
local function backup_config_files()
  local temppath
  while not temppath do
    temppath = os.getenv("temp").."\\LR-config-backup-"..tostring(math.random(10000))
    if exists(temppath) then temppath = nil end
  end
  vars.CONFBACKUPDIR = temppath
  mkdir(vars.CONFBACKUPDIR)
  exec(S[[COPY "$PREFIX\config*.*" "$CONFBACKUPDIR" >NUL]])
  exec(S[[COPY "$PREFIX\lua\luarocks\site_config*.*" "$CONFBACKUPDIR" >NUL]])
end

-- restore previously backed up config files
local function restore_config_files()
  if not vars.CONFBACKUPDIR then return end -- there is no backup to restore
  exec(S[[COPY "$CONFBACKUPDIR\config*.*" "$PREFIX" >NUL]])
  exec(S[[COPY "$CONFBACKUPDIR\site_config*.*" "$PREFIX\lua\luarocks" >NUL]])
  -- cleanup
  exec(S[[RD /S /Q "$CONFBACKUPDIR"]])
  vars.CONFBACKUPDIR = nil
end

-- ***********************************************************
-- Installer script start
-- ***********************************************************

-- Poor man's command-line parsing
local config = {}
local with_arg = { -- options followed by an argument, others are flags
	["/P"] = true,
	["/CONFIG"] = true,
	["/TREE"] = true,
	["/SCRIPTS"] = true,
	["/LUAMOD"] = true,
	["/CMOD"] = true,
	["/LV"] = true,
	["/LUA"] = true,
	["/INC"] = true,
	["/BIN"] = true,
	["/LIB"] = true,
}
-- reconstruct argument values with spaces and double quotes
-- this will be damaged by the batch construction at the start of this file
local oarg = arg  -- retain old table
if #arg > 0 then
	farg = table.concat(arg, " ") .. " "
	arg = {}
	farg = farg:gsub('%"', "")
	local i = 0
	while #farg>0 do
		i = i + 1
		if (farg:sub(1,1) ~= "/") and ((arg[i-1] or ""):sub(1,1) ~= "/") then
			i = i - 1       -- continued previous arg
			if i == 0 then i = 1 end
		end
		if arg[i] then
			arg[i] = arg[i] .. " "
		else
			arg[i] = ""
		end
		local v,r = farg:match("^(.-)%s(.*)$")
		arg[i], farg = arg[i]..v, r
		while farg:sub(1,1) == " " do farg = farg:sub(2,-1) end	-- remove prefix spaces
	end
end
for k,v in pairs(oarg) do if k < 1 then arg[k] = v end end -- copy 0 and negative indexes
oarg = nil

-- build config option table with name and value elements
local i = 1
while i <= #arg do
	local opt = arg[i]
	if with_arg[opt:upper()] then
		local value = arg[i + 1]
		if not value then
			die("Missing value for option "..opt)
		end
		config[#config + 1] = { name = opt, value = value }
		i = i + 1
	else
		config[#config + 1] = { name = opt }
	end
	i = i + 1
end

print(S"LuaRocks $VERSION.x installer.\n")

parse_options(config)

print([[

========================
== Checking system... ==
========================

]])

check_flags()

if not permission() then
	if not NOADMIN then
		-- must elevate the process with admin privileges
        if not exec("PowerShell /? >NUL 2>&1") then
          -- powershell is not available, so error out
          die("No administrative privileges detected and cannot auto-elevate. Please run with admin privileges or use the /NOADMIN switch")
        end
		print("Need admin privileges, now elevating a new process to continue installing...")
		local runner = os.getenv("TEMP").."\\".."LuaRocks_Installer.bat"
		local f = io.open(runner, "w")
		f:write("@echo off\n")
		f:write("CHDIR /D "..arg[0]:match("(.+)%\\.-$").."\n")  -- return to current dir, elevation changes current path
		f:write('"'..arg[-1]..'" "'..table.concat(arg, '" "', 0)..'"\n')
		f:write("ECHO Press any key to close this window...\n")
		f:write("PAUSE > NUL\n")
		f:write('DEL "'..runner..'"')  -- temp batch file deletes itself
		f:close()
		-- run the created temp batch file in elevated mode
		exec("PowerShell -Command (New-Object -com 'Shell.Application').ShellExecute('"..runner.."', '', '', 'runas')\n")
		print("Now exiting unprivileged installer")
       	os.exit()  -- exit here, the newly created elevated process will do the installing
	else
		print("Attempting to install without admin privileges...")
	end
else
	print("Admin privileges available for installing")
end

vars.PREFIX = vars.PREFIX or os.getenv("PROGRAMFILES")..[[\LuaRocks]]
vars.BINDIR = vars.PREFIX
vars.LIBDIR = vars.PREFIX
vars.LUADIR = S"$PREFIX\\lua"
vars.INCDIR = S"$PREFIX\\include"
vars.LUA_SHORTV = vars.LUA_VERSION:gsub("%.", "")

if INSTALL_LUA then
	if vars.LUA_VERSION ~= "5.1" then
		die("Cannot install own copy of Lua because only 5.1 is bundled")
	end
	vars.LUA_INTERPRETER = "lua5.1"
	vars.LUA_BINDIR = vars.BINDIR
	vars.LUA_LIBDIR = vars.LIBDIR
	vars.LUA_INCDIR = vars.INCDIR
	vars.LUA_LIBNAME = "lua5.1.lib"
	vars.LUA_RUNTIME = "MSVCR80"
	vars.UNAME_M = "x86"
else
	if not look_for_lua_install() then
		die("Could not find Lua. See /? for options for specifying the location of Lua, or installing a bundled copy of Lua 5.1.")
	end
    vars.UNAME_M = get_architecture()  -- can only do when installation was found
end

-- check location of system tree
if not vars.TREE_ROOT then
  -- no system tree location given, so we need to construct a default value
  if vars.LUA_BINDIR:lower():match([[([\/]+bin[\/]*)$]]) then
    -- lua binary is located in a 'bin' subdirectory, so assume
    -- default Lua layout and match rocktree on top
    vars.TREE_ROOT = vars.LUA_BINDIR:lower():gsub([[[\/]+bin[\/]*$]], [[\]])
  else
    -- no 'bin', so use a named tree next to the Lua executable
    vars.TREE_ROOT = vars.LUA_BINDIR .. [[\systree]]
  end
end

local datapath
vars.SYSCONFDIR = vars.SYSCONFDIR or vars.PREFIX
vars.SYSCONFFILENAME = S"config-$LUA_VERSION.lua"
vars.CONFIG_FILE = vars.SYSCONFDIR.."\\"..vars.SYSCONFFILENAME
if SELFCONTAINED then
	vars.SYSCONFDIR = vars.PREFIX
	vars.SYSCONFFORCE = true
	vars.TREE_ROOT = vars.PREFIX..[[\systree]]
	REGISTRY = false
end
vars.COMPILER_ENV_CMD = (USE_MINGW and "") or (USE_MSVC_MANUAL and "") or get_msvc_env_setup_cmd()

print(S[[

==========================
== System check results ==
==========================

Will configure LuaRocks with the following paths:
LuaRocks        : $PREFIX
Config file     : $CONFIG_FILE
Rocktree        : $TREE_ROOT

Lua interpreter : $LUA_BINDIR\$LUA_INTERPRETER
    binaries    : $LUA_BINDIR
    libraries   : $LUA_LIBDIR
    includes    : $LUA_INCDIR
    architecture: $UNAME_M
    binary link : $LUA_LIBNAME with runtime $LUA_RUNTIME.dll
]])

if USE_MINGW then
  print("Compiler        : MinGW (make sure it is in your path before using LuaRocks)")
else
  if vars.COMPILER_ENV_CMD == "" then
    print("Compiler        : Microsoft (make sure it is in your path before using LuaRocks)")
  else
    print(S[[Compiler        : Microsoft, using; $COMPILER_ENV_CMD]])
  end
end

if PROMPT then
	print("\nPress <ENTER> to start installing, or press <CTRL>+<C> to abort. Use install /? for installation options.")
	io.read()
end

print([[

============================
== Installing LuaRocks... ==
============================

]])

-- ***********************************************************
-- Install LuaRocks files
-- ***********************************************************

if exists(vars.PREFIX) then
  if not FORCE then
    die(S"$PREFIX exists. Use /F to force removal and reinstallation.")
  else
    backup_config_files()
    print(S"Removing $PREFIX...")
    exec(S[[RD /S /Q "$PREFIX"]])
    print()
  end
end

print(S"Installing LuaRocks in $PREFIX...")
if not exists(vars.BINDIR) then
	if not mkdir(vars.BINDIR) then
		die()
	end
end

if INSTALL_LUA then
	-- Copy the included Lua interpreter binaries
	if not exists(vars.LUA_BINDIR) then
		mkdir(vars.LUA_BINDIR)
	end
	if not exists(vars.LUA_INCDIR) then
		mkdir(vars.LUA_INCDIR)
	end
	exec(S[[COPY win32\lua5.1\bin\*.* "$LUA_BINDIR" >NUL]])
	exec(S[[COPY win32\lua5.1\include\*.* "$LUA_INCDIR" >NUL]])
	print(S"Installed the LuaRocks bundled Lua interpreter in $LUA_BINDIR")
end

-- Copy the LuaRocks binaries
if not exists(S[[$BINDIR\tools]]) then
	if not mkdir(S[[$BINDIR\tools]]) then
		die()
	end
end
if not exec(S[[COPY win32\tools\*.* "$BINDIR\tools" >NUL]]) then
	die()
end
-- Copy LR bin helper files
if not exec(S[[COPY win32\*.* "$BINDIR" >NUL]]) then
	die()
end
-- Copy the LuaRocks lua source files
if not exists(S[[$LUADIR\luarocks]]) then
	if not mkdir(S[[$LUADIR\luarocks]]) then
		die()
	end
end
if not exec(S[[XCOPY /S src\luarocks\*.* "$LUADIR\luarocks" >NUL]]) then
	die()
end
-- Create start scripts
if not exec(S[[COPY src\bin\*.* "$BINDIR" >NUL]]) then
	die()
end
for _, c in ipairs{"luarocks", "luarocks-admin"} do
	-- rename unix-lua scripts to .lua files
	if not exec( (S[[RENAME "$BINDIR\%s" %s.lua]]):format(c, c) ) then
		die()
	end
	-- create a bootstrap batch file for the lua file, to start them
	exec(S[[DEL /F /Q "$BINDIR\]]..c..[[.bat" 2>NUL]])
	local f = io.open(vars.BINDIR.."\\"..c..".bat", "w")
	f:write(S[[
@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION ENABLEEXTENSIONS
$COMPILER_ENV_CMD
SET "LUA_PATH=$LUADIR\?.lua;$LUADIR\?\init.lua;%LUA_PATH%"
IF NOT "%LUA_PATH_5_2%"=="" (
   SET "LUA_PATH_5_2=$LUADIR\?.lua;$LUADIR\?\init.lua;%LUA_PATH_5_2%"
)
IF NOT "%LUA_PATH_5_3%"=="" (
   SET "LUA_PATH_5_3=$LUADIR\?.lua;$LUADIR\?\init.lua;%LUA_PATH_5_3%"
)
SET "PATH=$BINDIR;%PATH%"
"$LUA_BINDIR\$LUA_INTERPRETER" "$BINDIR\]]..c..[[.lua" %*
SET EXITCODE=%ERRORLEVEL%
IF NOT "%EXITCODE%"=="2" GOTO EXITLR

REM Permission denied error, try and auto elevate...
REM already an admin? (checking to prevent loops)
NET SESSION >NUL 2>&1
IF "%ERRORLEVEL%"=="0" GOTO EXITLR

REM Do we have PowerShell available?
PowerShell /? >NUL 2>&1
IF NOT "%ERRORLEVEL%"=="0" GOTO EXITLR

:GETTEMPNAME
SET TMPFILE=%TEMP%\LuaRocks-Elevator-%RANDOM%.bat
IF EXIST "%TMPFILE%" GOTO :GETTEMPNAME 

ECHO @ECHO OFF                                  >  "%TMPFILE%"
ECHO CHDIR /D %CD%                              >> "%TMPFILE%"
ECHO ECHO %0 %*                                 >> "%TMPFILE%"
ECHO ECHO.                                      >> "%TMPFILE%"
ECHO CALL %0 %*                                 >> "%TMPFILE%"
ECHO ECHO.                                      >> "%TMPFILE%"
ECHO ECHO Press any key to close this window... >> "%TMPFILE%"
ECHO PAUSE ^> NUL                               >> "%TMPFILE%"
ECHO DEL "%TMPFILE%"                            >> "%TMPFILE%"

ECHO Now retrying as a privileged user...
PowerShell -Command (New-Object -com 'Shell.Application').ShellExecute('%TMPFILE%', '', '', 'runas')

:EXITLR
exit /b %EXITCODE% 
]])
	f:close()
	print(S"Created LuaRocks command: $BINDIR\\"..c..".bat")
end

-- ***********************************************************
-- Configure LuaRocks
-- ***********************************************************

restore_config_files()
print()
print("Configuring LuaRocks...")

-- Create a site-config file
local site_config = S("site_config_$LUA_VERSION"):gsub("%.","_")
if exists(S([[$LUADIR\luarocks\]]..site_config..[[.lua]])) then
	local nname = backup(S([[$LUADIR\luarocks\]]..site_config..[[.lua]]), site_config..".lua.bak")
	print("***************")
	print("*** WARNING *** LuaRocks site_config file already exists: '"..site_config..".lua'. The old file has been renamed to '"..nname.."'")
	print("***************")
end
local f = io.open(vars.LUADIR.."\\luarocks\\"..site_config..".lua", "w")
f:write(S[=[
local site_config = {}
site_config.LUA_INCDIR=[[$LUA_INCDIR]]
site_config.LUA_LIBDIR=[[$LUA_LIBDIR]]
site_config.LUA_BINDIR=[[$LUA_BINDIR]]
site_config.LUA_INTERPRETER=[[$LUA_INTERPRETER]]
]=])
if USE_MINGW then
	f:write("site_config.LUAROCKS_UNAME_S=[[MINGW]]\n")
else
	f:write("site_config.LUAROCKS_UNAME_S=[[WindowsNT]]\n")
end
f:write(S[=[
site_config.LUAROCKS_UNAME_M=[[$UNAME_M]]
site_config.LUAROCKS_ROCKS_TREE=[[$TREE_ROOT]]
site_config.LUAROCKS_PREFIX=[[$PREFIX]]
site_config.LUAROCKS_DOWNLOADER=[[wget]]
site_config.LUAROCKS_MD5CHECKER=[[md5sum]]
]=])
if FORCE_CONFIG then
	f:write("site_config.LUAROCKS_FORCE_CONFIG=true\n")
end
if vars.SYSCONFFORCE then  -- only write this value when explcitly given, otherwise rely on defaults
	f:write(S("site_config.LUAROCKS_SYSCONFIG=[[$CONFIG_FILE]]\n"))
end
f:write("return site_config\n")
f:close()
print(S([[Created LuaRocks site-config file: $LUADIR\luarocks\]]..site_config..[[.lua]]))

-- create config file
if not exists(vars.SYSCONFDIR) then
	mkdir(vars.SYSCONFDIR)
end
if exists(vars.CONFIG_FILE) then
	local nname = backup(vars.CONFIG_FILE, vars.SYSCONFFILENAME..".bak")
	print("***************")
	print(S"*** WARNING *** LuaRocks config file already exists: '$CONFIG_FILE'. The old file has been renamed to '"..nname.."'")
	print("***************")
end
local f = io.open(vars.CONFIG_FILE, "w")
f:write([=[
rocks_servers = {
   [[https://raw.githubusercontent.com/torch/rocks/master]],
   [[https://raw.githubusercontent.com/rocks-moonscript-org/moonrocks-mirror/master]]
}
rocks_trees = {
]=])
if FORCE_CONFIG then
	f:write("    home..[[/luarocks]],\n")
end
f:write(S"    { name = [[user]],\n")
f:write(S"         root    = home..[[/luarocks]],\n")
f:write(S"    },\n")
f:write(S"    { name = [[system]],\n")
f:write(S"         root    = [[$TREE_ROOT]],\n")
if vars.TREE_BIN then
  f:write(S"         bin_dir = [[$TREE_BIN]],\n")
end
if vars.TREE_CMODULE then
  f:write(S"         lib_dir = [[$TREE_CMODULE]],\n")
end
if vars.TREE_LMODULE then
  f:write(S"         lua_dir = [[$TREE_LMODULE]],\n")
end
f:write(S"    },\n")
f:write("}\n")
f:write("variables = {\n")
if USE_MINGW and vars.LUA_RUNTIME == "MSVCRT" then
	f:write("    MSVCRT = 'm',   -- make MinGW use MSVCRT.DLL as runtime\n")
else
	f:write("    MSVCRT = '"..vars.LUA_RUNTIME.."',\n")
end
f:write(S"    LUALIB = '$LUA_LIBNAME'\n")
f:write("}\n")
f:write("verbose = false   -- set to 'true' to enable verbose output\n")
f:close()

print(S"Created LuaRocks config file: $CONFIG_FILE")


print()
print("Creating rocktrees...")
if not exists(vars.TREE_ROOT) then
	mkdir(vars.TREE_ROOT)
	print(S[[Created system rocktree    : "$TREE_ROOT"]])
else
	print(S[[System rocktree exists     : "$TREE_ROOT"]])
end

vars.APPDATA = os.getenv("APPDATA")
vars.LOCAL_TREE = vars.APPDATA..[[\LuaRocks]]
if not exists(vars.LOCAL_TREE) then
	mkdir(vars.LOCAL_TREE)
	print(S[[Created local user rocktree: "$LOCAL_TREE"]])
else
	print(S[[Local user rocktree exists : "$LOCAL_TREE"]])
end

-- Load registry information
if REGISTRY then
	-- expand template with correct path information
	print()
	print([[Loading registry information for ".rockspec" files]])
	exec( S[[win32\lua5.1\bin\lua5.1.exe "$PREFIX\LuaRocks.reg.lua" "$PREFIX\LuaRocks.reg.template"]] )
	exec( S[[regedit /S "$PREFIX\\LuaRocks.reg"]] )
end

-- ***********************************************************
-- Cleanup
-- ***********************************************************
-- remove regsitry related files, no longer needed
exec( S[[del "$PREFIX\LuaRocks.reg.*" >NUL]] )
-- remove pe-parser module
exec( S[[del "$PREFIX\pe-parser.lua" >NUL]] )

-- ***********************************************************
-- Exit handlers 
-- ***********************************************************
vars.TREE_BIN     = vars.TREE_BIN     or vars.TREE_ROOT..[[\bin]]
vars.TREE_LMODULE = vars.TREE_LMODULE or vars.TREE_ROOT..[[\share\lua\]]..vars.LUA_VERSION
vars.TREE_CMODULE = vars.TREE_CMODULE or vars.TREE_ROOT..[[\lib\lua\]]..vars.LUA_VERSION
print(S[[

============================
== LuaRocks is installed! ==
============================


You may want to add the following elements to your paths;
Lua interpreter;
  PATH     :   $LUA_BINDIR
  PATHEXT  :   .LUA
LuaRocks;
  PATH     :   $PREFIX
  LUA_PATH :   $PREFIX\lua\?.lua;$PREFIX\lua\?\init.lua
Local user rocktree (Note: %APPDATA% is user dependent);
  PATH     :   %APPDATA%\LuaRocks\bin
  LUA_PATH :   %APPDATA%\LuaRocks\share\lua\$LUA_VERSION\?.lua;%APPDATA%\LuaRocks\share\lua\$LUA_VERSION\?\init.lua
  LUA_CPATH:   %APPDATA%\LuaRocks\lib\lua\$LUA_VERSION\?.dll
System rocktree
  PATH     :   $TREE_BIN
  LUA_PATH :   $TREE_LMODULE\?.lua;$TREE_LMODULE\?\init.lua
  LUA_CPATH:   $TREE_CMODULE\?.dll

Note that the %APPDATA% element in the paths above is user specific and it MUST be replaced by its actual value.
For the current user that value is: $APPDATA.

]])
os.exit(0)
