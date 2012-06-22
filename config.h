#ifndef _CONFIG_H
#define _CONFIG_H

#ifdef __cplusplus
	extern "C" {
#endif

/* This file contains the macro definitions needed to (not) compile in support for certain options. */

/* Define to compile in support for tr_curve_len[] and tr_error_len[], the transformed curve and "error"
 \ lengths.
 */
#define TR_CURVE_LEN 

/* Defined FFTW_CYCLES_PER_SEC to the clock frequency of your <BestThereIs> CPU (in Hz) -- if you have one.
 \ This will cause the timing routines to use the <BestThereIs>'s timing register directly, and not use
 \ gettimeofday for high resolution timing. The resolution obtained should be higher, and importantly,
 \ this is on the order of 15 times faster. (But this is only part of the timers' functionality, so the
 \ end gain is less).
 \ Similar functionality exists for several <BestThereIs> CPUs; Pentium, PowerPC, MIPS - not all implemented
 \ yet, of course. Pentium and PowerPC owners are the lucky ones at this time.
 \ See cpu_cycles_per_second for specific instructions; this is the sh command script that will output the correct
 \ value for known hosts, or a setting which will deactivate the functionality on others.
 */
#include "cpu_cycles_per_second.h"

/* We've begun to make xgraph safe for 64bit systems. Basically, it means that ascanf pointers should "work" */
#define SAFE_FOR_64BIT

/*  define HAVE_FFTW if you have FFTW installed and want to use it for runtime FFTs (in ascanfc2.c).
 \ If you don't, remove the line or #undef it since machdeplibs scans for a #define HAVE_FFTW
 */
#ifndef HAVE_FFTW
#	define HAVE_FFTW	1
#endif

/* define if you want to compile in support for dynamically loading FFTW (HAVE_FFTW should be defined,
 \ and XG_DYMOD_SUPPORT too). This allows to run xgraph on systems that don't have those libraries installed.
 */
#define FFTW_DYNAMIC

/* define if you want to be able to use FFTW's multi-threaded version. This is currently implemented only for
 \ the FFTW_DYNAMIC model.
 */
#ifdef __CYGWIN__
#	define FFTW_THREADED
#else
#	undef FFTW_THREADED
#endif

/* uncomment/define if you have the vscanf() function family. Many modern systems provide these calls in
 \ the standard libraries; for the others, an implementation taken from glibc is provided. The
 \ shell script machdeplibs in the toplevel directory should add these modules to the linker options
 \ on those systems that need it (e.g. Irix 6.3).
 */
#define HAVE_VSCANF

/* Define to compile in support for dynamic module (shared libraries) loading; needs the dlopen() call.	*/
#define XG_DYMOD_SUPPORT

/* There are systems (Linux, Irix, ...) where a dynamically loaded module has direct access to all the symbols
 \ in the main module (programme: xgraph). Sometimes, this requires a compilation/linking flag (-rdynamic on
 \ linux). There are other systems, where this is not possible at all: AIX, and apparently also Darwin/Mach-O/Mac-OSX.
 \ On these, one has to import the required symbols inside the dynamic module, and then use (function) pointers.
 \ On those systems, define XG_DYMOD_IMPORT_MAIN, and include dymod_interface.h in your module; you may need to extend
 \ this interface by adding your own required symbols.
 \ NB: the IMPORT_MAIN mechanism should work on all systems, but the automatic/system mechanism is likely to be
 \ somewhat faster (as it does not go through function pointers).
 \ all systems.
 */
#ifdef XG_DYMOD_SUPPORT
#	if defined(__MACH__) || defined(__APPLE_CC__) || defined(__CYGWIN__)
#		define XG_DYMOD_IMPORT_MAIN
#	else
		// 20111020: the "automatic/system mechanism" existing on Linux and Irix has not been maintained for too long, and
		// the targeted import protocol of XG_DYMOD_IMPORT_MAIN is probabler more secure anyway. It is now the default on
#		define XG_DYMOD_IMPORT_MAIN
#	endif
#endif

/* On some platforms, it is advantageous to use libltdl, part of the libtool tools. If this is not
 \ the intention, define NO_LTDL. When that flag is undefined, the decision to use ltdl or not depends
 \ on the platform, as controlled in dymod.h (USE_LTDL definition).
 \ Generally, it seems there are no advantages to doing this, on the contrary.
 */
#define NO_LTDL

/* uncomment/define when your system has the re_comp() and re_exec() regular expression matching functions.
 \ The xgraph distribution contains its own implementations of these routines that will be substituted for
 \ the system provided ones if you do not define this switch. The xgraph routines also contain a function
 \ xg_re_exec_len(), that compares over a fixed length. For best portability, this should be used in conjunction
 \ with xg_re_comp() [re_comp() will only work if HAVE_RE_COMP is undefined!].
 */
#if !defined(__MACH__) && !defined(__CYGWIN__)
#	define HAVE_RE_COMP
#endif

/* Whether or not we support selecting another visual after the initial startup. The only thing currently
 \ done in support of this is reallocating all colours. More may be necessary, windows may become unmanageable, etc.
 \ So use with caution!
 */
#define RUNTIME_VISUAL_CHANGE

/* a very simple attempt at smart handling of X11 exposure events: */
#define SMART_REFRESH

/* USE_AA_REGISTER: whether or not to include a little C++ module, ascanfcMap.cc, that will allow take_ascanf_address()
 \ and parse_ascanf_address() to maintain a C++ associative array (map, or hash_map) that can be used to avoid
 \ serious parsing errors, like trying to read the type of an ascanf_Function internal structure which resides at
 \ an invalid location (i.e. an invalid pointer).
 \ 1: STL map
 \ 2: __gnu_cxx hash_map implementation using void* cast to unsigned long
 \ 3: __gnu_cxx hash_map implementation "native void*" implementation
 \ 4: an alternative implementation in pure C, based on my old SymbolTable.
 \ 5: a newer, sorted version of that SymbolTable. Seems (marginally) faster in lookups than 4; registering is more expensive.
 \ The exact choice to be made may depend on the platform, and has an influence on timing (performance): therefore,
 \ this variable can be pre-defined in cpu_cycles_per_second.h . Method 5 seems faster in benching.xg, i.e. on highly
 \ repetitive tasks involving relatively few pointers. Otherwise, method 3 is just a bit faster.
 */
#if !defined(USE_AA_REGISTER) || defined(SAFE_FOR_64BIT)
#	undef USE_AA_REGISTER
#	if defined(_AIX)
#		define USE_AA_REGISTER 5
#	else
#		define USE_AA_REGISTER 3
#	endif
#endif

/* Set this to some value (in Mb) if you want to perform size-checking on the size argument passed to
 \ the ALLOCA() macro (defined in xgALLOCA.h; expands to a system-dependent interface to the alloca()
 \ memory allocator). The need for this will depend on how your system handles alloca allocation failures.
 \ The IRIX alloca() routine (and gcc's too) is guaranteed not to fail -- i.e. it always returns a non-null
 \ pointer. If however you ask for some huge amount of memory (e.g. -1 :)), the process will be killed by
 \ the system, without leaving any leeway to the programme to handle the situation gracefully. Also,
 \ utilities like Electric Fence do not trace allocations using alloca-like allocators.
 \ The size checking offered by xgraph does not alter this, but it will print an informative message whenever
 \ more memory is requested, which allows to trace the location where the allocation was made.
 \ It will also likely slow down the programme.
 \ To activate, set CHKLCA_MAX to something (e.g. 128 to warn about allocations larger than 128Mb). Don't
 \ remove the #ifndef pair around the definition; this makes it possible to do file-specific activation.
 */

/* set this directly in xgALLOCA.h ! */

/* =========================================== private section =========================================== */

#define VERSION_MAJOR	2012
#define VERSION_MINOR	05
#define VERSION_PATCHL	08

#define STR(name)	# name
#define STRING(name)	STR(name)

#if !defined(XG_DYMOD_SUPPORT) || !defined(HAVE_FFTW) || HAVE_FFTW==0
#	undef FFTW_DYNAMIC
#endif

#ifdef _MAIN_C

char *XGraph_Compile_Options= "$Options: @(#)"
#ifdef TR_CURVE_LEN
	" TR_CURVE_LEN"
#endif
#if HAVE_FFTW
	" HAVE_FFTW"
#endif
#ifdef FFTW_DYNAMIC
	" FFTW_DYNAMIC"
#endif
#ifdef FFTW_THREADED
	" FFTW_THREADED"
#endif
#ifdef HAVE_VSCANF
	" HAVE_VSCANF"
#endif
#ifdef XG_DYMOD_SUPPORT
	" XG_DYMOD_SUPPORT"
#endif
#ifdef HAVE_RE_COMP
	" HAVE_RE_COMP"
#endif
#ifdef MACH_ABSOLUTE_TIME_FACTOR
	" Lowlevel-clock-cycle=" STRING(MACH_ABSOLUTE_TIME_FACTOR) "ns"
#elif USE_PERFORMANCECOUNTER
	" PerformanceCounter; Lowlevel-clock-freq=" STRING(FFTW_CYCLES_PER_SEC) "Hz"
#elif defined(FFTW_CYCLES_PER_SEC)
	" Lowlevel-clock-freq=" STRING(FFTW_CYCLES_PER_SEC) "Hz"
#endif
#ifdef RUNTIME_VISUAL_CHANGE
	" RUNTIME_VISUAL_CHANGE"
#endif
#ifdef SAFE_FOR_64BIT
	" SAFE_FOR_64BIT"
#endif
#ifdef USE_AA_REGISTER
	" USE_AA_REGISTER=" STRING(USE_AA_REGISTER)
#endif
#ifdef __GNUC__
	" compiler=gcc-" STRING(__GNUC__) "." STRING(__GNUC_MINOR__)
#endif
	" $"
;

#endif

#if !defined(IDENTIFY)

#if defined(i386) || defined(__i386__)
#	define __ARCHITECTURE__	"i386"
#elif defined(__x86_64__) || defined(x86_64) || defined(_LP64)
#	define __ARCHITECTURE__	"x86_64"
#elif defined(__ppc__)
#	define __ARCHITECTURE__	"ppc"
#else
#	define __ARCHITECTURE__	""
#endif

#define XG_IDENTIFY()	"XGraph v" STRING(VERSION_MAJOR) STRING(VERSION_MINOR) STRING(VERSION_PATCHL) " '" __FILE__ "'-[" __DATE__ "," __TIME__ "]"

#ifndef SWITCHES
#	ifdef DEBUG
#		define _IDENTIFY(s,i)	static const char *xg_id_string= "$Id: @(#) "XG_IDENTIFY()"-(\015\013\t\t" s "\015\013\t) DEBUG version" i __ARCHITECTURE__" " " $"
#	else
#		define _IDENTIFY(s,i)	static const char *xg_id_string= "$Id: @(#) "XG_IDENTIFY()"-(\015\013\t\t" s "\015\013\t)" i __ARCHITECTURE__" " " $"
#	endif
#else
  /* SWITCHES contains the compiler name and the switches given to the compiler.	*/
#	define _IDENTIFY(s,i)	static const char *xg_id_string= "$Id: @(#) "XG_IDENTIFY()"-(\015\013\t\t" s "\015\013\t)[" __ARCHITECTURE__" "SWITCHES"]" " $"
#endif

#define __IDENTIFY(s,i)\
static const char *xg_id_string_stub(){ _IDENTIFY(s,i);\
	static char called=0;\
	if( !called){\
		called=1;\
		return(xg_id_string_stub());\
	}\
	else{\
		called= 0;\
		return(xg_id_string);\
	}}

#ifdef __GNUC__
#	define IDENTIFY(s)	__attribute__((used)) __IDENTIFY(s," (gcc" STRING(__GNUC__) ")")
#else
#	define IDENTIFY(s)	__IDENTIFY(s," (cc)")
#endif

#endif

#ifndef HAVE_RE_COMP
#	define re_comp(s)	xg_re_comp(s)
#	define re_exec(s)	xg_re_exec(s)
#endif

#include <sys/types.h>
extern void *_XGrealloc( void** ptr, size_t n, char *name, char *size);
#define XGrealloc(ptr,n)	_XGrealloc( (void**)&(ptr), (n), STRING(ptr), STRING(n) )
extern void *_XGreallocShared( void* ptr, size_t N, size_t oldN, char *name, char *size);
#define XGreallocShared(ptr,N,oldN)	_XGreallocShared( (void*)(ptr), (N), (oldN), STRING(ptr), STRING(n) )
extern void XGfreeShared(void **ptr, size_t N);

#ifdef __cplusplus
}
#endif

#endif
