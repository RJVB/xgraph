#ifndef _DYMOD_H
#define _DYMOD_H

/* 20031129: NB: this headerfile must be included before the xgraph headerfiles; the 'error' definition in DataSet.h
 \ can cause troubles otherwise.
 */

#include "config.h"

#ifndef NO_LTDL
#	if defined(linux)
#		define USE_LTDL
#	endif
#endif

#ifdef USE_LTDL
#	include <ltdl.h>
#	ifndef RTLD_LAZY
#		define RTLD_LAZY	0
#	endif
#	ifndef RTLD_GLOBAL
#		define RTLD_GLOBAL	0
#	endif
#	ifndef RTLD_NOW
#		define RTLD_NOW	0
#	endif
#else
#	include <dlfcn.h>
#endif
#include "ascanf.h"

#ifdef __cplusplus
	extern "C" {
#endif

typedef enum DyModTypes{ DM_Error= 0, DM_Unloaded, DM_FUnloaded, DM_Ascanf, DM_Support, DM_IO, DM_Python, DM_Unknown, DM_Types } DyModTypes;
extern char *DyModType[DM_Types];

typedef struct DyModDependency{
	struct DyModLists *dymod;
	  /* a pointer to the Compiled_Form* frame of the toplevel expression that depends on dymod: */
	struct Compiled_Form *caf_client;
	  /* 20031129: use the 'forward' forms, i.e. enum ascanf_Function_type instead of ascanf_type and
	   \ struct ascanf_Function instead of ascanf_Function.
	   */
	enum ascanf_Function_type caf_type;
	  /* pointers to the ascanf_Function entries associated with the toplevel frame, and with the current frame
	   \ (should be the depending function itself!)
	   */
	struct ascanf_Function *caf_af, *af;
#ifdef DEBUG
	char *caf_expr, *expr;
#endif
	struct Compiled_Form *form;
	struct DyModDependency *cdr;
} DyModDependency;

struct DyMod_Interface;

typedef struct DyModLists{
	  /* the name as specified by the user, and the pathname (can be identical to name) */
	char *name, *path;
	  /* flags: the flags passed to dlopen(): */
	int flags, auto_loaded, auto_unload, no_dump;
	  /* The handle returned by dlopen() */
	void *handle;
	  /* Dynamic (ELF) libraries can specify their own _init() and _fini() initialisation and
	   \ termination routines. Here, we provide another mechanism, the function initDyMod()
	   \ and closeDyMod() that can initialise/free a pointer to point to user data, and that
	   \ that can return a type specification.
	   */
	DyModTypes type;
	char *libname, *typestring, *description, *buildstring, *loaded4;
	void *libHook;
	DyModTypes (*initDyMod)( struct DyModLists *dymod, struct DyMod_Interface* (*initialise)(struct DyMod_Interface *base) ),
		(*closeDyMod)( struct DyModLists *dymod, int force );
	DyModDependency *Dependencies;
	  /* Pointers to the previous and next entry in the FILO linked list.
	   \ NB: the car is not maintained automatically by the load/unload routines, but is provided
	   \ only as a means to make a FIFO list when needed (e.g. to reconstruct the exact loading
	   \ order)!!
	   */
	struct DyModLists *car, *cdr,
		  /* 20060920: in case only a single version of a library can be loaded, the initDyMod() routine
		   \ can return DM_Error and set the already_loaded_version field to the DyModLists entry of the
		   \ version that ought to be used.
		   */
		*already_loaded_version;
} DyModLists;

extern DyModLists *DyModList;
extern int DyModsLoaded;

typedef struct DyModAutoLoadTables{
	char *functionName, *DyModName;
	int flags;					/* typically RTLD_LAZY|RTLD_GLOBAL */
	int warned, loaded;				/* if things go wrong; warn only once. */
	long hash;
	DyModLists *dymod;
} DyModAutoLoadTables;

extern void *XG_dlopen( char **name, int flags, char **error_mesg );
extern DyModLists *LoadDyMod( char *name, int flags, int no_dump, int auto_unload );
extern int UnloadDyMod( char *name, int *load_count, int force );
extern int UnloadDyMods();
extern int Auto_UnloadDyMods();
extern int Check_Doubles_Ascanf( struct ascanf_Function *af, char *label, int warn );
extern int Add_DyMod_Dependency(
	DyModLists *module, void *caf_client, struct ascanf_Function *caf_af, struct ascanf_Function *af,
	void *form
);
extern int Delete_DyMod_Dependency( DyModLists *module, void *caf_client );
extern int Delete_DyMod_Dependencies( struct Compiled_Form *form, DyModLists *module, char *caller );
  /* call with fname == NULL to initialise the table's hash values: */
extern int Auto_LoadDyMod( DyModAutoLoadTables *table, int N, char *fname );
extern int Auto_LoadDyMod_LastPtr( DyModAutoLoadTables *table, int N, char *fname, DyModLists **last );
extern DyModAutoLoadTables *Add_LoadDyMod( DyModAutoLoadTables *target, int *target_len, DyModAutoLoadTables *source, int n );

  /* 20090113: check for the presence of the Python opcode py:: or python::, load Python is not yet done
   \ and return a pointer to the part of the string after the opcode.
   */
extern char *PyOpcode_Check( char *expr );

  /* This flag should signal the fact that modules were forcedly unloaded when still in use */
extern int Unloaded_Used_Modules;

#if defined(__APPLE_CC__) && defined(NEEDS_XGDLSYM)

#include "xgALLOCA.h"

static void *XGdlsym(void * handle, const char *symbol)
{ void *ptr= dlsym(handle, symbol);
	if( !ptr ){
	  /* dangerous, but we can't touch dlerror() here. */
	  __ALLOCA(news, char, strlen(symbol)+2, nlen);
		strcpy( news, "_" );
		strcat( news, symbol );
		ptr= dlsym(handle, news);
		__GCA();
	}
	return(ptr);
}

#	define dlsym(lib,name)	XGdlsym(lib,name)
#endif

  /* A convenience macro that will initialise a function pointer with a reference from the
   \ specified library. NB: no type checking is done!!
   */
#ifdef USE_LTDL
#	define LOADFUNCTION(lib,ptr,name)	{ const char *c; void **p = &((void*)ptr); *p= lt_dlsym(lib, (const char*) name);\
	if( (c= lt_dlerror()) ){ \
		err+= 1; \
		fprintf( StdErr, "Error retrieving %s::%s: %s\n", \
			STRING(lib), name, c \
		); \
	} }
#else
#	define LOADFUNCTION(lib,ptr,name)	{ const char *c; void **p = (void**)&(ptr); *p= dlsym(lib, name);\
	if( (c= dlerror()) ){ \
		err+= 1; \
		fprintf( StdErr, "Error retrieving %s::%s: %s\n", \
			STRING(lib), name, c \
		); \
	} }
#endif

#define INIT_DYMOD_ARGUMENTS	DyModLists *theDyMod, struct DyMod_Interface* (*initialise)(DyMod_Interface *)

#ifdef __cplusplus
}
#endif

#endif
