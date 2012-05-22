#include "config.h"
IDENTIFY( "Constants ascanf library module; various system-defined floating point constants" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif


#include <stdio.h>
#include <stdlib.h>

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

extern FILE *StdErr;

#include "copyright.h"

  /* Include a whole bunch of headerfiles. Not all of them are strictly necessary, but if
   \ we want to have fdecl.h to know all functions we possibly might want to call, this
   \ list is needed.
   */
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
  /* If we want to be able to access the "expression" field in the callback argument, we need compiled_ascanf.h .
   \ If we don't include it, we will just get 'theExpr= NULL' ... (this applies to when -DDEBUG)
   */
#include "compiled_ascanf.h"
#include "ascanfc-table.h"

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   \ On some other systems, XG_DYMOD_IMPORT_MAIN should be defined (see config.h).
   */

#define DYMOD_MAIN
#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;

#include <float.h>

static ascanf_Function constants_Function[] = {
	{ "$FLT_RADIX", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		FLT_RADIX
	},
	{ "$FLT_ROUNDS", NULL, 2, _ascanf_variable, 
		"$FLT_ROUNDS: Addition rounds to 0: zero, 1: nearest, 2: +inf, 3: -inf, -1: unknown", 1, 0, 0, 0, 0, 0,
		0
	},
	{ "$DBL_MANT_DIG", NULL, 2, _ascanf_variable, 
		"$DBL_MANT_DIG: Number of base-FLT_RADIX digits in the significand of a double ", 1, 0, 0, 0, 0, 0,
		DBL_MANT_DIG
	},
	{ "$DBL_DIG", NULL, 2, _ascanf_variable, 
		"$DBL_DIG: Number of decimal digits of precision in a double", 1, 0, 0, 0, 0, 0,
		DBL_DIG
	},
	{ "$DBL_EPSILON", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		DBL_EPSILON
	},
	{ "$DBL_MIN_EXP", NULL, 2, _ascanf_variable, 
		"$DBL_MIN_EXP: Minimum int x such that FLT_RADIX**(x-1) is a normalised double", 1, 0, 0, 0, 0, 0,
		DBL_MIN_EXP
	},
	{ "$DBL_MIN", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		DBL_MIN
	},
	{ "$DBL_MIN_10_EXP", NULL, 2, _ascanf_variable, 
		"$DBL_MIN_10_EXP: Minimum int x such that 10**x is a normalised double", 1, 0, 0, 0, 0, 0,
		DBL_MIN_10_EXP
	},
	{ "$DBL_MAX_EXP", NULL, 2, _ascanf_variable, 
		"$DBL_MAX_EXP: Maximum int x such that FLT_RADIX**(x-1) is a representable double", 1, 0, 0, 0, 0, 0,
		DBL_MAX_EXP
	},
	{ "$DBL_MAX", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		DBL_MAX
	},
	{ "$DBL_MAX_10_EXP", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		DBL_MAX_10_EXP
	},
	{ "$DBL_MIN_EXP", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		DBL_MIN_EXP
	},
};
static int constants_Functions= sizeof(constants_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= constants_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< constants_Functions; i++, af++ ){
		if( !called ){
			if( af->name ){
				af->name= XGstrdup( af->name );
				  /* FLT_ROUNDS is a not-a-constant on some platforms: */
				if( strcmp(af->name, "$FLT_ROUNDS")== 0 ){
					af->value= FLT_ROUNDS;
				}
			}
			else{
				sprintf( buf, "Function-%d", i );
				af->name= XGstrdup( buf );
			}
			if( af->usage ){
				af->usage= XGstrdup( af->usage );
			}
			ascanf_CheckFunction(af);
			if( af->function!= ascanf_Variable ){
				set_NaN(af->value);
			}
			if( label ){
				af->label= XGstrdup( label );
			}
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;

	if( !DMBase ){
		DMBaseMem.sizeof_DyMod_Interface= sizeof(DyMod_Interface);
		if( !initialise(&DMBaseMem) ){
			fprintf( stderr, "Error attaching to xgraph's main (programme) module\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
		DMBase= &DMBaseMem;
		if( !DyMod_API_Check(DMBase) ){
			fprintf( stderr, "DyMod API version mismatch: either this module or XGraph is newer than the other...\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
	}

	  /* NB: XGRAPH_ATTACH() is the first thing we ought to do, but only when !intialised ! */
	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__,
		theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( constants_Function, constants_Functions, "constants::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-constants" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that contains\n"
		" hooks to various system-defined floating\n"
		" point constants.\n"
	);
	return( DM_Ascanf );
}

void initconstants()
{
	wrong_dymod_loaded( "initconstants()", "Python", "constants.so" );
}

void R_init_constants()
{
	wrong_dymod_loaded( "R_init_constants()", "R", "constants.so" );
}

/* The close handler. We can be called with the force flag set to True or False. True means force
 \ the unload, e.g. when exitting the programme. In that case, we are supposed not to care about
 \ whether or not there are ascanf entries still in use. In the alternative case, we *are* supposed
 \ to care, and thus we should heed remove_ascanf_function()'s return value. And not take any
 \ action when it indicates variables are in use (or any other error). Return DM_Unloaded when the
 \ module was de-initialised, DM_Error otherwise (in that case, the module will remain opened).
 */
int closeDyMod( DyModLists *target, int force )
{ static int called= 0;
  int i;
  DyModTypes ret= DM_Error;
  FILE *SE= (initialised)? StdErr : stderr;
	fprintf( SE, "%s::closeDyMod(%d): Closing %s loaded from %s, call %d", __FILE__,
		force, target->name, target->path, ++called
	);
	if( target->loaded4 ){
		fprintf( SE, "; auto-loaded because of \"%s\"", target->loaded4 );
	}
	if( initialised ){
	  int r= remove_ascanf_functions( constants_Function, constants_Functions, force );
		if( force || r== constants_Functions ){
			for( i= 0; i< constants_Functions; i++ ){
				constants_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			ret= target->type= DM_Unloaded;
			if( r<= 0 || ascanf_emsg ){
				fprintf( SE, " -- warning: variables are in use (remove_ascanf_functions() returns %d,\"%s\")",
					r, (ascanf_emsg)? ascanf_emsg : "??"
				);
				Unloaded_Used_Modules+= 1;
				if( force ){
					ret= target->type= DM_FUnloaded;
				}
			}
			fputc( '\n', SE );
		}
		else{
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() returns %d out of %d)\n",
				r, constants_Functions
			);
		}
	}
	return(ret);
}
