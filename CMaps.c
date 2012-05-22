#include "config.h"
IDENTIFY( "Utils ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif


#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>

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

#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;

static ascanf_Function CMaps_Function[] = {
};
static int CMaps_Functions= sizeof(CMaps_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= CMaps_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< CMaps_Functions; i++, af++ ){
		if( !called ){
			if( af->name ){
				af->name= XGstrdup( af->name );
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
  char fname[2*MAXPATHLEN];
  FILE *fp;
  DyModTypes ret= DM_Ascanf;

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

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( CMaps_Function, CMaps_Functions, "CMaps::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-CMaps" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" various ascanf ColourMap functions. It only\n"
#if defined(__APPLE_CC__) || defined(__MACH__)
		" includes CMaps.xg or ~/Library/xgraph/CMaps.xg\n"
#else
		" includes CMaps.xg or ~/.Preferences/.xgraph/CMaps.xg\n"
#endif
		" at each invocation; the definitions in this\n"
		" file are not deleted when the library is un-\n"
		" loaded.\n"
	);
	if( !(fp= fopen("CMaps.xg", "r")) ){
/* 		tildeExpand( fname, "~/.Preferences/.xgraph/CMaps.xg" );	*/
		snprintf( fname, sizeof(fname), "%s/CMaps.xg", PrefsDir );
		fp= fopen(fname, "r");
	}
	else{
		strcpy( fname, "CMaps.xg");
	}
	if( fp ){
		IncludeFile( ActiveWin, fp, fname, True, NULL );
		fclose(fp);
	}
	else{
		fprintf( StdErr, "Failure: Can't open neither 'CMaps.xg' nor '%s/CMaps.xg' (%s)\n", PrefsDir, serror() );
		ret= DM_Error;
	}
	return( ret );
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
		if( CMaps_Functions ){
		  int r= remove_ascanf_functions( CMaps_Function, CMaps_Functions, force );
			if( force || r== CMaps_Functions ){
				for( i= 0; i< CMaps_Functions; i++ ){
					CMaps_Function[i].dymod= NULL;
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
					r, CMaps_Functions
				);
			}
		}
		else{
			ret= DM_Unloaded;
		}
	}
	return(ret);
}
