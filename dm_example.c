#include "config.h"
IDENTIFY( "Sample ascanf library module" );

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
  /* Include the list (table) of ascanf function/callback declarations. This should be an exhaustive
   \ list as it is generated automatically.
   */
#include "ascanfc-table.h"

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   \ On some other systems, XG_DYMOD_IMPORT_MAIN should be defined (see config.h), as they do not export
   \ the main module's symbols to dynamically loaded modules at all.
   \ On those platforms, any needed symbols have to be imported actively. Part of the necessary symbols are
   \ grouped in the DyMod_Interface structure. See below.
   */

  /* Include the interface headerfile.*/
#define DYMOD_MAIN
#include "dymod_interface.h"
  /* Define the DyMod_Interface; this should be exactly as below (i.e. you *must* have a *DMBase). */
static DyMod_Interface DMBaseMem, *DMBase= NULL;

#ifdef XG_DYMOD_IMPORT_MAIN
  /* Any symbols not included in the DMBase should be obtained using (function) pointers and macros, as below: */
int (*ascanf_random_ptr)( ASCB_ARGLIST );
double (*Entier_ptr)(double x);
double (*conv_angle__ptr)( double phi, double base);
double (*arg_ptr)( double x, double y);

  /* Now that we have pointers, the actual code should be 'instructed' to use them. When done as follows, it
   \ should be largely transparent.
   \ NB: dymod_interface.h defines comparable macros for the fields in DMBase -- when XG_DYMOD_IMPORT_MAIN
   \ is set.
   */
#	define ascanf_random	(*ascanf_random_ptr)
#	define Entier	(*Entier_ptr)
#	define conv_angle_	(*conv_angle__ptr)
#	define arg	(*arg_ptr)

#endif

/* routine for the "iran[low,high]" ascanf syntax	*/
int dm_example_irandom ( ASCB_ARGLIST )
{ ASCB_FRAME
  int r;
  int av= ascanf_verbose;
  double rnd;
 
	ascanf_verbose= 1;
  	r= ascanf_random( ASCB_ARGUMENTS );
	rnd= *result;
	*result= Entier( *result );
	if( av ){
		fputc( '\n', StdErr );
	}
	fprintf( StdErr, "#I%d: \tEnt[ ran", *level );
	if( ascanf_arguments ){
	  int i;
		fputc( '[', StdErr );
		fprintf( StdErr, "%s", ad2str( args[0], d3str_format, 0) );
		for( i= 1; i< ascanf_arguments; i++ ){
			fprintf( StdErr, ",%s", ad2str( args[i], d3str_format, 0) );
		}
		fputc( ']', StdErr );
	}
	fprintf( StdErr, "== %s ]== ", ad2str( rnd, d3str_format,0) );
	if( !av ){
		fprintf( StdErr, "%s\n", ad2str( *result, d3str_format,0) );
	}
	ascanf_verbose= av;
	return(r);
}

int dm_example_progn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		return( 1 );
	}
}

#if 0
int dm_example_UnwrapAngles ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af;
  int i, N, w= 0;
  double radix, radix_2;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( !(af= parse_ascanf_address(args[0], _ascanf_array, "dm_example_UnwrapAngles", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		*result= -1;
		return(0);
	}
	if( ascanf_arguments== 1 ){
		radix= M_2PI;
	}
	else{
		radix= (args[1])? args[1] : M_2PI;
	}
	radix_2= radix/ 2;
	N= af->N;
	if( ascanf_SyntaxCheck ){
		*result= 0;
		return(1);
	}
	if( af->iarray ){
	  int aa= 0, da= 0, ca, pa;
		aa= pa= ca= af->iarray[0];
		for( i= 1; i< af->N; i++ ){
			if( (da= (ca= af->iarray[i])- pa) ){
				if( fabs(da)< radix_2 ){
					aa+= da;
				}
				else{
					aa+= conv_angle_(da, radix);
					w+= 1;
				}
			}
			af->value= af->iarray[i]= aa;
			pa= ca;
		}
	}
	else{
	  double aa= 0, da= 0, ca, pa;
		aa= pa= ca= af->array[0];
		for( i= 1; i< af->N; i++ ){
			if( (da= (ca= af->array[i])- pa)!= 0 ){
				if( fabs(da)< radix_2 ){
					aa+= da;
				}
				else{
					aa+= conv_angle_(da, radix);
					if( ascanf_verbose ){
						fprintf( StdErr, " (#%d fabs(%g-%g=%g) >= %g/2=%g: wrapping to %g) ",
							i, ca, pa, ca-pa, radix, radix_2, aa
						);
					}
					w+= 1;
				}
			}
			if( NaNorInf(da) ){
				set_NaN(af->value);
				set_NaN(af->array[i]);
			}
			else{
				af->value= af->array[i]= aa;
			}
			pa= ca;
		}
	}
	af->last_index= af->N-1;
	af->value= ASCANF_ARRAY_ELEM(af,af->last_index);
	if( af->accessHandler ){
		AccessHandler( af, "UnwrapAngles", level, ASCB_COMPILED, AH_EXPR, NULL );
	}
	*result= w;
	return(1);
}
#endif

/* routine for the "arg[x,y]" ascanf syntax	*/
int dm_example_arg ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		if( ascanf_arguments< 4 ){
			args[3]= 0;
		}
		if( args[0]== 0 && args[1]== 0 ){
			set_NaN(*result);
		}
		else{
			*result= args[2] * (arg( args[0], args[1] )/ M_2PI)- args[3];
		}
		return( 1 );
	}
}

static ascanf_Function dm_example_Function[] = {
	{ "iran", dm_example_irandom, 3, NOT_EOF,
		"iran or iran[low,high[,cond]]\n"
		" This function has nothing to do with a certain country. It is the sole function provided by " __FILE__ ",\n"
		" an example dynamic module for use with xgraph. It is equivalent to verbose[Ent[ran[...]]]\n"
	},
	  /* AMAXARGS is actually -1. This signals to ascanfc.c that the current maximum number of arguments
	   \ should be allowed (or required...).
	   */
	{ "dm-progn", dm_example_progn, AMAXARGS, NOT_EOF_OR_RETURN,
		"dm-progn[expr1[,..,expN]]: value set by return[x]\n"
		" A brethren of progn[], but provided by" __FILE__ "\n"
	},
	  /* NB: if we want to provide variables, we should use ascanf_Variable as the associated
	   \ (internal) callback routine. This we can't, as on the XG_DYMOD_IMPORT_MAIN platforms
	   \ that would involve intialising with a not-a-constant. Hence, use NULL temporarily, 
	   \ and a few lines af_initialise() below will install the right value.
	   */
	{ "$dm-variable", NULL, 2, _ascanf_variable, NULL, 1, 0, 0, 0, 0, 0,
		M_PI
	},
#if 0
	{ "test_UnwrapAngles", dm_example_UnwrapAngles, 2, NOT_EOF_OR_RETURN,
		"test_UnwrapAngles[<angles_p>[,radix]]: remove the jumps due to circularity (at e.g. +-180 or 0/360) from an array of consecutive angles\n"
		" Returns -1 upon error, or else the number of wraps made\n"
	},
#endif
	{ "test_arg", dm_example_arg, 4, NOT_EOF_OR_RETURN, "test_arg[x,y[,base[,offset]]] angle to (x,y) in 0..2PI (base), NaN when x=y=0" },
};
static int dm_example_Functions= sizeof(dm_example_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= dm_example_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< dm_example_Functions; i++, af++ ){
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
				  // 20100609: this had XGstrdup(af->label), which of course makes no sense...
				af->label= XGstrdup( label );
			}
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;

  /* This is the crucial function. It is called by xgraph's module loader (and should thus be called as shown here).
   \ It is charged with installing the (ascanf) code we provide into the ascanf function tables (on all platforms).
   \ On other platforms, it is also charged with obtaining the pointers to symbols and functions *we* need. For this
   \ it is passed the initialise argument; see below. It is safe to call this function on all platforms.
   \ 20040502: this, and the closeDyMod() routine may also have their name prepended with the module's "basename"
   \ (e.g. dm_example_initDyMod here). That means of course that the module *must* be installed using that name (plus
   \ a single extension, like .so or .dylib). The reason for this is that sometimes, you'll want to link a module A
   \ with a given other module B that A depends on. This is no replacement for having to load that module B before
   \ loading A, but it prevents the application (xgraph) from aborting if this has not yet been done (as it would on
   \ e.g. linux, as A will have unresolved symbols). When B is linked with A, the loader will give a warning, or even
   \ load B automatically (which will *not* call A's initDyMod routine, of course). So the interest of giving a unique
   \ name to the initDyMod and closeDyMod routines is that braindead linkers won't complain about multiply defined
   \ symbols (these 2 are the only entries that can *not* be made static, as the main application needs to find them).
   */
DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;
	
	if( !DMBase ){
		  /* Use the initialise routine to initialise DMBaseMem. As that
		   \ routine resides in the main programme, it does not need dlsym() to get the things it wants.
		   \ It is important to immediately 'bail out' here, returning DM_Error when anything goes wrong.
		   */
		DMBaseMem.sizeof_DyMod_Interface= sizeof(DyMod_Interface);
		if( !initialise(&DMBaseMem) ){
			fprintf( stderr, "Error attaching to xgraph's main (programme) module\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
		  /* If we were successfull, set DMBase to point to DMBaseMem. This has the advantage that we do not
		   \ need to deallocate DMBase; this will happen automatically when and if we are unloaded. On some
		   \ systems, loaded modules cannot be unloaded (Mac OS X), so there may be a benefit of maintaining
		   \ a valid DMBase around, *should* we somehow get called after an *UNLOAD_MODULE* (theoretically,
		   \ this is very likely not possible).
		   */
		DMBase= &DMBaseMem;
		if( !DyMod_API_Check(DMBase) ){
			fprintf( stderr, "DyMod API version mismatch: either this module or XGraph is newer than the other...\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
#ifdef XG_DYMOD_IMPORT_MAIN
		  /* The XGRAPH_FUNCTION macro can be used to easily initialise the additional variables we need.
		   \ In line with the bail out remark above, this macro returns DM_Error when anything goes wrong -
		   \ i.e. aborts initDyMod!
		   */
		XGRAPH_FUNCTION(ascanf_random_ptr, "ascanf_random");
		XGRAPH_FUNCTION(Entier_ptr, "Entier");
		XGRAPH_FUNCTION(conv_angle__ptr, "conv_angle_");
		XGRAPH_FUNCTION(arg_ptr, "arg");
#endif
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		  /* Perform all initialisations that are necessary. */
		af_initialise( theDyMod, theDyMod->name );
		  /* And now add the functions we provide! */
		add_ascanf_functions( dm_example_Function, dm_example_Functions, "dm_example::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-ex" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" An example dynamic module (library) that contains\n"
		" some initialisation and ascanf routines.\n"
	);
	return( DM_Ascanf );
}

// see the explanation printed by wrong_dymod_loaded().
// When loading a module FOO, Python will try to find a function called initFOO to do the necessary initialisation.
// If by chance it loads us instead of the intended library, we'd better print a clear warning about what's happening.
// wrong_dymod_loaded() is defined in dymod_interface.h if DYMOD_MAIN has been defined (only a single copy is needed per
// dynamic module).
void initdm_example()
{
	wrong_dymod_loaded( "initdm_example()", "Python", "dm_example.so" );
}

// see the explanation printed by wrong_dymod_loaded():
// R looks for a function called R_init_FOO when loading a library/package FOO
void R_init_dm_example()
{
	wrong_dymod_loaded( "R_init_dm_example()", "R", "dm_example.so" );
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
	  int r= remove_ascanf_functions( dm_example_Function, dm_example_Functions, force );
		if( force || r> 0 ){
			for( i= 0; i< dm_example_Functions; i++ ){
				dm_example_Function[i].dymod= NULL;
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
			  /* This would be the place to deallocate a dynamically allocated DMBase variable. In that case,
			   \ we'd also have to move the notification message inside the test for initialised!.
			   */
		}
		else{
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() returns %d)\n", r );
		}
	}
	return(ret);
}

/* _init() and _fini() are called at first initialisation, and final unloading respectively. This works under linux, and
 \ maybe solaris - not under Irix 6.3. It also requires that the -nostdlib flag is passed to gcc.
 \ NB: These are example invocations. Care should be taken to use stderr and not StdErr, as XGRAPH_ATTACH will not yet
 \ have been called.
 */
void __attribute__((constructor)) dm_example_init()
{ static int called= 0;
	fprintf( stderr, "%s::_init(): call #%d\n", __FILE__, ++called );
}

void __attribute__((destructor)) dm_example_fini()
{ static int called= 0;
	fprintf( stderr, "%s::_fini(): call #%d\n", __FILE__, ++called );
}

