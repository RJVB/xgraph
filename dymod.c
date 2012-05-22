#include "config.h"
IDENTIFY( "Dynamic library loading support" );

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__MACH__)
#	include <libgen.h>
#endif

#include "dymod.h"
#define _DYMOD_C
#include "dymod_interface.h"

#include "xgout.h"
#include "xgraph.h"

#include "ascanf.h"
#include "compiled_ascanf.h"

#include "fdecl.h"

#include "Python/PythonInterface.h"

DyModLists *DyModList= NULL;
int DyModsLoaded= 0;

DyModLists *dm_pythonLib= NULL;
DM_Python_Interface *dm_python= NULL;

extern char *tildeExpand(), *XGstrdup();
extern FILE *StdErr;
extern int debugFlag, scriptVerbose;

#include "xfree.h"

char *DyModType[DM_Types]= { "Error", "Unloaded", "Forced-unloaded", "Ascanf library", "Support Library", "IO Library", "Python library", "Unknown library" };

#ifdef __GNUC__
inline
#endif
char *DyModTypeString( DyModTypes type )
{
	return( (type>=DM_Error && type< DM_Types)? DyModType[type] : "<unknown returncode>" );
}

#ifdef USE_LTDL
static int lt_dlinited= False, lt_dlsearchpath= False;
#endif

char* XG_dlopen_findlib(char *name)
{
#ifdef USE_LTDL
		if( !DyModList && !lt_dlinited ){
			if( !lt_dlinit() ){
				lt_dlinited= True;
			}
			else{
				fprintf( StdErr, "Failure: error calling lt_dlinit() (%s)\n", lt_dlerror() );
				xfree(name);
				return(NULL);
			}
		}
		if( lt_dlinited && !lt_dlsearchpath ){
		  char *pname= concat( ".:", PrefsDir, NULL );
			if( lt_dlsetsearchpath(pname) ){
				fprintf( StdErr, "Warning: can't prepend '%s' to lt_dlopen search path (%s)\n",
					pname, lt_dlerror()
				);
			}
			else{
				lt_dlsearchpath= True;
			}
			xfree(pname);
		}
#else
		if( name[0]!= '/' && !(name[0]== '.' && (name[1]== '/' || (name[1]== '.' && name[2]== '/'))) ){
		  struct stat Stat;
		  char *aname= concat( PrefsDir, "/", name, NULL );
			if( stat( name, &Stat) ){
				if( stat( aname, &Stat )== 0 ){
					  /* There is a library with the requested name in the preferences directory. */
					xfree(name);
					name= aname;
				}
				else{
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr,
							"Warning: \"%s\" does not exist, nor does \"%s\" - let's hope dlopen() will find something!\n",
							name, aname
						);
					}
					xfree(aname);
				}
			}
			else{
			  /* We need to put a './' in front of name in order to be sure dlopen() will find it. This
			   \ is not necessary when LD_LIBRARY_PATH includes '.', but one can't be sure of that - nor
			   \ of the exact name of the env. variable on different platforms. The trick here should work
			   \ everywhere.
			   */
				xfree(aname);
				aname= concat( "./", name, NULL );
				xfree(name);
				name= aname;
			}
		}
#endif
		return(name);
}

void *XG_dlopen_now( char *name, int flags, char **error_mesg )
{ void *handle;
  const char *c;
#ifdef RTLD_LOCAL
	if( !CheckMask(flags, RTLD_LOCAL) )
#endif
			flags|= RTLD_GLOBAL;

	if( debugFlag ){
		fprintf( StdErr, "dlopen(%s,0x%lx)\n", name, flags );
	}
#ifdef USE_LTDL
	{ char *so= &name[ strlen(name)-3 ];
		if( strcmp(so, ".so")== 0 ){
			*so= '\0';
		}
		else{
			so= NULL;
		}
ltdlopen:;
		handle= lt_dlopenext(name);
		if( (c= lt_dlerror()) ){
			fprintf( StdErr, "Warning: lt_dlopenext() didn't find '%s' (%s); trying lt_dlopen()\n",
				name, c
			);
			handle= lt_dlopen(name);
			c= lt_dlerror();
		}
		if( so ){
			*so= '.';
			so= NULL;
			if( c || !handle ){
				goto ltdlopen;
			}
		}
	}
#else
	handle= dlopen( name, flags );
	c= dlerror();
#endif
	if( error_mesg ){
		*error_mesg= (char*) c;
	}
	return( handle );
}

void *XG_dlopen( char **name, int flags, char **error_mesg )
{
	if( (*name= XG_dlopen_findlib(*name)) ){
		return( XG_dlopen_now(*name, flags, error_mesg) );
	}
	else{
		return( NULL );
	}
}

void *XG_dlopen_now_ext( char **Name, int flags, char **error_mesg )
{ void *handle= NULL;
  char *name= *Name, *bname= NULL, *ext[]= { ".dylib", ".so", ".dll", NULL };
  int i= 0;
	if( name ){
		bname= XGstrdup(name);
	}
	while( name && !handle ){
		name= XG_dlopen_findlib(name);
		if( !(handle= XG_dlopen_now( name, flags, error_mesg)) ){
			if( ext[i] ){
				xfree(name);
				name= concat( bname, ext[i], NULL );
				*Name= name;
				i++;
			}
			else{
				name= NULL;
			}
		}
		else{
			*Name= name;
		}
	}
	xfree(bname);
	return(handle);
}

void *DM_dlsym( void *dmlib, char *name, char *dm_name, char **err_rtn )
{ const char *c;
  void *ptr;
#ifdef USE_LTDL
	ptr= lt_dlsym( dmlib, name );
	c= lt_dlerror();
#else
	ptr= dlsym( dmlib, name);
	c= dlerror();
#endif
	if( c ){
	  char *dname= strdup( basename(dm_name) ), *d;
	  char *ename;
	  int warn= (strcmp(name, "initDyMod") && strcmp(name, "closeDyMod"));
		if( warn ){
			fprintf( StdErr, "Error retrieving address for symbol %s::%s(): %s\n",
				dm_name, name, c
			);
		}
		if( dname ){
			  /* remove extension from library name */
			if( (d= rindex(dname, '.')) ) {
				*d= '\0';
			}
			if( (ename= concat( dname, "_", name, NULL)) ){
				if( warn ){
					fprintf( StdErr, "\tTrying with prepended module name (%s).\n", ename );
				}
#ifdef USE_LTDL
				ptr= lt_dlsym( dmlib, ename );
				c= lt_dlerror();
#else
				ptr= dlsym( dmlib, ename);
				c= dlerror();
#endif
				xfree(ename);
			}
			xfree(dname);
		}
	}
	if( err_rtn ){
		*err_rtn= c;
	}
	return(ptr);
}

DyModLists *LoadDyMod( char *Name, int flags, int no_dump, int auto_unload )
{
#ifdef XG_DYMOD_SUPPORT
  DyModLists *new= NULL;
  char *name= NULL;
	if( Name && *Name ){
		  /* expand a potentially present first '~', and create a copy that is stored
		   \ in name:
		   */
		name= tildeExpand( NULL, Name );
	}
	else{
		return( NULL );
	}
	if( name && *name ){
/* 		name= XG_dlopen_findlib(name);	*/
		if( name && *name ){
		  DyModLists *current= DyModList;
			while( current ){
				if( strcmp( Name, current->name)== 0 ){
					  /* 20020519: We shouldn't of course allow DyMods to be loaded twice. Since this would change the
					   \ registration (set to the DyModLists entry, unique for each loaded instance), and thus render
					   \ the registry of the Compiled_Forms compiled with earlier instances useless.
					   */
					fprintf( StdErr, "LoadDyMod(%s): request ignored because a module with this name has already been loaded.\n"
						"\tUnload that module (%s from %s) first, and try again.\n",
						Name, current->name, current->path
					);
					xfree(name);
					return(current);
				}
				current= current->cdr;
			}
			if( (new= calloc( 1, sizeof(DyModLists))) ){
			  char *c;
				new->handle= XG_dlopen_now_ext( &name, flags, &c );
				if( new->handle ){
					new->name= strdup(Name);
					new->path= name;
					new->flags= flags;
					new->auto_unload= auto_unload;
					new->no_dump= no_dump;
					if( DyModList ){
						new->cdr= DyModList;
					}
					DyModList= new;
					DyModsLoaded+= 1;
					new->closeDyMod= DM_dlsym( new->handle, "closeDyMod", Name, &c);
					if( new->closeDyMod ){
						if( c ){
							fprintf( StdErr, "Error retrieving address for closing routine %s::closeDyMod(): %s\n",
								Name, c
							);
							new->closeDyMod= NULL;
						}
					}
					else{
						if( c ){
							fprintf( StdErr, "Warning: no closing routine %s::closeDyMod(): %s\n",
								Name, c
							);
						}
					}
					new->initDyMod= DM_dlsym( new->handle, "initDyMod", Name, &c);
					if( new->initDyMod ){
						if( !c ){
							new->type= (*new->initDyMod)( new, Init_DyMod_Interface );
						}
						else{
							fprintf( StdErr, "Error retrieving address for initialisation routine %s::initDyMod(): %s\n",
								Name, c
							);
							new->initDyMod= NULL;
							new->type= DM_Unknown;
						}
					}
					else{
						if( c ){
							fprintf( StdErr, "Warning: no initialisation routine %s::initDyMod(): %s\n",
								Name, c
							);
						}
						new->type= DM_Unknown;
					}
					if( new->type== DM_Error ){
					  int c;
						if( new->already_loaded_version ){
							new= new->already_loaded_version;
						}
						else{
						  /* 20060920: should return NULL upon a generic loading error?! ... */
							new= NULL;
						}
						UnloadDyMod( Name, &c, True );
					}
					else{
						if( new->type== DM_Python && ((DM_Python_Interface*)new->libHook)->type== DM_Python ){
							dm_pythonLib= new;
							dm_python= new->libHook;
						}
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "Loaded module %s from %s, total modules %d\n", Name, name, DyModsLoaded );
							if( new->buildstring ){
								fprintf( StdErr, "\t(%s)\n", new->buildstring );
							}
							if( new->description ){
								fprintf( StdErr, new->description );
							}
						}
					}
				}
				else{
					fprintf( StdErr, "Error: can't load module " ); fflush( StdErr );
					fprintf( StdErr, "\"%s\" (last tried \"%s\") (%s)(%s)\n", Name, name, serror(), (c)? c : "?" );
					xfree(new);
					xfree(name);
				}
			}
		}
	}
	return( new );
#else
	fprintf( StdErr, "Error: dynamic modules not supported (compile with XG_DYMOD_SUPPORT defined)\n" );
	return(NULL);
#endif
}

static int ascanf_noop ( ASCB_ARGLIST )
{ extern char *ascanf_emsg;
	ascanf_emsg= " (calling remnant of a library ascanf function) ";
	ascanf_arg_error= 1;
	return( 0 );
}

static ascanf_Function noop = { "Removed Ascanf Library Function", ascanf_noop, AMAXARGS, NOT_EOF_OR_RETURN,
		"This is a remnant of a function that was provided by an Ascanf Library, now unloaded.\n"
	};

/* For all nodes attached to the Compiled_Form expression node <form> (which claims
 \ to depend on <module>), find those that reference an ascanf_Function that depends
 \ on that module. These references are replaced by references to the static "noop"
 \ function, such that after unloading <module>, attempts to access the functions do
 \ not result in illegal behaviour.
 */
int Form_Remove_DyMod_Functions( Compiled_Form *form, DyModLists *module )
{ int n= 0;
	while( form ){
		  /* 20020519: a bit more extensive search for nodes depending on the module being unloaded. Because
		   \ if the user managed to load a module twice or more, those instances will have different dymod
		   \ structures, but will (hopefully) share a same dymod->handle. When the one instance has been unloaded,
		   \ the checking via form->dymod will become necessary if the library's closing handler unsets the fun->dymod
		   \ field.
		   */
		if( (form->fun && (form->fun->dymod== module || (form->fun->dymod && form->fun->dymod->handle== module->handle))) ||
			(form->dymod && (form->dymod== module || form->dymod->handle== module->handle))
		){
			form->fun= NULL;
			form->fun= &noop;
			form->fun->links= MAXINT;
		}
		n+= Form_Remove_DyMod_Functions( form->args, module )+ 1;
		form= form->cdr;
	}
	return(n);
}

int Unloaded_Used_Modules= 0;

extern Time_Struct AscanfCompileTimer;
extern SimpleStats SS_AscanfCompileTime;

int UnloadDyMod( char *Name, int *load_count, int force )
{
#ifdef XG_DYMOD_SUPPORT
  DyModLists *target= DyModList, *car= DyModList, *free_target= NULL;
  int unloaded= 0, samelib= False;
  char *libPath= NULL;
	if( Name && *Name ){
		*load_count= 0;
		while( target ){
			  /* 20040619: when unloading a library, unload others with the same name, but also
			   \ and especially, with the same path (a much stricter indicator of the same library!!)
			   */
			if( strcmp( Name, target->name)== 0 ||
				( libPath && (samelib= (strcmp(libPath, target->path)== 0)) )
			){
			  int dl, aul= target->auto_unload;
			  char *c;
			  DyModTypes ttype= target->type;
				target->auto_unload= False;
				(*load_count)+= 1;
				if( force && target->Dependencies ){
				  DyModDependency *list= target->Dependencies, *cdr;
				  ascanf_Function *af;
				  int n= 0, m= 0;
					Elapsed_Since(&AscanfCompileTimer, True);
					fprintf( StdErr, "Deleting ascanf procedures depending on \"%s\": ",
						target->name
					);
					while( list ){
						cdr= list->cdr;
						if( list->caf_client && list->caf_type== _ascanf_procedure &&
								(af= Procedure_From_Code( list->caf_client))
						){
							n+= 1;
/* 								if( ascanf_verbose ){	*/
									fprintf( StdErr, " (deleted %s A=%d R=%d L=%d) ",
										af->name, af->assigns,
										af->reads, af->links
									);
									fflush( StdErr );
/* 								}	*/
							if( af->internal ){
								Delete_Internal_Variable( NULL, af );
							}
							else{
								Delete_Variable( af );
							}
						}
						list= cdr;
					}
					if( n /* && ascanf_verbose */ ){
						fputc( '\n', StdErr );
					}
					list= target->Dependencies;
					n= 0;
					while( list ){
						cdr= list->cdr;
						{ Compiled_Form *form= list->form, *caf_client= list->caf_client;
						  int free_list= True;
							n+= Form_Remove_DyMod_Functions( form, target );
							if( form && form->DyMod_DependList && form->DyMod_DependList->cdr!= cdr ){
								m+= Delete_DyMod_Dependencies( form, target, "UnloadDyMod" );
								free_list= False;
							}
							n+= Form_Remove_DyMod_Functions( caf_client, target );
							if( caf_client && caf_client->DyMod_DependList && caf_client->DyMod_DependList->cdr!= cdr ){
								m+= Delete_DyMod_Dependencies( caf_client, target, "UnloadDyMod" );
								free_list= False;
							}
							if( free_list ){
								xfree( list );
							}
						}
						list= cdr;
					}
					if( n || m ){
						Elapsed_Since( &AscanfCompileTimer, False );
						SS_Add_Data_( SS_AscanfCompileTime, 1, AscanfCompileTimer.HRTot_T, 1.0 );
						fprintf( StdErr,
							"Removed references to %s functions from %d compiled expression nodes, and\n"
							" unregistered %d dependencies.\n",
							target->name, n, m
						);
					}
					target->Dependencies= NULL;
				}
				  /* 20040619: we should prevent calling a closeDyMod() routine in an unloaded library!! */
				  /* 20070201: and therefore, we ought to unset closeDyMod() after unloading! */
				if( !samelib && target->closeDyMod ){
				  DyModTypes ret;
					ret= (*target->closeDyMod)( target, force );
					if( force ){
						if( ret!= DM_Unloaded ){
							fprintf( StdErr,
								"Warning: closing the module while its close handler would have refused (returned %d=%s)!\n",
								ret,
								(target->typestring)? target->typestring : DyModTypeString(ret)
							);
							Unloaded_Used_Modules+= 1;
						}
/* 						target->type= ret;	*/
						target->type= DM_FUnloaded;
						target->closeDyMod= NULL;
						xfree(target->loaded4);
						target->loaded4= NULL;
						unloaded+= 1;
					}
					else if( ret== DM_Unloaded ){
						target->type= ret;
						target->closeDyMod= NULL;
						xfree(target->loaded4);
						target->loaded4= NULL;
						unloaded+= 1;
					}
					else{
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "Unloading %s from %s refused by the module's close handler.\n",
								target->name, target->path
							);
						}
						target->auto_unload= aul;
						goto unload_next_entry;
					}
				}
				else{
					target->type= (force)? DM_FUnloaded : DM_Unloaded;
					target->closeDyMod= NULL;
					xfree(target->loaded4);
					target->loaded4= NULL;
					unloaded+= 1;
				}
				if( ttype== DM_Python && dm_pythonLib== target ){
					dm_pythonLib= NULL;
					dm_python= NULL;
				}
#ifdef USE_LTDL
				dl= lt_dlclose( target->handle );
				c= lt_dlerror();
#else
				dl= dlclose( target->handle );
				c= (char*)dlerror();
#endif
				if( c ){
					fprintf( StdErr, "Error unloading %s::%s (%s)\n", target->name, target->path, c );
				}
				else if( debugFlag || scriptVerbose ){
					fprintf( StdErr, "Unloaded %s from %s; total %d\n", target->name, target->path, DyModsLoaded-1 );
				}
// 20070201: I actually don't see why one wouldn't free the name argument in all cases...
// 				if( Name!= target->name ){
					xfree( target->name );
// 				}
// 				else{
// 					free_target= target;
// 				}
				if( libPath && target->path!= libPath ){
					xfree( target->path );
				}
				else{
					libPath= target->path;
					target->path= NULL;
				}
				  /* 20070201: it *might* be a good idea to unset the handle here. But I may have had good reasons not to
				   \ in the past, so I'll leave it for the time being.
				   */
				if( target== DyModList ){
					DyModList= target->cdr;
				}
				else{
					car->cdr= target->cdr;
				}
				{ DyModLists *dunk= target;
					target= target->cdr;
					if( 0 ){
						  /* 20020517: actually, maybe it's better to not free the DyModList entry. It is quite
						   \ likely (given practical experiences) that there are still Compiled_Form's around
						   \ that reference it.
						   */
						xfree(dunk);
					}
				}
				DyModsLoaded-= 1;
			}
			else{
unload_next_entry:;
				  /* 20070215: don't set the car to an unloaded module */
				if( target->type!= DM_Unloaded && target->type!= DM_FUnloaded ){
					car= target;
				}
				target= target->cdr;
			}
		}
		xfree(libPath);
	}
#ifdef USE_LTDL
	if( !DyModList && lt_dlinited ){
		if( lt_dlexit() ){
			fprintf( StdErr, "Failure calling lt_dlexit(): %s\n", lt_dlerror() );
		}
		else{
			lt_dlinited= False;
			lt_dlsearchpath= False;
		}
	}
#endif
	if( free_target ){
		xfree( free_target->name );
	}
	return( unloaded );
#else
	fprintf( StdErr, "Error: dynamic modules not supported (compile with XG_DYMOD_SUPPORT defined)\n" );
	return(NULL);
#endif
}

int UnloadDyMods()
{ int r= 0;
	while( DyModList ){
	  int c;
		while( DyModList && (!DyModList->name || DyModList->type== DM_FUnloaded || DyModList->type== DM_Unloaded ) ){
			DyModList= DyModList->cdr;
		}
		if( DyModList ){
			r+= UnloadDyMod( DyModList->name, &c, True );
		}
	}
	return(r);
}

int Auto_UnloadDyMods()
{ int n= 0, r= 0;
  DyModLists *list= DyModList;
  static char active= 0;
	if( !active ){
		active+= 1;
		do{
			r= 0;
			while( list ){
			  int c;
				if( list->auto_unload== -1
					&& (list->name && list->type!= DM_FUnloaded && list->type!= DM_Unloaded && list->type!= DM_Python)
				){
					r+= UnloadDyMod( list->name, &c, True );
				}
				list= list->cdr;
			}
			n+= r;
			if( r ){
				list= DyModList;
			}
		} while( list );
		active-= 1;
	}
	return(n);
}

extern LocalWin *ActiveWin, StubWindow;

int Check_Doubles_Ascanf( ascanf_Function *af, char *label, int warn )
{ int j, delete= 0, r= 0;
  extern ascanf_Function vars_ascanf_Functions[];
  extern int ascanf_Functions;
	for( j= 0; j< ascanf_Functions; j++ ){
	  ascanf_Function *vaf= &vars_ascanf_Functions[j];
		while( vaf ){
			if( af!= vaf && strcmp( af->name, vaf->name)== 0 && vaf->type != _ascanf_novariable ){
				if( warn ){
				  LocalWin *wi= (ActiveWin)? ActiveWin : &StubWindow;
					errno= 0;
					if( XG_error_box( &wi,
							(label)? label : "Warning",
							vaf->name, ":\n",
							" There is at least 1 variable or function already defined.\n",
							" Should this/those variable(s) be deleted?\n",
							NULL
						) > 0
					){
						delete= True;
					}
					else{
						delete= False;
					}
				}
				if( delete ){
					Delete_Variable( vaf );
					fprintf( StdErr, "%s: deleted double %s (A=%d R=%d L=%d)\n",
						(label)? label : "dymod::Check_Doubles_Ascanf()",
						vaf->name, vaf->assigns, vaf->reads, vaf->links
					);
					xfree( vaf->name );
/* 					vaf->name= strdup("<deleted double>");	*/
					vaf->name= strdup("");
					vaf->name_length= strlen(vaf->name);
					vaf->hash= ascanf_hash(vaf->name, NULL);
				}
				r= 1;
			}
			vaf= vaf->cdr;
		}
	}
	return(r);
}

int Add_DyMod_Dependency( DyModLists *module, void *ptr, ascanf_Function *caf_af, ascanf_Function *af, void *ptr2 )
{ DyModDependency *dep, *new, *new2;
  Compiled_Form *caf_client= ptr, *form= ptr2;
  int n= 0;
	if( module && caf_client ){
		n= 1;
		dep= module->Dependencies;
		while( dep && dep->caf_client!= caf_client && dep->cdr ){
			dep= dep->cdr;
		}
		if( !dep || dep->caf_client!= caf_client ){
			if( (new= (DyModDependency*) calloc( 1, sizeof(DyModDependency))) ){
				new->dymod= module;
				new->caf_client= caf_client;
				  /* 20020402: this appears to be necessary; there must be a bug somewhere that causes
				   \ entries to be leftover in the dependency list(s); accessing their caf_client field
				   \ can then cause SIGSEGV crashes when those fields have already been destroyed.
				   */
				new->caf_type= caf_client->type;
				new->af= af;
				new->caf_af= caf_af;
#ifdef DEBUG
				new->caf_expr= XGstrdup( caf_client->expr );
				new->expr= XGstrdup( form->expr );
#endif
				new->form= form;
				new->cdr= NULL;
				if( (new2= (DyModDependency*) calloc( 1, sizeof(DyModDependency))) ){
					new2->dymod= module;
					new2->caf_client= caf_client;
					new2->caf_type= caf_client->type;
					new2->caf_af= caf_af;
					new2->af= af;
#ifdef DEBUG
					new2->caf_expr= XGstrdup( caf_client->expr );
					new2->expr= XGstrdup( form->expr );
#endif
					new2->form= form;
					new2->cdr= caf_client->DyMod_DependList;
					caf_client->DyMod_DependList= new2;
					caf_client->DyMod_Dependency+= 1;
				}
				if( dep ){
					dep->cdr= new;
				}
				else{
					module->Dependencies= new;
				}
			}
			else{
				fprintf( StdErr, "Add_DyMod_Dependency( \"%s\", 0x%lx): can't add new caf_client to dependency list: %s\n",
					module->name, caf_client, serror()
				);
				n= 0;
			}
		}
	}
	if( form ){
	  /* 20020519: in any event, register the direct dependency in the form: */
		form->dymod= module;
	}
	return(n);
}

/* Find and remove the entry in the Dependencies list of the given module
 \ that references the specified pointer (ptr).
 */
int Delete_DyMod_Dependency( DyModLists *module, void *ptr )
{ DyModDependency *list, *target;
  int hit= False, hits= 0;
#ifdef DEBUG
	if( !module || module->type== DM_Unloaded )
		return(0);
	else if( !(list= module->Dependencies) )
		return(0);
	else if( !ptr )
		return(0);
	else
#else
	if( module && module->type!= DM_Unloaded &&
		(list= module->Dependencies) &&
		ptr
	)
#endif
	{
		do{
			hit= False;
			list= module->Dependencies;
			if( list->caf_client== ptr || list->form== ptr ){
				target= list;
				module->Dependencies= list->cdr;
			}
			else{
				target= list->cdr;
				while( list && target && target->caf_client!= ptr && list->form!= ptr ){
					list= target;
					target= list->cdr;
				}
			}
			if( target && (target->caf_client== ptr || target->form==ptr) ){
				list->cdr= target->cdr;
#ifdef DEBUG
				xfree( target->expr );
				xfree( target->caf_expr );
#endif
				memset( target, 0, sizeof(DyModDependency) );
				xfree( target );
				hit= True;
				hits+= 1;
			}
		} while( hit && module->Dependencies );
	}
	return(hits);
}

/* Unregister the specified Compiled_Form <form>, removing all references of
 \ the form's expression nodes from the Dependency lists of the corresponding
 \ Modules.
 \ 20051216: if a module is specified, only remove <form> from the Dependency list
 \ of that particular module. Thus, unloading that module will not discard the dependency
 \ information a particular <form> node might have relative to another module.
 \ We did not do this before, which resulted in the auto-unloading of modules that were
 \ incorrectly marked as no longer needed, but without the corresponding 'noop' protection
 \ of the function the <form> points/ed to -> crash!
 */
int Delete_DyMod_Dependencies( struct Compiled_Form *form, DyModLists *module, char *caller )
{ int n= 0;
  DyModDependency *DDList= form->DyMod_DependList, *dep= DDList, *cdr;
	while( dep ){
		if( !module || module== dep->dymod ){
			if( dep->dymod ){
				if( !Delete_DyMod_Dependency( dep->dymod, form )
#ifndef DEBUG
					&& debugFlag
#endif
				){
					fprintf( StdErr, "%s->Delete_DyMod_Dependencies(", caller );
					Print_Form( StdErr, &form, 0, True, "             ", NULL, "\n", True );
					fprintf( StdErr, "): a DyMod dependency was registered (%s module 0x%lx=%s), but unregistering failed!\n",
						(dep->dymod->typestring)? dep->dymod->typestring : DyModTypeString(dep->dymod->type),
						dep->dymod, (dep->dymod)? dep->dymod->name : "no/null name"
					);
				}
				if( !dep->dymod->Dependencies && dep->dymod->auto_unload> 0 ){
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr, "%s->Delete_DyMod_Dependencies: Marking \"%s\" for auto-unloading since no longer needed\n",
							caller, dep->dymod->name
						);
					}
					  /* Mark for imminent unloading. Can't do it here since the module may still be actually
					   \ necessary.
					   */
					dep->dymod->auto_unload= -1;
				}
			}
			cdr= dep->cdr;
			  /* check if we have to update the list anchor: */
			if( form->DyMod_DependList== dep ){
				form->DyMod_DependList= cdr;
			}
			  /* update the list to reflect the removal of this element: */
			DDList->cdr= cdr;
#if DEBUG==2
			xfree( dep->expr );
			xfree( dep->caf_expr );
#endif
			memset( dep, 0, sizeof(DyModDependency) );
			xfree(dep);
			n+= 1;
			dep= cdr;
		}
		else{
		  /* an entry we have to skip at this time: */
			DDList= dep;
			dep= dep->cdr;
		}
	}
	return(n);
}

int Correct_DyMod_Dependencies( struct Compiled_Form *top, struct Compiled_Form *old )
{ int n= 0;
	while( old->DyMod_DependList ){
	  DyModDependency *dep= old->DyMod_DependList;
		if( !Delete_DyMod_Dependency( dep->dymod, old ) ){
			if( dep->dymod ){
				fprintf( StdErr, "Correct_DyMod_Dependencies(" );
				Print_Form( StdErr, &old, 0, True, "             ", NULL, "\n", True );
				fprintf( StdErr, "): a DyMod dependency was registered (%s module 0x%lx=%s), but unregistering failed!\n",
					(dep->dymod->typestring)? dep->dymod->typestring : DyModTypeString(dep->dymod->type),
					dep->dymod, (dep->dymod)? dep->dymod->name : "no/null name"
				);
			}
		}
		  /* 20020517: I think we should re-register even if unregistering failed?!! */
		/* else */{
#if DEBUG==2
			fprintf( StdErr, "Correct_DyMod_Dependencies(): re-registering %s (was %s) depending on %s (sub-expr %s)\n",
				top->expr, old->expr, top->fun->name, dep->form->expr
			);
#endif
			Add_DyMod_Dependency( dep->dymod, top, top->fun, dep->af, dep->form );
		}
		old->DyMod_DependList= dep->cdr;
#ifdef DEBUG
		xfree( dep->expr );
		xfree( dep->caf_expr );
#endif
		memset( dep, 0, sizeof(DyModDependency) );
		xfree(dep);
		n+= 1;
	}
	return(n);
}

int Auto_LoadDyMod_LastPtr( DyModAutoLoadTables *Table, int N, char *fname, DyModLists **last )
{ DyModAutoLoadTables *table;
  int n= 0;
	 /* 20040924: */
	if( !Table ){
	  extern DyModAutoLoadTables *AutoLoadTable;
	  extern int AutoLoads;
		Table= AutoLoadTable;
		N= AutoLoads;
	}
	if( (table= Table) ){
	  int i;
		if( !fname ){
			for( i= 0; i< N; i++ ){
				if( table->functionName && !table->hash ){
					table->hash= ascanf_hash( table->functionName, NULL );
				}
				table++;
			}
		}
		else{
		  long hash;
		  int hash_len;
			hash= ascanf_hash( fname, &hash_len );
			for( i= 0; i< N; i++ ){
				if( hash== table->hash && table->functionName && strncmp( fname, table->functionName, hash_len )== 0 ){
					table->loaded= False;
					if( table->DyModName ){
					  DyModLists *current= DyModList;
						while( current && !table->loaded ){
							if( strcmp( table->DyModName, current->name)== 0 ){
								table->loaded= True;
							}
							current= current->cdr;
						}
						if( !table->loaded ){
						  DyModLists *new= LoadDyMod( table->DyModName, table->flags, 1, 1 );
							if( new ){
								new->auto_loaded= True;
								new->loaded4= strdup(fname);
								table->dymod= new;
								table->loaded= True;
								n+= 1;
								if( debugFlag || scriptVerbose || table->warned ){
									fprintf( StdErr, "Auto-loaded \"%s\" to resolve reference to \"%s\" (matches %s)\n",
										table->DyModName, fname, table->functionName
									);
									table->warned= False;
								}
							}
							else{
								if( !table->warned ){
								  ALLOCA( errbuf, char, strlen(table->DyModName)+strlen(fname)+strlen(table->functionName)+64, elen );
								  LocalWin *wi= (ActiveWin)? ActiveWin : &StubWindow;
									sprintf( errbuf, "!! Failed to auto-load \"%s\" to resolve reference to \"%s\" (matches %s)\n",
										table->DyModName, fname, table->functionName
									);
									fputs( errbuf, StdErr );
									table->warned= True;
									XG_error_box( &wi, "Error", errbuf, NULL );
								}
							}
						}
					}
					if( table->loaded && last ){
						*last = table->dymod;
					}
				}
				table++;
			}
		}
	}
	return(n);
}

int Auto_LoadDyMod( DyModAutoLoadTables *Table, int N, char *fname )
{
	return( Auto_LoadDyMod_LastPtr( Table, N, fname, NULL ) );
}

DyModAutoLoadTables *Add_LoadDyMod( DyModAutoLoadTables *target, int *target_len, DyModAutoLoadTables *source, int n )
{ int tl= *target_len, i;
	if( source && n> 0 ){
		Auto_LoadDyMod( source, n, NULL );
	}
	else{
		return( target );
	}
	if( target ){
	  int j, N= n;
		for( j= 0; j< n && N> 0; j++ ){
			for( i= 0; i< *target_len && N> 0; i++ ){
				if( target[i].hash== source[j].hash && strcmp( target[i].functionName, source[j].functionName)==0 ){
					N-= 1;
				}
			}
		}
		*target_len+= N;
	}
	else{
		tl= 0;
		*target_len= n;
	}
	if( *target_len!= tl ){
		target= (DyModAutoLoadTables*) XGrealloc( target, sizeof(DyModAutoLoadTables)* *target_len );
		for( i= tl; i< *target_len; i++ ){
			memset( &target[i], 0, sizeof(target[i]) );
		}
	}
	if( target ){
	  int j, t= tl;
		for( j= 0; j< n; j++ ){
		  int ok= 0;
			for( i= 0; i< tl; i++ ){
				if( target[i].hash== source[j].hash && strcmp( target[i].functionName, source[j].functionName)==0 ){
					xfree(target[i].functionName);
					xfree(target[i].DyModName);
					target[i]= source[j];
					target[i].functionName= strdup(source[j].functionName);
					target[i].DyModName= strdup(source[j].DyModName);
					ok+= 1;
				}
			}
			if( !ok ){
				if( t< *target_len ){
					xfree(target[t].functionName);
					xfree(target[t].DyModName);
					target[t]= source[j];
					target[t].functionName= strdup(source[j].functionName);
					target[t].DyModName= strdup(source[j].DyModName);
					t+= 1;
				}
				else{
					fprintf( StdErr, "Add_LoadDyMod(): internal error adding element #%d, target #%d> length %d...\n",
						j, t, *target_len
					);
				}
			}
		}
	}
	else{
		*target_len= 0;
	}
	return( target );
}

int Init_Python()
{ char *XG_PYTHON_DYMOD= getenv( "XG_PYTHON_DYMOD" );
	if( !dm_pythonLib ){
		dm_pythonLib= LoadDyMod( (XG_PYTHON_DYMOD)?XG_PYTHON_DYMOD:"Python", RTLD_LAZY|RTLD_GLOBAL, True,False );
	}
	if( !dm_python && dm_pythonLib && dm_pythonLib->type== DM_Python ){
		dm_python= dm_pythonLib->libHook;
		if( dm_python->type!= DM_Python ){
			dm_python= NULL;
		}
	}
	if( dm_python && dm_pythonLib ){
		return( 1 );
	}
	else{
		return( 0 );
	}
}

char *PyOpcode_Check( char *expr )
{ char *pyExpr= NULL;
	if( strncasecmp( expr, "python::", 8) == 0 ){
		pyExpr= &expr[8];
	}
	else if( strncasecmp( expr, "py::", 4) == 0 ){
		pyExpr= &expr[4];
	}
	if( pyExpr ){
		Init_Python();
	}
	return( pyExpr );
}

struct DyMod_Interface *Init_DyMod_Interface( struct DyMod_Interface *base )
{ int ours= False;
	if( !base ){
		base= (struct DyMod_Interface *) calloc( 1, sizeof(struct DyMod_Interface) );
		ours= True;
	}
	if( base ){
	  char *c;
		base->XGraphHandle= dlopen( NULL, RTLD_GLOBAL|RTLD_LAZY );
		c= (char*) dlerror();
		if( !base->XGraphHandle || c ){
			fprintf( StdErr, "Error: can't attach to xgraph's main (programme) module " );
			fprintf( StdErr, " (%s)(%s)\n", serror(), (c)? c : "?" );
			if( ours ){
				xfree(base);
			}
		}
		else{
		  extern double ascanf_progn_return, *ascanf_setNumber;
		  extern int ascanf_Variable( ASCB_ARGLIST );
		  extern char *_callback_expr( struct ascanf_Callback_Frame *__ascb_frame, char *fn, int lnr, char **stub );
		  int evaluate_procedure( int *n, ascanf_Function *proc, double *args, int *level );
		  extern int ascanf_check_event(char *caller);
		  extern int Resize_ascanf_Array_force, ascanf_interrupt, ascanf_exit;
		  extern int StartUp;
		  extern void **gnu_rl_event_hook;
		  extern char ascanf_separator;

			if( base->sizeof_DyMod_Interface && base->sizeof_DyMod_Interface!= sizeof(DyMod_Interface) ){
				fprintf( StdErr, "xgraph::Init_DyMod_Interface(): Probable API mismatch, not initialising this module!\n" );
				if( ours ){
					xfree(base);
				}
				return(base);
			}
			base->sizeof_DyMod_Interface= sizeof(DyMod_Interface);
			base->sizeof_ascanf_Function= sizeof(ascanf_Function);
			base->sizeof_Compiled_Form= sizeof(Compiled_Form);
			base->sizeof_LocalWin= sizeof(LocalWin);
			base->sizeof_DataSet= sizeof(DataSet);
			base->sizeof_Python_Interface= sizeof(DM_Python_Interface);

			base->p_DyModList= &DyModList;
			base->p_StdErr= &StdErr;
			base->p_NullDevice= &NullDevice;
			base->p_ActiveWin= &ActiveWin;
			base->p_StubWindow_ptr= &StubWindow_ptr;
			base->p_ascanf_separator= &ascanf_separator;
			base->p_ascanf_emsg= &ascanf_emsg;
			base->p_ascanf_escape= &ascanf_escape;
			base->p_ascanf_exit = &ascanf_exit;
			base->p_ascanf_interrupt= &ascanf_interrupt;
			base->p_ascanf_arg_error= &ascanf_arg_error;
			base->p_ascanf_arguments= &ascanf_arguments;
			base->p_ascanf_SyntaxCheck= &ascanf_SyntaxCheck;
			base->p_ascanf_verbose= &ascanf_verbose;
			base->p_ascanf_setNumber= &ascanf_setNumber;
			base->p_Unloaded_Used_Modules= &Unloaded_Used_Modules;
			base->p_ascanf_progn_return= &ascanf_progn_return;
			base->p_debugFlag= &debugFlag;
			base->p_debugLevel= &debugLevel;
			base->p_ascanf_array_malloc= (void*)&ascanf_array_malloc;
			base->p_ascanf_array_free= (void*)&ascanf_array_free;
			base->p_Resize_ascanf_Array_force= &Resize_ascanf_Array_force;
			base->p_MaxSets= &MaxSets;
			base->p_setNumber= &setNumber;
			base->p_AllSets= &AllSets;
			base->p_ascanf_window= (unsigned long*)&ascanf_window;
			base->p_af_ArgList= &af_ArgList;
			base->p_af_ArgList_address= &af_ArgList_address;
			base->p_ascanf_update_ArgList= &ascanf_update_ArgList;
			base->p_Ascanf_Max_Args= &Ascanf_Max_Args;
			base->p_SwapEndian= &SwapEndian;
			base->p_EndianType= &EndianType;
			base->p_scriptVerbose= &scriptVerbose;
			base->p_PrefsDir= &PrefsDir;

			base->p_disp= (void*) &disp;
			base->p_RemoteConnection= &RemoteConnection;

			base->p_StartUp= &StartUp;
			base->p_dm_python= &dm_python;

			base->p_d3str_format= &d3str_format[0];
			base->p_EmptySimpleStats= &EmptySimpleStats;
			base->p_EmptySimpleAngleStats= &EmptySimpleAngleStats;

			base->p_xtb_error_box= xtb_error_box;
			base->p_ascanf_CheckFunction= ascanf_CheckFunction;
			base->p_add_ascanf_functions= add_ascanf_functions;
			base->p_add_ascanf_functions_with_autoload= add_ascanf_functions_with_autoload;
			base->p_remove_ascanf_functions= remove_ascanf_functions;
			base->p_Copy_preExisting_Variable_and_Delete= Copy_preExisting_Variable_and_Delete;
			base->p_XGstrdup= XGstrdup;
			base->p_XGstrcmp= XGstrcmp;
			base->p_XGstrncmp= XGstrncmp;
			base->p_XGstrcasecmp= XGstrcasecmp;
			base->p_XGstrncasecmp= XGstrncasecmp;
			base->p_Check_Doubles_Ascanf= Check_Doubles_Ascanf;
			base->p_parse_ascanf_address= parse_ascanf_address;
			base->p_take_ascanf_address= take_ascanf_address;
			base->p_get_VariableWithName= get_VariableWithName;
			base->p_AccessHandler= AccessHandler;
			base->p_Auto_LoadDyMod= Auto_LoadDyMod;
			base->p_Auto_LoadDyMod_LastPtr= Auto_LoadDyMod_LastPtr;
			base->p_d2str= d2str;
			base->p_ad2str= ad2str;
			base->p_fascanf2= fascanf2;
			base->p_ascanf_Variable= ascanf_Variable;
			base->p_Resize_ascanf_Array= Resize_ascanf_Array;
			base->p_tildeExpand= tildeExpand;
			base->p_IncludeFile= IncludeFile;
			base->p__xfree= _xfree;
			base->p__callback_expr= _callback_expr;
			base->p__XGrealloc= _XGrealloc;
			base->p_xgalloca= xgalloca;
			base->p_GetEnv= GetEnv;
			base->p_SetEnv= SetEnv;
			base->p_ascanf_check_event= ascanf_check_event;
			base->p_evaluate_procedure= evaluate_procedure;
			base->p_q_Permute= q_Permute;
			base->p_Elapsed_Since= Elapsed_Since;
			base->p_Elapsed_Since_HR= Elapsed_Since_HR;
			base->p_SwapEndian_int16= SwapEndian_int16;
			base->p_SwapEndian_int= SwapEndian_int;
			base->p_SwapEndian_int32= SwapEndian_int32;
			base->p_SwapEndian_float= SwapEndian_float;
			base->p_SwapEndian_double= SwapEndian_double;
			base->p_StringCheck= StringCheck;
			base->p_XGStringList_AddItem= XGStringList_AddItem;
			base->p_XGStringList_Delete= XGStringList_Delete;
			base->p_XGStringList_FindItem= XGStringList_FindItem;
			base->p_SS_Mean= SS_Mean;
			base->p_SS_Mean= SS_Mean;
			base->p_SS_St_Dev= SS_St_Dev;
			base->p_SS_sprint_full= SS_sprint_full;
			base->p_SS_Add_Data= SS_Add_Data;
			base->p_SAS_Add_Data= SAS_Add_Data;
			base->p_SAS_Mean= SAS_Mean;
			base->p_SAS_St_Dev= SAS_St_Dev;
			base->p_SAS_sprint_full= SAS_sprint_full;
			base->p_SS_St_Dev= SS_St_Dev;
			base->p_SS_sprint_full= SS_sprint_full;
			base->p_SS_Add_Data= SS_Add_Data;
			base->p_SAS_Add_Data= SAS_Add_Data;
			base->p_SAS_Mean= SAS_Mean;
			base->p_SAS_St_Dev= SAS_St_Dev;
			base->p_SAS_sprint_full= SAS_sprint_full;
			base->p_Sinc_string_behaviour= Sinc_string_behaviour;
			base->p_Sflush= Sflush;
			base->p_Sputs= Sputs;
			base->p_concat= concat;
			base->p_concat2= concat2;

			base->p_aWindow= aWindow;
			base->p_parse_codes= parse_codes;
			base->p_ParseTitlestringOpcodes= ParseTitlestringOpcodes;
			base->p_strrstr= strrstr;
			base->p_xg_re_comp= xg_re_comp;
			base->p_xg_re_exec= xg_re_exec;
			base->p_xtb_input_dialog= xtb_input_dialog;
			base->p_RedrawNow= RedrawNow;
			base->p_RedrawSet= RedrawSet;
			base->p_new_param_now= new_param_now;
			base->p_Ascanf_AllocMem= Ascanf_AllocMem;

			base->p_gnu_rl_event_hook= &gnu_rl_event_hook;
			base->p_Num_Windows = &Num_Windows;
		}
	}
	return( base );
}
