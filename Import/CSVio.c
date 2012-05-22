#include "config.h"
IDENTIFY( "CSVio import library module for tab-separated datafiles where the 1st line can be taken as column labels" );

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
#include "ReadData.h"

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
   */

#include <float.h>

#define DYMOD_MAIN
#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;

double** (*realloc_columns_ptr)( DataSet *this_set, int ncols );
void (*realloc_points_ptr)( DataSet *this_set, int allocSize, int force );
int (*LabelsList_N_ptr)( LabelsList *llist );
LabelsList* (*Parse_SetLabelsList_ptr)( LabelsList *llist, char *labels, char separator, int nCI, int *ColumnInclude );
LabelsList* (*Add_LabelsList_ptr)( LabelsList *current_LList, int *current_N, int column, char *label );
char* (*ColumnLabelsString_ptr)( DataSet *set, int column, char *newstr, int new, int nCI, int *ColumnInclude );
char* (*time_stamp_ptr)( FILE *fp, char *name, char *buf, int verbose, char *postfix);
int *ascanf_AutoVarWouldCreate_msg_ptr;

#	define realloc_columns	(*realloc_columns_ptr)
#	define realloc_points	(*realloc_points_ptr)
#	define Parse_SetLabelsList	(*Parse_SetLabelsList_ptr)
#	define Add_LabelsList	(*Add_LabelsList_ptr)
#	define LabelsList_N	(*LabelsList_N_ptr)
#	define ColumnLabelsString	(*ColumnLabelsString_ptr)
#	define time_stamp	(*time_stamp_ptr)
#	define ascanf_AutoVarWouldCreate_msg	(*ascanf_AutoVarWouldCreate_msg_ptr)

// data_separator exists globally too, use a local version in preparation of making it configurable one day.
static char data_separator = '\0';

static ascanf_Function CSVio_Function[] = {
	{ "$CSV-Import-Column-Selector", NULL, 2, _ascanf_variable,
		"$CSV-Import-Column-Selector: point this variable to an array specifying which columns to import.\n"
		" The array may be floating point or integer, but should enumerate the column numbers (i.e. do not\n"
		" use flags per column). Invalid values are silently ignored.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$CSV-Import-Feedback", NULL, 2, _ascanf_variable,
		"$CSV-Import-Feedback: shows and stores information about the import.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$CSV-French-Numerals", NULL, 2, _ascanf_variable,
		"$CSV-French-Numerals: activate support for French numerals (with a decimal comma instead of a decimal point)\n"
		" This is done by replacing all commas (surrounded by digits) by dots in the input string\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
};
static int CSVio_Functions= sizeof(CSVio_Function)/sizeof(ascanf_Function);

static double *ColumnSelector= &CSVio_Function[0].value;
static double *ImportFeedback= &CSVio_Function[1].value;
static double *FrenchNumerals= &CSVio_Function[2].value;

DM_IO_Handler CSVio_Handlers;

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= CSVio_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< CSVio_Functions; i++, af++ ){
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
			Copy_preExisting_Variable_and_Delete(af, label);
			if( label ){
				af->label= XGstrdup( label );
			}
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

/* remove heading and trailing whitespace	*/
char *cleanup( char *T )
{  char *c= T;
   static int h= 0, t= 0;
	if( !T ){
		return(NULL);
	}
	else if( ! *T ){
		return(T);
	}
	h= 0;
	t= 0;
	if( debugFlag ){
		fprintf( StdErr, "cleanup(0x%lx=\"%s\") ->\n", T, T);
	}
	  /* remove heading whitespace	*/
	if( isspace(*c) ){
		while( *c && isspace(*c) ){
			c++;
			h++;
		}
		strcpy( T, c);
	}
	  /* remove trailing whitespace	*/
	if( strlen( T) ){
		c= &T[ strlen(T)-1 ];
		if( isspace(*c) ){
			while( isspace(*c) && c> T ){
				c--;
				t++;
			}
			c[1]= '\0';
		}
	}
	if( debugFlag ){
		fprintf( StdErr, "\"%s\" (h=%d,t=%d)\n", (h||t)? T : "<no change>", h, t);
		fflush( StdErr );
	}
	return(T);
}

#ifndef sgi
#	define READ_ATONCE
#endif

char *fgs_realloc( char *buf, size_t *buflen, size_t newsize )
{
	if( !(buf = XGrealloc( buf, newsize * sizeof(char))) ){
		fprintf( StdErr, "fgetstr(): error (re)allocating %u elem string buffer (%s)\n", newsize, serror() );
	}
	else{
		*buflen = newsize;
	}
	return(buf);
}

char *fgetstr(FILE *fp, char *buf, size_t *buflen, int *err_rtn )
{ int n= 0, finished= 0;
  char c;
	if( !buflen ){
		fprintf( StdErr, "fgetstr(): buflen argument cannot be NULL\n" );
		if( err_rtn ){
			*err_rtn = 1;
		}
		return( buf );
	}
	if( err_rtn ){
		*err_rtn = 0;
	}
	if( !buf || *buflen <= 0 ){
		if( !(buf= fgs_realloc(buf, buflen, 256)) ){
			if( err_rtn ){
				*err_rtn = 2;
			}
			return(NULL);
		}
	}
	c= fgetc(fp);
	while( c!= EOF && !finished && !ferror(fp) ){
		  // always append a nullbyte
		if( n+1 >= *buflen ){
			if( !(buf= fgs_realloc(buf, buflen, *buflen * 2)) ){
				if( err_rtn ){
					*err_rtn = 2;
				}
				return(NULL);
			}
		}
		if( c == '\n' ){
			finished = 1;
			if( buf[n-1]== '\r' ){
			  // remove \r\n sequences
				n -= 1;
			}
		}
		buf[n++] = c;
		buf[n] = '\0';
		if( !finished ){
			c = fgetc(fp);
		}
	}
	if( ferror(fp) ){
		fprintf( StdErr, "fgetstr(): read error (%s)\n", serror() );
		if( err_rtn ){
			*err_rtn = -1;
		}
	}
	return(buf);
}

void Convert_French_Numerals( char *rbuf )
{ char *c = &rbuf[1];
  int i, n = strlen(rbuf)-1;
	for( i = 1 ; i < n ; i++, c++ ){
		if( *c == ',' && (isdigit(c[-1]) || isdigit(c[1])) ){
			*c = '.';
		}
	}
}

int import_CSV(FILE *stream, char *the_file, int filenr, int setNum, struct DataSet *this_set, ReadData_States *state )
{ int ret= 0, rerr;
  char *rbuf = NULL, *c;
  size_t rbuflen= 0, ncols, nicols, nc, i, poffset, imported= 0;
  int *ImportColumn= NULL, *TargetColumn= NULL;
  char fbbuf[1024];
  int fblen= sizeof(fbbuf)-1;
  Sinc sinc;
  long here= 0, line, line0;
	if( stream ){
		here = ftell(stream);
	}
	if( stream && (rbuf = fgetstr(stream, rbuf, &rbuflen, &rerr)) && !rerr ){
	  double *data;
	  ascanf_Function *af = NULL;
	  char *hline= NULL, *dline1= NULL;

		line = 1;
		while( rbuf[0] == '\0' ){
			here = ftell(stream);
			rbuf = fgetstr(stream, rbuf, &rbuflen, &rerr);
			line += 1;
			if( rerr ){
				goto iCSV_cleanup;
			}
		}

		Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
		Sflush(&sinc);
		Sputs( this_set->set_info, &sinc );

		line0 = line;
		{ int headerOK, findFirstLine = True;
			do{
				c = rbuf;
				while( *c && isspace(*c) ){
					c++;
				}
				headerOK = True;
				// as long as we're on a line with column labels (detected simply by the presence of a double quote
				// as the 1st non-white-space character), skip, in order to avoid chosing the wrong separator character...
				// 20100526: do accept a headerline that doesn't contain whitespace in its first label (between "");
				// this makes it a bit more possible to read sparse files.
				if( c[0] == '"' ){
				  char *cc = &c[1];
					while( *cc && *cc!='"' && !isspace(*cc) ){
						cc++;
					}
					if( isspace(*cc) ){
						headerOK = False;
					}
				}
				if( !headerOK ){
					xfree(hline);
					hline = XGstrdup(c);
					rbuf = fgetstr(stream, rbuf, &rbuflen, &rerr);
					line += 1;
					findFirstLine = True;
				}
				else{
					findFirstLine = False;
				}
			} while( findFirstLine && rbuf && !rerr && !feof(stream) && !ferror(stream) );
		}
		if( !(rbuf && !rerr && !feof(stream) && !ferror(stream)) ){
			goto iCSV_cleanup;
		}
		if( hline ){
			xfree(dline1);
			dline1 = XGstrdup(c);
		}
		{ int sep= False;
			for( ncols= 1; *c; c++ ){
				if( isspace(*c) && data_separator == '\0' ){
					data_separator = *c;
					sep= True;
					if( scriptVerbose || debugFlag ){
						fprintf( StdErr, "CSVio::import_CSV(%s:%ld): using separator '%c' (0x%x)\n",
						    the_file, line, data_separator, data_separator
						);
					}
				}
				else if( *c == data_separator ){
					sep= True;
				}
				else if( sep && !isspace(*c) ){
					ncols += 1;
					sep = False;
				}
			}
		}
#if 0
		// rewind to before the line with the column labels:
		if( line != line0 ){
			if( !fseek( stream, here, SEEK_SET ) ){
				rbuf = fgetstr(stream, rbuf, &rbuflen, &rerr);
				line = line0;
			}
			else{
			// 20090811: can do better. Cache the 1st line of data and the line with the labels, and then
			// handle things appropriately in the import loop.
				fprintf( StdErr, "CSVio::import_CSV(%s:%ld):can't seek on this file (%s)! - is it compressed?\n",
				    the_file, line, serror()
				);
			}
		}
#else
		if( hline ){
			strcpy( rbuf, hline );
			c = rbuf;
			xfree(hline);
		}
#endif

		if( !(data= (double*) calloc(ncols, sizeof(double)))
		    || !(ImportColumn= (int*) calloc(ncols, sizeof(int)))
		    || !(TargetColumn= (int*) calloc(ncols, sizeof(int)))
		){
			fprintf( StdErr, "CSVio::import_CSV(%s:%ld): error allocating %u data buffers (%s)\n",
			    the_file, line, ncols, serror()
		    );
		    goto iCSV_cleanup;
		}
		if( *ColumnSelector && (af=
				parse_ascanf_address( *ColumnSelector, _ascanf_array, "CSVio::import_CSV()",
					ascanf_verbose, NULL))
		){ int ic;
			memset( ImportColumn, 0, ncols*sizeof(int) );
			for( i= 0; i< af->N; i++ ){
				if( (ic= (int) ASCANF_ARRAY_ELEM(af,i))>= 0 && ic< ncols ){
					ImportColumn[ic]= 1;
				}
			}
		}
		else{
			for( i= 0; i< ncols; i++ ){
				ImportColumn[i] = 1;
			}
		}
		for( i= 0, nicols= 0; i< ncols; i++ ){
			if( ImportColumn[i] ){
				TargetColumn[i] = nicols++;
			}
		}
		if( ASCANF_TRUE(*ImportFeedback) ){
		  ALLOCA( buf, char, strlen(the_file)+256, blen );
			time_stamp( stream, the_file, buf, True, "\n" );
			snprintf( fbbuf, fblen,
				" CSVio::import_CSV(): reading on %u from %u channels from %s", 
				nicols, ncols, buf
			);
			Sputs( fbbuf, &sinc );
			if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
				fputs( fbbuf, StdErr );
			}
			if( af ){
				snprintf( fbbuf, fblen, "\tColumnSelector: '%s'[%d]\n", af->name, af->N );
				if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
					fputs( fbbuf, StdErr );
				}
				Sputs( fbbuf, &sinc );
			}
			GCA();
		}

		poffset = this_set->numPoints;

		if( this_set->ncols != nicols ){
			if( this_set->numPoints> 0 ){
				this_set->columns= realloc_columns( this_set, nicols );
			}
			else{
				this_set->ncols= nicols;
			}
		}
		  // previous DM_IO code used tests against this_set->numPoints: use allocSize which is cleaner.
		if( this_set->allocSize <= 1 ){
			this_set->allocSize = 64;
			realloc_points( this_set, this_set->allocSize, True );
		}
		{ int ww= ascanf_AutoVarWouldCreate_msg;
			  // no need to yell about unknown/undefined variables here, warnings are to be expected
			  // but we're not interested in creating variables, ever.
			ascanf_AutoVarWouldCreate_msg = False;
			nc = ncols;

			if( rbuf && ASCANF_TRUE(*FrenchNumerals) && data_separator != ',' ){
				Convert_French_Numerals( rbuf );
			}

			fascanf2( (int*) &nc, rbuf, data, data_separator);
			ascanf_AutoVarWouldCreate_msg = ww;
			if( nc == ncols ){
			  int allstrings= True;
			  ascanf_Function *af;
				for( i= 0; i< nc && allstrings; i++ ){
					if( !(af= parse_ascanf_address( data[i], 0, "CSVio::import_CSV()", (int) ascanf_verbose, NULL))
					    || !af->usage
					){
						allstrings= False;
					}
				}
				if( allstrings ){
				  // all strings, separated by tabs
					nc = 0;
				}
			}
		}
		if( nc < ncols ){
		  // this is most likely an indication that we just read a line with (string) labels
		  LabelsList *llist;
			ColumnLabelsString( this_set, -1, rbuf, 1, ncols, ImportColumn );
			if( (llist= this_set->ColumnLabels) ){
				while( llist ){
				  char *l = llist->label;
				  size_t llen = (l)? strlen(l) : 0;
					if( l[0] == '"' && l[llen-1] == '"' ){
					  // remove enclosing double quotes:
						l[llen-1] = '\0';
						llist->label = XGstrdup(&l[1]);
						xfree(l);
					}
					if( llist->min != llist->max ){
						llist++;
					}
					else{
						llist = NULL;
					}
				}
			}
		}
		else{
			for(i = 0; i< nc; i++ ){
				if( ImportColumn[i] ){
					this_set->columns[TargetColumn[i]][poffset] = data[i];
					imported += 1;
				}
			}
#if ADVANCED_STATS == 1
			this_set->N[poffset]= 1;
#endif
			
			*(state->spot)+= 1;
			*(state->Spot)+= 1;
			poffset+= 1;
		}
		/*
		 \ First line has been handled, now do the rest of the file
		 */
		do{
			if( dline1 ){
				strcpy( rbuf, dline1 );
				xfree(dline1);
			}
			else{
				rbuf = fgetstr(stream, rbuf, &rbuflen, &rerr);
			}
			if( rbuf && !rerr && !feof(stream) ){
				line += 1;
				nc = ncols;

				if( ASCANF_TRUE(*FrenchNumerals) && data_separator != ',' ){
					Convert_French_Numerals( rbuf );
				}

				fascanf2( (int*) &nc, rbuf, data, data_separator);
				if( nc< ncols && debugFlag ){
					fprintf( StdErr, "CSVio::import_CSV(%s:%ld): warning: read only %u of %u values\n",
					    the_file, line, nc, ncols
				    );
				}
				if( *(state->Spot)>= this_set->allocSize || *(state->spot)>= this_set->allocSize ){
					this_set->allocSize*= 2;
					realloc_points( this_set, this_set->allocSize, True );
				}
				for(i = 0; i< nc; i++ ){
					if( ImportColumn[i] ){
						this_set->columns[TargetColumn[i]][poffset] = data[i];
						imported += 1;
					}
				}
#if ADVANCED_STATS == 1
				this_set->N[poffset]= 1;
#endif
				
				*(state->spot)+= 1;
				*(state->Spot)+= 1;
				poffset+= 1;
			} 
		} while( rbuf && !rerr && !feof(stream) && !ferror(stream) );

		  // since we used allocSize to grow the set when necessary, numPoints is unchanged at this point ...
		this_set->numPoints= *(state->spot);
		realloc_points( this_set, this_set->numPoints, False );

		if( ASCANF_TRUE(*ImportFeedback) ){
			// determine a useful form of feedback
			snprintf( fbbuf, fblen, " CSVio::import_CSV(%s:%ld): imported %u values (%u x %u columns)"
				, the_file, line
				, imported, this_set->numPoints, nicols
			);
			strcat( fbbuf, "\n" );
			StringCheck( fbbuf, fblen, __FILE__, __LINE__ );
			Sputs( fbbuf, &sinc );
			if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
				fputs( fbbuf, StdErr );
			}
			if( sinc.sinc.string ){
				xfree( this_set->set_info );
				this_set->set_info= sinc.sinc.string;
			}
		}
		
		if( !this_set->setName ){
			this_set->setName= concat( the_file, " ", "%CY", NULL );
		}

		xfree(data);
		xfree(ImportColumn);
		xfree(TargetColumn);
	}
	else{
		fprintf( StdErr, "CSVio::import_CSV(%s:%ld): file not open or other read problem (%s)\n",
			the_file, line, serror()
		);
	}
iCSV_cleanup:;
	GCA();
	xfree(rbuf);
	return( ret );
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
		  /* The XGRAPH_FUNCTION macro can be used to easily initialise the additional variables we need.
		   \ In line with the bail out remark above, this macro returns DM_Error when anything goes wrong -
		   \ i.e. aborts initDyMod!
		   */
		XGRAPH_FUNCTION(realloc_columns_ptr, "realloc_columns");
		XGRAPH_FUNCTION(realloc_points_ptr, "realloc_points");
		XGRAPH_FUNCTION(Parse_SetLabelsList_ptr, "Parse_SetLabelsList");
		XGRAPH_FUNCTION(Add_LabelsList_ptr, "Add_LabelsList");
		XGRAPH_FUNCTION(LabelsList_N_ptr, "LabelsList_N");
		XGRAPH_FUNCTION( ColumnLabelsString_ptr, "ColumnLabelsString" );
		XGRAPH_FUNCTION(time_stamp_ptr, "time_stamp");
		XGRAPH_VARIABLE(ascanf_AutoVarWouldCreate_msg_ptr, "ascanf_AutoVarWouldCreate_msg");
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( CSVio_Function, CSVio_Functions, "utils::initDyMod()" );
		initialised= True;
	}
	  /* Initialise the library hook. For now, we only provide the import routine. */
	CSVio_Handlers.type= DM_IO;
	CSVio_Handlers.import= import_CSV;
	theDyMod->libHook= (void*) &CSVio_Handlers;
	theDyMod->libname= XGstrdup( "DM-CSVio" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" I/O facilities for CSV files.\n"
	);

	return( DM_IO );
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
	  int r= remove_ascanf_functions( CSVio_Function, CSVio_Functions, force );
		if( force || r== CSVio_Functions ){
			for( i= 0; i< CSVio_Functions; i++ ){
				CSVio_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->typestring );
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
				r, CSVio_Functions
			);
		}
	}
	return(ret);
}

// see the explanation printed by wrong_dymod_loaded():
void initCSVio()
{
	wrong_dymod_loaded( "initCSVio()", "Python", "CSVio.so" );
}

// see the explanation printed by wrong_dymod_loaded():
void R_init_CSVio()
{
	wrong_dymod_loaded( "R_init_CSVio()", "R", "CSVio.so" );
}

