#include "config.h"
IDENTIFY( "Strings ascanf library module" );

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
   */

#include <libgen.h>
#include <float.h>

#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;
	ascanf_Function* (*Create_Internal_ascanfString_ptr)( char *string, int *level );
	xtb_hret (*SimpleEdit_h_ptr)( Window win, int bval, xtb_data info );
	xtb_hret (*SimpleFileDialog_h_ptr)( Window win, int bval, xtb_data info );
	int (*sprint_string_string_ptr)( char **target, char *header, char *trailer, char *string, int *instring );

#	define Create_Internal_ascanfString	(*Create_Internal_ascanfString_ptr)
#	define SimpleEdit_h	(*SimpleEdit_h_ptr)
#	define SimpleFileDialog_h	(*SimpleFileDialog_h_ptr)
#	define sprint_string_string	(*sprint_string_string_ptr)

int ascanf_basename ( ASCB_ARGLIST ) 
{ ASCB_FRAME
  ascanf_Function *s1, *s2= NULL;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_basename", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || (!s1->usage && ascanf_verbose) )
		){
			fprintf( StdErr, " (warning: argument is not a valid string)== " );
		}
		if( s1 && s1->usage ){
		  char *bn= basename(s1->usage), *ext= NULL;
		  ascanf_Function *allocated;
			if( ascanf_arguments> 1 && args[1] &&
				(!(s2= parse_ascanf_address(args[1], 0, "ascanf_basename", (int) ascanf_verbose, NULL )) || 
				(s2->type== _ascanf_procedure || s2->type== _ascanf_function || (!s2->usage && ascanf_verbose) ))
			){
				fprintf( StdErr, " (warning: second argument is not a valid string -- ignored)== " );
				s2= NULL;
			}
			if( s2 ){
				if( (ext= strrstr( bn, s2->usage )) ){
					*ext= '\0';
				}
			}
			if( (allocated= Create_Internal_ascanfString( bn, level )) ){
				*result= take_ascanf_address( allocated );
			}
			else{
				fprintf( StdErr, " (error: could not duplicate basename(%s)=\"%s\": %s)== ", s1->name, bn, serror() );
				ascanf_arg_error= True;
				*result= 0;
			}
			if( ext ){
			  /* if ext!=NULL, then also s2!=NULL */
				*ext= s2->usage[0];
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_encode ( ASCB_ARGLIST ) 
{ ASCB_FRAME
  ascanf_Function *s1;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strdup", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || (!s1->usage && ascanf_verbose) )
		){
			fprintf( StdErr, " (warning: argument is not a valid string)== " );
		}
		if( s1 && s1->usage ){
		  char *encoded= NULL;
		  int instring= True;
			sprint_string_string( &encoded, NULL, NULL, s1->usage, &instring );
			*result= args[0];
			if( encoded ){
			  ascanf_Function *allocated;
				if( ascanf_arguments> 1 && !ASCANF_TRUE(args[1]) ){
					if( (allocated= Create_Internal_ascanfString( encoded, level )) ){
						*result= take_ascanf_address( allocated );
						xfree(encoded);
					}
					else{
						fprintf( StdErr, " (error: could not duplicate encode(%s)=\"%s\": %s)== ", s1->name, encoded, serror() );
						ascanf_arg_error= True;
						*result= 0;
					}
				}
				else{
					xfree(s1->usage);
					s1->usage= encoded;
				}
			}
			else{
				ascanf_arg_error= True;
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_getstring ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1= NULL, *ret= NULL, *title= NULL, *msg= NULL;
  ascanf_Function *af= NULL;
  static ascanf_Function AF= {NULL};
  static char *AFname= "GetString-Static-StringPointer";
  int take_usage, maxlen= -1;
  FILE *fp= NULL;
	af= &AF;
	if( af->name ){
	  double oa= af->own_address;
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
		af->own_address= oa;
	}
	else{
		af->usage= NULL;
		af->type= _ascanf_variable;
		af->internal= True;
		af->name= AFname;
		take_ascanf_address(af);
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->internal= True;
	ascanf_arg_error= 0;
	if( ascanf_arguments> 0 ){
		if( !(s1= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_getstring", (int) ascanf_verbose, &take_usage )) || 
			!take_usage
		){
			fprintf( StdErr, " (warning: 1st argument is not a valid stringpointer)== " );
			ret= s1= NULL;
			maxlen= (int) args[0];
		}
		else{
			ret= s1;
			maxlen= ret->value;
		}
	}
	if( ascanf_arguments> 1 ){
		if( !(msg= parse_ascanf_address(args[1], 0, "ascanf_getstring", (int) ascanf_verbose, NULL )) || 
			(msg->type== _ascanf_procedure || msg->type== _ascanf_function || !msg->usage )
		){
			fprintf( StdErr, " (warning: message (2nd) argument is not a valid string)== " );
			if( msg && msg->fp ){
				fp= msg->fp;
			}
			msg= NULL;
		}
		else if( msg && msg->fp ){
			fp= msg->fp;
		}
	}
	if( !fp && ascanf_arguments> 2 ){
		if( !(title= parse_ascanf_address(args[2], 0, "ascanf_getstring", (int) ascanf_verbose, NULL )) || 
			(title->type== _ascanf_procedure || title->type== _ascanf_function || !title->usage )
		){
			fprintf( StdErr, " (warning: title (3rd) argument is not a valid string)== " );
			title= NULL;
		}
	}
	if( ret ){
		af= ret;
	}
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	if( !ascanf_SyntaxCheck ){
	  LocalWin *wi= aWindow(ActiveWin);
		if( maxlen> 0 && (wi || fp) ){
		  ALLOCA( buf, char, maxlen, blen );
		  char *nbuf;
			if( fp ){
				xfree( af->usage );
				if( (nbuf= fgets( buf, maxlen-1, fp )) ){
					af->usage= strdup(nbuf);
				}
				else{
					af->value= 0;
				}
			}
			else{
				if( s1 && s1->usage ){
					strncpy( buf, s1->usage, maxlen-1 );
					buf[maxlen-1]= '\0';
				}
				else{
					buf[0]= '\0';
				}
				if( wi && (nbuf= xtb_input_dialog( wi->window, buf, (*buf)? strlen(buf)* 1.5 : 80, maxlen,
						(msg)? parse_codes(msg->usage) : "Please enter a string",
						(title)? parse_codes(title->usage) :parse_codes( "#x01Request"),
						  /* should this one be modal??? */
						False,
						NULL, NULL, "Files", SimpleFileDialog_h_ptr, "Edit", SimpleEdit_h_ptr
					))
				){
					xfree( af->usage );
					if( ret ){
						ret->usage= strdup(nbuf);
					}
					else{
						af->usage= strdup(nbuf);
					}
					if( nbuf!= buf ){
						xfree(nbuf);
					}
				}
				else{
					af->value= 0;
				}
			}
		}
		else{
			ascanf_emsg= "invalid max. string length or no active window!";
			ascanf_arg_error= True;
		}
	}

	*result= af->own_address;
	return(!ascanf_arg_error);
}

#ifdef linux
/* For linux, we need to define _REGEX_RE_COMP in order to get the declarations we want.... */
#	define _REGEX_RE_COMP
#endif
#include <regex.h>

int ascanf_strcasecmp ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *s2;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strcasecmp", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (warning: 1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 1st argument is NULL)== " );
			}
		}
		if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_strcasecmp", (int) ascanf_verbose, NULL )) || 
			(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
		){
			if( s2 ){
				fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
					s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 2nd argument is NULL)== " );
			}
		}
		if( s1 && s2 && s1->usage && s2->usage ){
			if( ascanf_arguments> 2 ){
				if( args[2]< 0 ){
					*result= strncasecmp( s1->usage, s2->usage, strlen(s2->usage) );
				}
				else{
					*result= strncasecmp( s1->usage, s2->usage, (int) args[2] );
				}
			}
			else{
				*result= strcasecmp( s1->usage, s2->usage );
			}
		}
		else{
			set_Inf(*result, 1);
		}
	}
	return(!ascanf_arg_error);
}

static char *_strcasestr( const char *a,  const char *b)
{ unsigned int len= strlen(b), lena= strlen(a);
  int nomatch= 0, n= len;

	while( ( nomatch= (strncasecmp(a, b, len)) ) && n< lena ){
		a++;
		n+= 1;
	}
	return( (nomatch)? NULL : (char*) a );
}

int ascanf_strcasestr ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *s2, *af;
  static ascanf_Function AF= {NULL};
  static char *AFname= "StrCaseStr-Static-StringPointer";
	af= &AF;
	if( AF.name ){
	  double oa= af->own_address;
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
		af->own_address= oa;
	}
	else{
		af->usage= NULL;
		af->type= _ascanf_variable;
		af->is_address= af->take_address= True;
		af->is_usage= af->take_usage= True;
		af->internal= True;
		af->name= AFname;
		take_ascanf_address(af);
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strcasestr", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (warning: 1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 1st argument is NULL)== " );
			}
		}
		if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_strcasestr", (int) ascanf_verbose, NULL )) || 
			(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
		){
			if( s2 ){
				fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
					s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 2nd argument is NULL)== " );
			}
		}
		if( ascanf_SyntaxCheck ){
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= af->own_address;
		}
		else{
			*result= 0;
			if( s1 && s2 && s1->usage && s2->usage ){
			  char *c= _strcasestr( s1->usage, s2->usage );
				if( c ){
					xfree( af->usage );
					af->usage= strdup( c );
					*result= af->own_address;
				}
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_strcmp ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *s2;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strcmp", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (warning: 1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 1st argument is NULL)== " );
			}
		}
		if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_strcmp", (int) ascanf_verbose, NULL )) || 
			(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
		){
			if( s2 ){
				fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
					s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 2nd argument is NULL)== " );
			}
		}
		if( s1 && s2 && s1->usage && s2->usage ){
			if( strncmp( s2->usage, "RE^", 3)== 0 && s2->usage[ strlen(s2->usage)-1]== '$' ){
			  char *c;
				if( (c= re_comp( &(s2->usage[2]) )) ){
					ascanf_emsg= c;
				}
				else{
					*result= !re_exec( s1->usage );
				}
			}
			else{
				if( ascanf_arguments> 2 ){
					if( args[2]< 0 ){
						*result= strncmp( s1->usage, s2->usage, strlen(s2->usage) );
					}
					else{
						*result= strncmp( s1->usage, s2->usage, (int) args[2] );
					}
				}
				else{
					*result= strcmp( s1->usage, s2->usage );
				}
			}
		}
		else{
			set_Inf(*result, 1);
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_strdup ( ASCB_ARGLIST ) 
{ ASCB_FRAME
  ascanf_Function *s1;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strdup", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || (!s1->usage && ascanf_verbose) )
		){
			fprintf( StdErr, " (warning: argument is not a valid string)== " );
		}
		if( s1 && s1->usage ){
		  ascanf_Function *allocated;
			if( (allocated= Create_Internal_ascanfString( s1->usage, level )) ){
				  /* We must now make sure that the just created internal variable behaves as a user variable: */
				allocated->user_internal= True;
				allocated->internal= True;
				*result= take_ascanf_address( allocated );
			}
			else{
				fprintf( StdErr, " (error: could not duplicate %s=\"%s\": %s)== ", s1->name, s1->usage, serror() );
				ascanf_arg_error= True;
				*result= 0;
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_strlen ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strlen", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || (!s1->usage && ascanf_verbose) )
		){
			fprintf( StdErr, " (warning: argument is not a valid string)== " );
		}
		if( s1 && s1->usage ){
			*result= strlen( s1->usage );
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_strstr ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *s2, *af;
  static ascanf_Function AF= {NULL};
  static char *AFname= "StrStr-Static-StringPointer";
	af= &AF;
	if( AF.name ){
	  double oa= af->own_address;
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
		af->own_address= oa;
	}
	else{
		af->usage= NULL;
		af->type= _ascanf_variable;
		af->is_address= af->take_address= True;
		af->is_usage= af->take_usage= True;
		af->internal= True;
		af->name= AFname;
		take_ascanf_address(af);
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strstr", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (warning: 1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 1st argument is NULL)== " );
			}
		}
		if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_strstr", (int) ascanf_verbose, NULL )) || 
			(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
		){
			if( s2 ){
				fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
					s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 2nd argument is NULL)== " );
			}
		}
		if( ascanf_SyntaxCheck ){
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= af->own_address;
		}
		else{
			*result= 0;
			if( s1 && s2 && s1->usage && s2->usage ){
			  char *c= strstr( s1->usage, s2->usage );
				if( c ){
					xfree( af->usage );
					af->usage= strdup( c );
					*result= af->own_address;
				}
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_strrstr ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *s2, *af;
  static ascanf_Function AF= {NULL};
  static char *AFname= "StrRStr-Static-StringPointer";
	af= &AF;
	if( AF.name ){
	  double oa= af->own_address;
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
		af->own_address= oa;
	}
	else{
		af->usage= NULL;
		af->type= _ascanf_variable;
		af->is_address= af->take_address= True;
		af->is_usage= af->take_usage= True;
		af->internal= True;
		af->name= AFname;
		take_ascanf_address(af);
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strrstr", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (warning: 1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 1st argument is NULL)== " );
			}
		}
		if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_strrstr", (int) ascanf_verbose, NULL )) || 
			(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
		){
			if( s2 ){
				fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
					s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 2nd argument is NULL)== " );
			}
		}
		if( ascanf_SyntaxCheck ){
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= af->own_address;
		}
		else{
			*result= 0;
			if( s1 && s2 && s1->usage && s2->usage ){
			  char *c= strrstr( s1->usage, s2->usage );
				if( c ){
					xfree( af->usage );
					af->usage= strdup( c );
					*result= af->own_address;
				}
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_strstr_count ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *s2, *af;
  static ascanf_Function AF= {NULL};
  static char *AFname= "StrStr-Static-StringPointer";
	af= &AF;
	if( AF.name ){
	  double oa= af->own_address;
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
		af->own_address= oa;
	}
	else{
		af->usage= NULL;
		af->type= _ascanf_variable;
		af->is_address= af->take_address= True;
		af->is_usage= af->take_usage= True;
		af->internal= True;
		af->name= AFname;
		take_ascanf_address(af);
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_strstr_count", (int) ascanf_verbose, NULL )) || 
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (warning: 1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 1st argument is NULL)== " );
			}
		}
		if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_strstr_count", (int) ascanf_verbose, NULL )) || 
			(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
		){
			if( s2 ){
				fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
					s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
				);
			}
			else{
				fprintf( StdErr, " (warning: 2nd argument is NULL)== " );
			}
		}
		*result= 0;
		if( s1 && s2 && s1->usage && s2->usage ){
		  char *c= s1->usage;
			*result= 0;
			while( c && *c && (c= strstr( c, s2->usage )) ){
				*result+= 1;
				c++;
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_concat ( ASCB_ARGLIST ) 
{ ASCB_FRAME
  ascanf_Function *s1;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
	 int idx= 0;
	 char *rstring= NULL;
		for( idx= 0; idx< ascanf_arguments; idx++ ){
			if( (s1= parse_ascanf_address(args[idx], 0, "ascanf_concat", (int) ascanf_verbose, NULL ))
				&& s1->usage
				&& !(s1->type== _ascanf_procedure || s1->type== _ascanf_function)
			){
				rstring= concat2( rstring, s1->usage, NULL );
			}
			else if( s1 ){
				if( s1->type== _ascanf_array && s1->N ){
				  int i;
					rstring= concat2( rstring, "{", ad2str( ASCANF_ARRAY_ELEM(s1,0), d3str_format, NULL ), NULL );
					for( i= 1; i< s1->N-1; i++ ){
						rstring= concat2( rstring, ",", ad2str( ASCANF_ARRAY_ELEM(s1,i), d3str_format, NULL ), NULL );
					}
					rstring= concat2( rstring, ",", ad2str( ASCANF_ARRAY_ELEM(s1,i), d3str_format, NULL ), "}", NULL );
				}
				else{
					rstring= concat2( rstring, ad2str(s1->value, d3str_format, NULL), NULL );
				}
			}
			else{
				rstring= concat2( rstring, ad2str(args[idx], d3str_format, NULL), NULL );
			}
		}
		if( rstring ){
		  ascanf_Function *allocated;
			if( (allocated= Create_Internal_ascanfString( rstring, level )) ){
				  /* We must now make sure that the just created internal variable behaves as a user variable: */
				allocated->user_internal= True;
				allocated->internal= True;
				*result= take_ascanf_address( allocated );
			}
			else{
				fprintf( StdErr, " (error: could not duplicate %s=\"%s\": %s)== ", s1->name, rstring, serror() );
				ascanf_arg_error= True;
				*result= 0;
			}
			xfree(rstring);
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_timecode ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  double t;
  int base= 0, hbase= 3600, mbase= 60;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
	  int hh, mm, ssi, ssir;
	  double ss;
	  char buf[256];
	  ascanf_Function *af= NULL;
		if( ascanf_arguments> 1 && ASCANF_TRUE(args[1]) && args[1]> 0 ){
			base= (int) args[1];
			hbase*= base;
			mbase*= base;
			t= args[0] * base;
			hh= (int) (t / hbase);
			mm= (int) ((t-hh*hbase) / mbase);
			ss= (t-hh*hbase-mm*mbase);
			ssi= (int) (ss/base);
			ssir= (int) ((ss-ssi*base));
			snprintf( buf, sizeof(buf)/sizeof(char), "%02d:%02d:%02d.%0d/%d",
				hh, mm, ssi, ssir, base
			);
		}
		else{
			base= 100;
			hbase*= base;
			mbase*= base;
			t= args[0] * base;
			hh= (int) (t / hbase);
			mm= (int) ((t-hh*hbase) / mbase);
			ss= (t-hh*hbase-mm*mbase)/ base;
			snprintf( buf, sizeof(buf)/sizeof(char), "%02d:%02d:%s",
				hh, mm, ad2str(ss, d3str_format, NULL)
			);
		}
		if( ascanf_arguments> 2 && ASCANF_TRUE(args[2]) ){
		  int take_usage;
			if( (af= parse_ascanf_address(args[2], _ascanf_variable, "ascanf_timecode", (int) ascanf_verbose, &take_usage )) && 
				!take_usage
			){
				af= NULL;
				if( ascanf_verbose ){
					fprintf( StdErr, " (2nd argument %s is not a valid stringpointer, ignored) ",
						ad2str( args[2], d3str_format, NULL )
					);
				}
			}
		}
		if( !af ){
		  static ascanf_Function AF= {NULL};
		  static char *AFname= "timecode-Static-StringPointer";
			af= &AF;
			if( AF.name ){
			  double oa= af->own_address;
				xfree(af->usage);
				memset( af, 0, sizeof(ascanf_Function) );
				af->own_address= oa;
			}
			else{
				af->usage= NULL;
				af->type= _ascanf_variable;
				af->is_address= af->take_address= True;
				af->is_usage= af->take_usage= True;
				af->internal= True;
				af->name= AFname;
				take_ascanf_address(af);
			}
			af->name= AFname;
			af->type= _ascanf_variable;
			af->is_address= af->take_address= True;
			af->is_usage= af->take_usage= True;
			af->internal= True;
		}
		xfree( af->usage );
		af->usage= strdup( buf );
		*result= af->own_address;
	}
	return( !ascanf_arg_error );
}

static ascanf_Function strings_Function[] = {
	{ "basename", ascanf_basename, 2, NOT_EOF_OR_RETURN,
		"basename[string1[,ext]]: returns strdup(basename(string1,ext)); see `man basename`.\n"
		, 0, 1
	},
	{ "concat", ascanf_concat, AMAXARGS, NOT_EOF_OR_RETURN,
		"concat[string1[,string2[,...]]]: concatenates strings and string representations of\n"
		" arrays (pointers to, of course) and variables.\n"
		, 0, 1
	},
	{ "encode", ascanf_encode, 2, NOT_EOF_OR_RETURN,
		"encode[string[,inplace?]: encodes a string, replacing newlines with #xn, etc.\n"
		" This is done in place by default (modifying <string>!), unless <inplace> is False.\n"
	},
	{ "getstring", ascanf_getstring, 3, NOT_EOF_OR_RETURN,
		"getstring[string[, dialog_message[, dialog_title]]]: post a dialog that asks the user\n"
		" to input a string. The max. length is taken from <string>'s value. If <string> is a\n"
		" valid stringpointer, the dialog is initialised with this string, and the result is\n"
		" returned in it (otherwise, an internal buffer is returned).\n"
		"getstring[string,&file]: read text from file <file>, max. as many characters as\n"
		" specified by <string>'s value. If <string> is a stringpointer, the result is stored\n"
		" in it, and returned -- otherwise an internal buffer is used.\n"
		" Both versions set the value of the returned stringpointer to 0 in case an error or EOF occurs.\n"
		, 0, 1
	},
	{ "strcasecmp", ascanf_strcasecmp, 3, NOT_EOF_OR_RETURN,
		"strcasecmp[string,pattern[,n]]: see `man strcasecmp(2)`,\n"
		" or `man strcasencmp(2)` for when n is given (for n<0, compare over strlen(pattern)).\n"
		" Returns NaN on error, Inf when one or both args are null or not strings,\n"
		" number of mismatching characters."
	},
	{ "strcasestr", ascanf_strcasestr, 2, NOT_EOF_OR_RETURN,
		"strcasestr[string1,string2]: case-independent version of strstr: see `man strstr(2)`.\n"
		" Returns NaN on error, 0 when one or both args are null or not strings,\n"
		" otherwise a pointer to an internal string variable with a copy of the 1st match\n"
		" of string2 in string1\n", 0, 1
	},
	{ "strcmp", ascanf_strcmp, 3, NOT_EOF_OR_RETURN,
		"strcmp[string,pattern[,n]]: see `man strcmp(2)`,\n"
		" or `man strncmp(2)` for when n is given (for n<0, compare over strlen(pattern)).\n"
		" When pattern starts with RE^ and ends with $, the pattern (incl. the ^ and the $)\n"
		" is used as a regular expression (using re_comp() and re_exec()). In this case,\n"
		" the n argument is ignored. However, the result is still 0 for a match!!\n"
		" Returns NaN on error, Inf when one or both args are null or not strings,\n"
		" number of mismatching characters."
	},
	{ "strdup", ascanf_strdup, 1, NOT_EOF,
		"strdup[stringptr]: duplicates <stringptr>, creating a new internal variable.\n"
		" Returns the new stringpointer, or NaN on error (or missing stringdata)\n"
		" This function is relatively expensive.\n"
	},
	{ "strlen", ascanf_strlen, 1, NOT_EOF,
		"strlen[stringptr]: returns the length of <stringptr>, or NaN on error (or missing stringdata)"
	},
	{ "strrstr", ascanf_strrstr, 2, NOT_EOF_OR_RETURN,
		"strrstr[string1,string2]: see `man strrstr(2)`.\n"
		" Returns NaN on error, 0 when one or both args are null or not strings,\n"
		" otherwise a pointer to an internal string variable with a copy of the last match\n"
		" of string2 in string1\n", 0, 1
	},
	{ "strstr", ascanf_strstr, 2, NOT_EOF_OR_RETURN,
		"strstr[string1,string2]: see `man strstr(2)`.\n"
		" Returns NaN on error, 0 when one or both args are null or not strings,\n"
		" otherwise a pointer to an internal string variable with a copy of the 1st match\n"
		" of string2 in string1\n", 0, 1
	},
	{ "strstr-count", ascanf_strstr_count, 2, NOT_EOF_OR_RETURN,
		"strstr-count[string1,string2]: returns the number of occurrences of string2 in string1.\n"
		" (Or NaN on error.)\n"
		, 0, 1
	},
	{ "timecode", ascanf_timecode, 3, NOT_EOF_OR_RETURN,
		"timecode[t[,base[,ret_string]]]: returns a string of the form HH:MM:SS[.sfrac] or HH:MM:SS.srem/base\n"
		" where sfrac is a decimal fraction of seconds and srem is the integer remainder in <base>\n"
		" 'ticks' per second timebase when base is a positive, finite value (typically for video purposes).\n"
	},
};
static int strings_Functions= sizeof(strings_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= strings_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< strings_Functions; i++, af++ ){
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
		XGRAPH_FUNCTION(Create_Internal_ascanfString_ptr, "Create_Internal_ascanfString");
		XGRAPH_FUNCTION(SimpleEdit_h_ptr, "SimpleEdit_h");
		XGRAPH_FUNCTION(SimpleFileDialog_h_ptr, "SimpleFileDialog_h");
		XGRAPH_FUNCTION(sprint_string_string_ptr, "sprint_string_string");
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){

		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( strings_Function, strings_Functions, "strings::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-strings" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" various string functions.\n"
	);

	return( DM_Ascanf );
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
	  int r= remove_ascanf_functions( strings_Function, strings_Functions, force );
		if( force || r== strings_Functions ){
			for( i= 0; i< strings_Functions; i++ ){
				strings_Function[i].dymod= NULL;
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
				r, strings_Functions
			);
		}
	}
	return(ret);
}
