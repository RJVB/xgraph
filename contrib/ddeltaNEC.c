#include "config.h"
IDENTIFY( "ddeltaNEC ascanf library module" );

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

  /* Any symbols not included in the DMBase should be obtained using (function) pointers and macros, as below: */

  /* Now that we have pointers, the actual code should be 'instructed' to use them. When done as follows, it
   \ should be largely transparent.
   \ NB: dymod_interface.h defines comparable macros for the fields in DMBase -- when XG_DYMOD_IMPORT_MAIN
   \ is set.
   */


typedef union {
	unsigned short m16;
	short i16;
	struct {
		unsigned char pf, pF;
	} M_8;
} CV_16;

#undef DEBUG_VBRAQ
#undef DEBUG_VBRAQ2
#ifdef DEBUG_VBRAQ
static unsigned short prt= 0;
#endif

typedef unsigned int COMPTEUR;
//typedef double COMPTEUR;
typedef unsigned short SORTIE_COMPTEUR;
//typedef double SORTIE_COMPTEUR;
typedef unsigned int NECtimer;
//typedef double NECtimer;

static COMPTEUR sommeVitessePos= 0, sommeVitesseNeg= 0;
static COMPTEUR sommeDTPos=0, sommeDTNeg= 0;
static COMPTEUR nbrEchantillonT=0;
static COMPTEUR nbrEchantillon0=0;
NECtimer tempsPredPos, tempsPredNeg, dernierDPos= 0;
static CV_16 posVolant, posVolantPred, *pVolant= &posVolant;
int prev_dpos= 0;
char avec_tempsPred= 1;

void COM1_Emission_Chaine(char *string )
{
	fputs( string, stderr );
}

int GestionCabine(NECtimer temps, NECtimer timer)
{
	int flush= 0;
	//LanceConvAna(7);
	//AcquisitionCapteur(temps);
	
	
	if ( /*pVolant->i16 != posVolantPred.i16 &&*/ (timer > tempsPredPos || timer > tempsPredNeg) )
	{
		register NECtimer dt;
		int dpos = (pVolant->i16 - posVolantPred.i16);
		COMPTEUR udpos;
#ifdef DEBUG_VBRAQ2
		// RJVB: tampon pour stoquer des valeurs pour déboguage, à imprimer 1 fois toutes les N invocations, pour
		// pouvoir juger la vraie fréquence (et vraies valeurs de temps) qu'on peut atteindre hors mode débogue.
#define VNBUF	2
		static struct data{
			short pos, posPred;
			int dpos;
			unsigned int timer, tempsPred, dt;
			unsigned long sVitPos, sVitNeg;
			unsigned int Npos, Nneg;
		} buffer[VNBUF], *bptr=buffer;
		static int i=0;
		NECtimer rdt=dt;
#endif
		
		
		// au lieu de calculer la moyenne de udpos/dt (donc la somme des udpos/dt et le nombre d'echantillons)
		// on fait une pondération de chaque échantillon par dt, et donc on calcule la somme des udpos et la somme des dt
		if( dpos< 0 ){
			// vitesse negative
			dt= timer - tempsPredNeg;
			udpos = (COMPTEUR)-dpos;
			sommeVitesseNeg += (COMPTEUR)udpos;
			sommeDTNeg+= dt;
			nbrEchantillonT += 1;
			tempsPredNeg= timer;
		}
		else	if( dpos ){
			// vitesse positive
			dt= timer - tempsPredPos;
			udpos = (COMPTEUR) dpos;
			sommeVitessePos += (COMPTEUR)udpos;
			sommeDTPos+= dt;
			nbrEchantillonT += 1;
			tempsPredPos= timer;
		}
#if 1
		else{
			//sommeDTPos+= dt;
			//sommeDTNeg+= dt;
			nbrEchantillon0 += 1;
		}
#endif
		
		if( sizeof(SORTIE_COMPTEUR) == 4 ){
			if( sommeDTNeg > (unsigned short)(0xFFFF-dt) ){
				//COM1_Emission_Chaine( "\n\rflushN!" );
				flush= 1;
			}
			else if( sommeDTPos > (unsigned short)(0xFFFF-dt) ){
				//COM1_Emission_Chaine( "\n\rflushP!" );
				flush= 1;
			}
		}
		
		posVolantPred.i16 = pVolant->i16;
		
	}
	return(flush);
}

NECtimer tpsEnMilliSec;
SORTIE_COMPTEUR sVP=0, sVN=0, sTP=0, sTN=0;
short NN= 0, N0= 0;
unsigned short maxTempsConstVit= 200;

unsigned char GestionCom(unsigned char dixms, unsigned char centms, NECtimer timer)
{
	if( dixms ){
		// si nouveau(x) échantillon(s) de vitesseBraquage, mettre à jour tabValeursBraquage:
		if( nbrEchantillonT ){
			// vitesse braquage moyenne en centitops/ms :
			if( sizeof(SORTIE_COMPTEUR) == 4 ){
				if( sommeVitessePos > (SORTIE_COMPTEUR)(0xFFFF) && sommeDTPos ){
					sommeVitessePos/= sommeDTPos;
					sommeDTPos= 1;
				}
			}
			sVP = (SORTIE_COMPTEUR) sommeVitessePos;
			if( sizeof(SORTIE_COMPTEUR) == 4 ){
				if( sommeVitesseNeg > (SORTIE_COMPTEUR)(0xFFFF) && sommeDTNeg ){
					sommeVitesseNeg/= sommeDTNeg;
					sommeDTNeg= 1;
				}
			}
			sVN = (SORTIE_COMPTEUR) sommeVitesseNeg;
			{ CV_16 Np, Nn;
				sTP= (SORTIE_COMPTEUR) sommeDTPos;
				sTN= (SORTIE_COMPTEUR) sommeDTNeg;
			}
			NN= nbrEchantillonT;
			N0= nbrEchantillon0;
			dernierDPos = tpsEnMilliSec;
		}
		else if( tpsEnMilliSec - dernierDPos>= maxTempsConstVit ){
			// la meme vitesse est transmise via CAN pour au maximum 0.2s
			sVN = sVP = sTP = sTN = 0;
			NN= 0; N0= 0;
			dernierDPos = tpsEnMilliSec;
			tempsPredPos = timer;
			tempsPredNeg = timer;
			// pas de changement de position par définition, donc pas raison de MAJ posVolantPred!
		}
		else if( NN> 0 ){
			NN= -NN;
		}
		// RJVB: pour un "vrai" échantillonage continu, il ne faut pas remettre à 0 les références de temps et de position!
		//tempsPred = 0;
		//posVolantPred.i16 = pVolant->i16;
		sommeVitesseNeg= sommeVitessePos = 0;
		sommeDTPos= 0;
		sommeDTNeg = 0;
		nbrEchantillonT = 0;
		nbrEchantillon0 = 0;
	}
	return(1);
}

int runNEC ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double freq= 1, loopdelay= 0, delay= 0;
  int n, i, resetTemps= 1, dixms= 0;
  ascanf_Function *Time, *Pos, *sampleT, *Delta, *DDelta,
	  *sumPos= NULL, *sumNeg= NULL, *sDTpos= NULL, *sDTneg= NULL, *Samples= NULL;
  ascanf_Function *sumPosHR= NULL, *sumNegHR= NULL, *sDTposHR= NULL, *sDTnegHR= NULL;
  NECtimer tmpTimer, tempsAbs, tpsEnMilliSec10ms= 0, dureeCycle1000;
  char *caller= "runNEC";
  
	ascanf_arg_error= False;

	if( !(Time= parse_ascanf_address( args[0], _ascanf_array, caller, (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (array with time values is required) ";
		ascanf_arg_error= True;
	}
	if( !(Pos= parse_ascanf_address( args[1], _ascanf_array, caller, (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (array with Positions at each time is required) ";
		ascanf_arg_error= True;
	}
	if( (freq= args[2])<= 0 ){
		fprintf( StdErr, "runNEC(): frequency should be positive, not %s: assuming 1Hz\n", ad2str(freq, d3str_format, 0 ) );
		freq= 1;
	}
	if( (loopdelay= args[3])< 0 ){
		fprintf( StdErr, "runNEC(): loopTime should be >=0 && <=1, not %s: assuming 0s\n", ad2str(loopdelay, d3str_format, 0 ) );
		loopdelay= 0;
	}
	else if( loopdelay> 1 ){
		fprintf( StdErr, "runNEC(): loopTime should be <=, not %s: assuming 1s\n", ad2str(loopdelay, d3str_format, 0 ) );
		delay= 1;
	}
	if( (delay= args[4])< 0 ){
		fprintf( StdErr, "runNEC(): sendDelay should be >=0 && <=1, not %s: assuming 0s\n", ad2str(delay, d3str_format, 0 ) );
		delay= 0;
	}
	else if( delay> 1 ){
		fprintf( StdErr, "runNEC(): sendDelay should be <=, not %s: assuming 1s\n", ad2str(delay, d3str_format, 0 ) );
		delay= 1;
	}
	if( !(sampleT= parse_ascanf_address( args[5], _ascanf_array, caller, (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (array for returning sample times is required) ";
		ascanf_arg_error= True;
	}
	if( !(Delta= parse_ascanf_address( args[6], _ascanf_array, caller, (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (array for returning position at each sample time is required) ";
		ascanf_arg_error= True;
	}
	if( !(DDelta= parse_ascanf_address( args[7], _ascanf_array, caller, (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (array for returning speed at each sample time is required) ";
		ascanf_arg_error= True;
	}
	if( ASCANF_ARG_TRUE(8) ){
		sumPos= parse_ascanf_address( args[8], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(9) ){
		sumNeg= parse_ascanf_address( args[9], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(10) ){
		sDTpos= parse_ascanf_address( args[10], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(11) ){
		sDTneg= parse_ascanf_address( args[11], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(12) ){
		Samples= parse_ascanf_address( args[12], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(13) ){
		sumPosHR= parse_ascanf_address( args[13], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(14) ){
		sumNegHR= parse_ascanf_address( args[14], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(15) ){
		sDTposHR= parse_ascanf_address( args[15], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}
	if( ASCANF_ARG_TRUE(16) ){
		sDTnegHR= parse_ascanf_address( args[16], _ascanf_array, caller, (int) ascanf_verbose, NULL );
	}

	if( !ascanf_arg_error ){
		Resize_ascanf_Array(sampleT, Time->N, NULL );
		Resize_ascanf_Array(Delta, Time->N, NULL );
		Resize_ascanf_Array(DDelta, Time->N, NULL );
		Resize_ascanf_Array(sumPos, Time->N, NULL );
		Resize_ascanf_Array(sumNeg, Time->N, NULL );
		Resize_ascanf_Array(sDTpos, Time->N, NULL );
		Resize_ascanf_Array(sDTneg, Time->N, NULL );
		Resize_ascanf_Array(Samples, Time->N, NULL );
		Resize_ascanf_Array(sumPosHR, Time->N, NULL );
		Resize_ascanf_Array(sumNegHR, Time->N, NULL );
		Resize_ascanf_Array(sDTposHR, Time->N, NULL );
		Resize_ascanf_Array(sDTnegHR, Time->N, NULL );
	}

	set_NaN(*result);
	if( !ascanf_arg_error ){
	  double t= 0, pt= 0;

		posVolantPred.i16= 0;
		dureeCycle1000= (unsigned int) (1000 / freq );
		tmpTimer= 0;
		tempsAbs= 0;
		tpsEnMilliSec= 0;
		sVP=0, sVN=0, sTP=0, sTN=0, NN= 0;
		sommeVitesseNeg= sommeVitessePos = 0;
		sommeDTPos= 0;
		sommeDTNeg = 0;
		nbrEchantillonT = 0;
		nbrEchantillon0 = 0;
		tempsPredPos= 0;
		tempsPredNeg= 0;

		for( i= 0, n= 0; i< Time->N; i++ ){
		  double interval, skipTime= loopdelay;

			// emulate a delay due to sending data on the CANbus
			if( dixms ){
				skipTime+= delay;
			}
			if( i && skipTime> 0 ){
			  double nan;
				set_NaN(nan);
				while( i< Time->N && ASCANF_ARRAY_ELEM(Time,i)< t+skipTime ){
					if( sumPosHR ){
						ASCANF_ARRAY_ELEM_SET( sumPosHR, i, nan );
					}
					if( sumNegHR ){
						ASCANF_ARRAY_ELEM_SET( sumNegHR, i, nan );
					}
					if( sDTposHR ){
						ASCANF_ARRAY_ELEM_SET( sDTposHR, i, nan );
					}
					if( sDTnegHR ){
						ASCANF_ARRAY_ELEM_SET( sDTnegHR, i, nan );
					}
					i+= 1;
				}
			}

			pt= t;
			t= ASCANF_ARRAY_ELEM(Time,i);
			if( i ){
				interval= 500 * 1000 * (t - pt);
			}
			else{
				interval= 500 * 1000 * t;
			}

			dixms= 0;
			tmpTimer += (unsigned int) interval;
			tempsAbs += (unsigned int) interval;
			pVolant->i16= ASCANF_ARRAY_ELEM(Pos,i);

			// RJVB: la RAZ de tempsAbs se fait ici, pour pouvoir preserver la difference
			// tempsAbs-tempsPred (calculee en GestionCab)
			if( resetTemps ){
				if( avec_tempsPred ){
					extern NECtimer tempsPredPos, tempsPredNeg;
					NECtimer dtP, dtN;
					if( tempsPredPos < tempsAbs ){
						// RJVB: s'assurer que la difference tempsAbs - tempsPred reste inchangee
						dtP= (unsigned) ((NECtimer)tempsAbs - tempsPredPos);
					}
					else{
						dtP= 0;
					}
					if( tempsPredNeg < tempsAbs ){
						// RJVB: s'assurer que la difference tempsAbs - tempsPred reste inchangee
						dtN= (unsigned) ((NECtimer)tempsAbs - tempsPredNeg);
					}
					else{
						dtN= 0;
					}
					if( dtP> dtN ){
						tempsAbs = dtP;
						tempsPredPos= 0;
						tempsPredNeg= tempsAbs - dtN;
					}
					else{
						tempsAbs = dtN;
						tempsPredNeg= 0;
						tempsPredPos= tempsAbs - dtP;
					}
				}
				else{
					tempsAbs= 0;
				}
				resetTemps = 0;
			}

#define une_milliseconde	500
			while (tmpTimer >= une_milliseconde)
			{
				tmpTimer -= une_milliseconde;
				tpsEnMilliSec += 1;
			}
			if ((tpsEnMilliSec - tpsEnMilliSec10ms) >= dureeCycle1000)
			{
				dixms = 1;
				tpsEnMilliSec10ms = tpsEnMilliSec10ms + dureeCycle1000;
			}
			
			if( GestionCabine( (NECtimer) tpsEnMilliSec, (NECtimer) tempsAbs) ){
				dixms = 1;
			}

			if( sumPosHR ){
				ASCANF_ARRAY_ELEM_SET( sumPosHR, i, sommeVitessePos );
			}
			if( sumNegHR ){
				ASCANF_ARRAY_ELEM_SET( sumNegHR, i, sommeVitesseNeg );
			}
			if( sDTposHR ){
				ASCANF_ARRAY_ELEM_SET( sDTposHR, i, sommeDTPos );
			}
			if( sDTnegHR ){
				ASCANF_ARRAY_ELEM_SET( sDTnegHR, i, sommeDTNeg );
			}

			if( GestionCom( dixms, 0, tempsAbs ) ){
				if( dixms ){
					if( n< i ){
					  // output calculated values ("on the CANbus")
						ASCANF_ARRAY_ELEM_SET( sampleT, n, t );
						ASCANF_ARRAY_ELEM_SET( Delta, n, pVolant->i16 );
						{ double Vp= (sTP)? (double)(sVP)/(double)(sTP) : 0,
								Vn= (sTN)? (double)(sVN)/(double)(sTN) : 0;
							ASCANF_ARRAY_ELEM_SET( DDelta, n, (Vp - Vn) * 500000.0 );
						}
						if( sumPos ){
							ASCANF_ARRAY_ELEM_SET( sumPos, n, sVP );
						}
						if( sumNeg ){
							ASCANF_ARRAY_ELEM_SET( sumNeg, n, sVN );
						}
						if( sDTpos ){
							ASCANF_ARRAY_ELEM_SET( sDTpos, n, sTP );
						}
						if( sDTneg ){
							ASCANF_ARRAY_ELEM_SET( sDTneg, n, sTN );
						}
						if( Samples ){
							ASCANF_ARRAY_ELEM_SET( Samples, n, NN );
						}
						n+= 1;
					}
					if( sizeof(NECtimer)<=8 ){
						resetTemps= 1;
					}
				}
			}
		}
		if( n< Time->N ){
			Resize_ascanf_Array(sampleT, n, NULL );
			Resize_ascanf_Array(Delta, n, NULL );
			Resize_ascanf_Array(DDelta, n, NULL );
			Resize_ascanf_Array(sumPos, n, NULL );
			Resize_ascanf_Array(sumNeg, n, NULL );
			Resize_ascanf_Array(sDTpos, n, NULL );
			Resize_ascanf_Array(sDTneg, n, NULL );
			Resize_ascanf_Array(Samples, n, NULL );
		}
		*result= n;
	}
	return(!ascanf_arg_error);
}


static ascanf_Function ddeltaNEC_Function[] = {
	{ "NEC", runNEC, 17, NOT_EOF_OR_RETURN,
		"NEC[&t,&pos,freq,loopTime,sendDelay,&sample_t,&delta,&ddelta[,&sumPos,&sumNeg,&sDTpos,&sDTneg,&N[,&sumPosHR,&sumNegHR,&sDTposHR,&sDTnegHR]]]\n"
		" \n"
	},
	  /* AMAXARGS is actually -1. This signals to ascanfc.c that the current maximum number of arguments
	   \ should be allowed (or required...).
	   */
};
static int ddeltaNEC_Functions= sizeof(ddeltaNEC_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= ddeltaNEC_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< ddeltaNEC_Functions; i++, af++ ){
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

  /* This is the crucial function. It is called by xgraph's module loader (and should thus be called as shown here).
   \ It is charged with installing the (ascanf) code we provide into the ascanf function tables (on all platforms).
   \ On other platforms, it is also charged with obtaining the pointers to symbols and functions *we* need. For this
   \ it is passed the initialise argument; see below. It is safe to call this function on all platforms.
   \ 20040502: this, and the closeDyMod() routine may also have their name prepended with the module's "basename"
   \ (e.g. ddeltaNEC_initDyMod here). That means of course that the module *must* be installed using that name (plus
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
		  /* The XGRAPH_FUNCTION macro can be used to easily initialise the additional variables we need.
		   \ In line with the bail out remark above, this macro returns DM_Error when anything goes wrong -
		   \ i.e. aborts initDyMod!
		   */
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, new->name, new->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		  /* Perform all initialisations that are necessary. */
		af_initialise( new, new->name );
		  /* And now add the functions we provide! */
		add_ascanf_functions( ddeltaNEC_Function, ddeltaNEC_Functions, "ddeltaNEC::initDyMod()" );
		initialised= True;
	}
	new->libHook= NULL;
	new->libname= XGstrdup( "ddeltaNEC" );
	new->buildstring= XGstrdup(XG_IDENTIFY());
	new->description= XGstrdup(
		" A dynamic module (library) that contains\n"
		" some test code for a Nec V853 microcontroller.\n"
	);
	return( DM_Ascanf );
}

// see the explanation printed by wrong_dymod_loaded():
void initddeltaNEC()
{
	wrong_dymod_loaded( "initddeltaNEC()", "Python", "ddeltaNEC.so" );
}

// see the explanation printed by wrong_dymod_loaded():
void R_init_ddeltaNEC()
{
	wrong_dymod_loaded( "R_init_ddeltaNEC()", "R", "ddeltaNEC.so" );
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
	  int r= remove_ascanf_functions( ddeltaNEC_Function, ddeltaNEC_Functions, force );
		if( force || r> 0 ){
			for( i= 0; i< ddeltaNEC_Functions; i++ ){
				ddeltaNEC_Function[i].dymod= NULL;
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
int _init()
{ static int called= 0;
	fprintf( stderr, "%s::_init(): call #%d\n", __FILE__, ++called );
}

int _fini()
{ static int called= 0;
	fprintf( stderr, "%s::_fini(): call #%d\n", __FILE__, ++called );
}

