#include "config.h"
IDENTIFY( "GSRio import library module for .gsr/RED1995 datafiles" );

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
char* (*time_stamp_ptr)( FILE *fp, char *name, char *buf, int verbose, char *postfix);

#	define realloc_columns	(*realloc_columns_ptr)
#	define realloc_points	(*realloc_points_ptr)
#	define Parse_SetLabelsList	(*Parse_SetLabelsList_ptr)
#	define Add_LabelsList	(*Add_LabelsList_ptr)
#	define LabelsList_N	(*LabelsList_N_ptr)
#	define time_stamp	(*time_stamp_ptr)

#include "Import/gsr.h"

static ascanf_Function GSRio_Function[] = {
	{ "$GSR-Import-Prunes-Empty", NULL, 2, _ascanf_variable,
		"$GSR-Import-Prunes-Empty: when set, empty channels are removed during the import,\n"
		" that would otherwise show up as columns of zeroes in the target dataset.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$GSR-Import-Original-Segment-Duration", NULL, 2, _ascanf_variable,
		"$GSR-Import-Original-Segment-Duration: .GSR files are likely to have been created through\n"
		" a DOS utility that can't sample for long durations, and that will thus have dumped the\n"
		" data in segments of (theoretically) equal duration (except for the last). This variable\n"
		" allows to specify that duration (this value can be found in the KK.RED file that should\n"
		" accompany the KK.GSR file). If set, markers will be added to the dataset's events column\n"
		" that indicate the end of one segment (-n) and the beginning of the next (-n-1)\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$GSR-Import-Column-Selector", NULL, 2, _ascanf_variable,
		"$GSR-Import-Column-Selector: point this variable to an array specifying which columns to import.\n"
		" The array may be floating point or integer, but should enumerate the column numbers (i.e. do not\n"
		" use flags per column). Invalid values are silently ignored.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$GSR-Import-Feedback", NULL, 2, _ascanf_variable,
		"$GSR-Import-Feedback: shows and stores information about the import.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
};
static int GSRio_Functions= sizeof(GSRio_Function)/sizeof(ascanf_Function);

static double *PruneEmptyChannels= &GSRio_Function[0].value;
static double *Original_Segment_Duration= &GSRio_Function[1].value;
static double *ColumnSelector= &GSRio_Function[2].value;
static double *ImportFeedback= &GSRio_Function[3].value;

DM_IO_Handler GSRio_Handlers;

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= GSRio_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< GSRio_Functions; i++, af++ ){
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

/* 20040210 02:15 : skeleton ready: data is imported seemingly OK. Now, we must do something with it...! */
/* 20040211 12:30 : conversion from raw samples to real values more or less figured out... 	*/

#ifndef sgi
#	define READ_ATONCE
#endif

int import_GSR(FILE *stream, char *the_file, int filenr, int setNum, struct DataSet *this_set, ReadData_States *state )
{ GSRHeaders gheader;
  GSRHeadersRaw gheaderRaw;
  GSRChannelHeaders *cheader= NULL;
  int ret= 0;
	memset( &gheaderRaw, 0, sizeof(gheaderRaw) );
#ifdef READ_ATONCE
	if( stream && fread( &gheaderRaw, sizeof(gheaderRaw), 1, stream)== 1 )
#else
	if( stream && fread( &gheaderRaw.magic, sizeof(gheaderRaw.magic), 1, stream)== 1 )
#endif
	{
		  /* Attention: gheader.magic is likely *not* to be null-terminated! We do, however, force the padding
		   \ field to be null-terminated at it's end; that way, we're sure that gheader *can* be printed as a string.
		   */
		gheaderRaw.padding[ sizeof(gheaderRaw.padding)-1 ]= '\0';
		if( strncmp( gheaderRaw.magic, "RED 1995", 8)== 0 ){
		  GSREventRecords Event;
		  int event;
		  size_t nread= 0, ncalls= 0;
			memset( &gheader, 0, sizeof(gheader) );
#ifndef READ_ATONCE
			nread+= fread( &gheader.NChannels, sizeof(gheader.NChannels), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.TResolution, sizeof(gheader.TResolution), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.TResolutionCorrection, sizeof(gheader.TResolutionCorrection), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.Samples, sizeof(gheader.Samples), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.theChannel, sizeof(gheader.theChannel), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.CursorPosition, sizeof(gheader.CursorPosition), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.FirstShownSample, sizeof(gheader.FirstShownSample), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.ShownSamples, sizeof(gheader.ShownSamples), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.FirstLoadedSample, sizeof(gheader.FirstLoadedSample), 1, stream ), ncalls+= 1;
			  /* ought to be float?! No idea what it represents! */
			nread+= fread( &gheader.AbsoluteTime, sizeof(gheader.AbsoluteTime), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.NEvents, sizeof(gheader.NEvents), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.GridState, sizeof(gheader.GridState), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.bgColour, sizeof(gheader.bgColour), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.TracingSpeed, sizeof(gheader.TracingSpeed), 1, stream ), ncalls+= 1;
			nread+= fread( &gheader.padding, sizeof(gheader.padding), 1, stream ), ncalls+= 1;
			gheader.padding[ sizeof(gheader.padding)-1 ]= '\0';
			if( nread!= ncalls ){
				goto no_valid_gsr;
			}
#else
			  /* we don't copy the magic field: not really necessary... */
			gheader.NChannels= *( (GSRshort*) gheaderRaw.NChannels );
			gheader.TResolution= *( (float*) gheaderRaw.TResolution );
			gheader.TResolutionCorrection= *( (double*) gheaderRaw.TResolutionCorrection );
			gheader.Samples= *( (GSRlong*) gheaderRaw.Samples );
			gheader.theChannel= *( (GSRshort*) gheaderRaw.theChannel );
			gheader.CursorPosition= *( (GSRlong*) gheaderRaw.CursorPosition );
			gheader.FirstShownSample= *( (GSRlong*) gheaderRaw.FirstShownSample );
			gheader.ShownSamples= *( (GSRlong*) gheaderRaw.ShownSamples );
			gheader.FirstLoadedSample= *( (GSRlong*) gheaderRaw.FirstLoadedSample );
			  /* ought to be float?! No idea what it represents! */
			gheader.AbsoluteTime.dAT= *( (double*) gheaderRaw.AbsoluteTime );
			gheader.NEvents= *( (GSRshort*) gheaderRaw.NEvents );
			gheader.GridState= *( (GSRshort*) gheaderRaw.GridState );
			gheader.bgColour= *( (GSRshort*) gheaderRaw.bgColour );
			gheader.TracingSpeed= *( (GSRshort*) gheaderRaw.TracingSpeed );
#endif
			  /* We honour requests to correct the endianness, but we do know that this format is PC-native,
			   \ so we also do the correction if it turns out to be necessary! Unasked.
			   */
			if( SwapEndian || EndianType!= 1 ){
				SwapEndian_int16( &gheader.NChannels, 1 );
				SwapEndian_float( &gheader.TResolution, 1 );
				SwapEndian_double( &gheader.TResolutionCorrection, 1 );
				SwapEndian_int32( (int32_t*) &gheader.Samples, 1 );
				SwapEndian_int16( &gheader.theChannel, 1 );
				SwapEndian_int32( (int32_t*) &gheader.CursorPosition, 1 );
				SwapEndian_int32( (int32_t*) &gheader.FirstShownSample, 1 );
				SwapEndian_int32( (int32_t*) &gheader.ShownSamples, 1 );
				SwapEndian_int32( (int32_t*) &gheader.FirstLoadedSample, 1 );
/* 				SwapEndian_double( &gheader.AbsoluteTime.dAT, 1 );	*/
				SwapEndian_float( gheader.AbsoluteTime.fAT, 2 );
				SwapEndian_int16( &gheader.NEvents, 1 );
				SwapEndian_int16( &gheader.GridState, 1 );
				SwapEndian_int16( &gheader.bgColour, 1 );
				SwapEndian_int16( &gheader.TracingSpeed, 1 );
			}
			cheader= (GSRChannelHeaders*) calloc( gheader.NChannels, sizeof(GSRChannelHeaders) );
			if( cheader ){
			  GSRSamples **sample= NULL;
			  int channel;
			  LabelsList *llist= this_set->ColumnLabels;
			  int poffset= this_set->numPoints;
			  GSRshort NChannels= 0;

				if( (sample= (GSRSamples**) calloc( gheader.NChannels, sizeof(GSRSamples*) )) ){
				  char fbbuf[1024];
				  int fblen= sizeof(fbbuf)-1;
				  Sinc sinc;
				  ascanf_Function *af= NULL;
				  __ALLOCA( ImportColumn, short, gheader.NChannels, IClen );

					Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
					Sflush(&sinc);
					Sputs( this_set->set_info, &sinc );

					{ int i, c;
						if( *ColumnSelector && (af=
								parse_ascanf_address( *ColumnSelector, _ascanf_array, "GSRio::import_GSR()",
									ascanf_verbose, NULL))
						){
							memset( ImportColumn, 0, IClen );
							for( i= 0; i< af->N; i++ ){
								if( (c= ASCANF_ARRAY_ELEM(af,i))>= 0 && c< gheader.NChannels ){
									ImportColumn[c]= 1;
								}
							}
						}
						else{
							for( i= 0; i< gheader.NChannels; i++ ){
								ImportColumn[i]= 1;
							}
						}
					}

					if( *ImportFeedback ){
					  ALLOCA( buf, char, strlen(the_file)+256, blen );
						time_stamp( stream, the_file, buf, True, "\n" );
						snprintf( fbbuf, fblen,
							"GSRio::import_GSR(): reading %d samples on %d channels, T=0-%g @ %gHz from\n"
							" %s",
							gheader.Samples, gheader.NChannels,
							gheader.Samples * gheader.TResolution/ gheader.TResolutionCorrection,
							1.0/ (gheader.TResolution/ gheader.TResolutionCorrection),
							buf
						);
						Sputs( fbbuf, &sinc );
						if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
							fputs( fbbuf, StdErr );
							if( af ){
								fprintf( StdErr, "\tColumnSelector: '%s'[%d]\n", af->name, af->N );
							}
						}
						GCA();
					}

					  /* Read the sample data from file in a loop. We don't check for EOF or file error in the loop,
					   \ but perform it for all channels; thus, the user cannot avoid knowing that *no* data was read
					   \ if things went awry in the first channel.
					   \ Note that this serial format doesn't really permit to recover graciously from errors: we can't
					   \ expect to read a correct channel header after a read failure. It may be that all channels have the
					   \ same number of samples that is different from gheader.Samples, but there is no way we can attempt
					   \ to validate such a guess without risking a crash. Future version may have an option to correct the
					   \ gheader.Samples value if this turns out to be of interest.
					   */
					for( channel= 0; channel< gheader.NChannels; channel++ ){
					  GSRChannelHeaders *ch= &cheader[channel];
#ifndef READ_ATONCE
						memset( ch, 0, sizeof(GSRChannelHeaders) );
						nread= ncalls= 0;
						nread+= fread( ch->ChannelName, sizeof(ch->ChannelName), 1, stream ), ncalls+= 1;
						nread+= fread( &ch->ChannelCoefficient, sizeof(ch->ChannelCoefficient), 1, stream ), ncalls+= 1;
						nread+= fread( &ch->ChannelOffset, sizeof(ch->ChannelOffset), 1, stream ), ncalls+= 1;
						nread+= fread( &ch->SelectedGain, sizeof(ch->SelectedGain), 1, stream ), ncalls+= 1;
						nread+= fread( &ch->SelectedOffset, sizeof(ch->SelectedOffset), 1, stream ), ncalls+= 1;
						nread+= fread( &ch->ChannelState, sizeof(ch->ChannelState), 1, stream ), ncalls+= 1;
						nread+= fread( &ch->ChannelColour, sizeof(ch->ChannelColour), 1, stream ), ncalls+= 1;
						nread+= fread( ch->ChannelUnits, sizeof(ch->ChannelUnits), 1, stream ), ncalls+= 1;
						nread+= fread( ch->padding, sizeof(ch->padding), 1, stream ), ncalls+= 1;
						if( nread== ncalls )
#else
					  GSRChannelHeadersRaw cheaderRaw;
						if( fread( &cheaderRaw, sizeof(cheaderRaw), 1, stream)== 1 )
#endif
						{
#ifdef READ_ATONCE
							memset( ch, 0, sizeof(GSRChannelHeaders) );
							  /* Now initialise the fields. We could skip the ChannelName, in fact... */
							memcpy( ch->ChannelName, cheaderRaw.ChannelName, sizeof(ch->ChannelName) );
							ch->ChannelCoefficient= *( (GSRshort*) cheaderRaw.ChannelCoefficient );
							ch->ChannelOffset= *( (GSRshort*) cheaderRaw.ChannelOffset );
							ch->SelectedGain= *( (float*) cheaderRaw.SelectedGain );
							ch->SelectedOffset= *( (float*) cheaderRaw.SelectedOffset );
							ch->ChannelState= *( (GSRshort*) cheaderRaw.ChannelState );
							ch->ChannelColour= *( (GSRshort*) cheaderRaw.ChannelColour );
							memcpy( ch->ChannelUnits, cheaderRaw.ChannelUnits, sizeof(ch->ChannelUnits) );
#endif
							if( SwapEndian || EndianType!= 1 ){
								SwapEndian_int16( &ch->ChannelCoefficient, 1 );
								SwapEndian_int16( &ch->ChannelOffset, 1 );
								SwapEndian_float( &ch->SelectedGain, 1 );
								SwapEndian_float( &ch->SelectedOffset, 1 );
								SwapEndian_int16( &ch->ChannelState, 1 );
								SwapEndian_int16( &ch->ChannelColour, 1 );
							}

							sample[channel]= (GSRSamples*) calloc( gheader.Samples, sizeof(GSRSamples) );
							if( sample[channel] &&
								(nread= fread( sample[channel], sizeof(GSRSamples), gheader.Samples, stream))== gheader.Samples
							){
							  int empty= True;
								if( SwapEndian || EndianType!= 1 ){
									SwapEndian_int16( sample[channel], nread );
								}

								ch->ChannelName[ sizeof(ch->ChannelName)-1 ]= '\0';
								ch->ChannelUnits[ sizeof(ch->ChannelUnits)-1 ]= '\0';
								cleanup( ch->ChannelName );
								cleanup( ch->ChannelUnits );

								if( !ImportColumn[channel] ){
									xfree( sample[channel] );
								}
								else if( *PruneEmptyChannels ){
								  int i;
									for( i= 0; i< gheader.Samples && empty; i++ ){
										if( sample[channel][i] ){
											empty= False;
										}
									}
									if( empty ){
										xfree( sample[channel] );
									}
								}

								if( *ImportFeedback ){
									if( sample[channel] ){
									  SimpleStats rSS, SS;
									  int i;
									  double calib0= ch->ChannelCoefficient/ch->SelectedGain;
									    /* 20040307: clearly, the above formula doesn't correctly convert all
									     \ possible gsr files. The calibration factor below is what is used
										\ in most gsr files I've seen, and it does give the correct values.
										\ So for now, we ignore the SelectedGain,Offset (called 'facteur de reglage'
										\ of gain and offset in the original format description). I need more
										\ data to make a more educated guess as to how to use these 4 values
										\ in an operation for which 2 would be enough. NB: it is suspicious that
										\ the 'best' SelectedGain, 3276.8 is exactly 2**16 / 20.0 ....
										\ NB2: the actual conversion is done below; here it is only for feedback
										\ purposes!!!!!!
										*/
									  double calib= ch->ChannelCoefficient/3276.8;
										SS_Init_(rSS);
										SS_Init_(SS);
										for( i= 0; i< nread; i++ ){
											SS_Add_Data_( rSS, 1, sample[channel][i], 1.0 );
											SS_Add_Data_( SS, 1,
												sample[channel][i]* calib + ch->ChannelOffset,
												1.0 );
										}
										snprintf( fbbuf, fblen, "GSRio::import_GSR(%s): channel %d: read %d samples%s:\n"
											"\tName/Units   : %s [%s]\n"
												"\tCoeff/Offset : %d/%d\t\tRaw : %s\n"
													"\tSetting (g/o): %g/%g\t\tReal: %s\n"
 														"\t  conversion : <sample> * %d/%g + %d%s\n"
															"\tState/Colour : %d/%d\n",
											the_file, channel, nread, (empty)? " (empty!)" : "",
											ch->ChannelName, ch->ChannelUnits,
												ch->ChannelCoefficient, ch->ChannelOffset,
												SS_sprint_full( NULL, "%g", " #xb1 ", 0, &rSS ),
													ch->SelectedGain, ch->SelectedOffset,
													SS_sprint_full( NULL, "%g", " #xb1 ", 0, &SS ),
 														ch->ChannelCoefficient, 3276.8, ch->ChannelOffset,
 														(( floor(10*ch->SelectedGain)!=32768)? "; Setting values not used" :
 																"; Setting offset not used"),
															ch->ChannelState, ch->ChannelColour
										);
									}
									else{
										snprintf( fbbuf, fblen, "GSRio::import_GSR(%s): channel %d: read %d samples"
											" -- empty or deselected by request.\n",
											the_file, channel, nread
										);
									}
									StringCheck( fbbuf, 1024, __FILE__, __LINE__ );
									Sputs( fbbuf, &sinc );
									if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
										fputs( fbbuf, StdErr );
									}
								}

								if( sample[channel] ){
									NChannels+= 1;
									ret+= gheader.Samples;
								}
							}
							else{
								fprintf( StdErr,
									"GSRio::import_GSR(%s): allocation or read problem reading data for channel %d"
									": read %d of %d entries (%s)\n",
									the_file, channel, nread, gheader.Samples, serror()
								);
							}
						}
						else{
							fprintf( StdErr, "GSRio::import_GSR(%s): read problem reading header for channel %d (%s)\n",
								the_file, channel, serror()
							);
						}
					}

					  /* set the proper set size, supposing that we'll get at least some of the Events
					   \ (we'll read those later). For the sake of comparable output, we always allocate
					   \ an events column. NB: also allocate a column for the time stamps...!
					   */

					{ int i, j, tchannel;
					  int N;
						this_set->numPoints+= gheader.Samples;
						realloc_points( this_set, this_set->numPoints, False );
						if( this_set->ncols< NChannels+2 ){
							this_set->columns= realloc_columns( this_set, NChannels+2 );
						}

						N= LabelsList_N( llist );
						llist= Add_LabelsList( llist, &N, 0, "Time [s]" );
						for( i= 0, j= poffset; i< gheader.Samples; i++, j++ ){
							this_set->columns[0][j]= i* gheader.TResolution/ gheader.TResolutionCorrection;
							this_set->columns[NChannels+1][j]= 0;
						}
						for( channel= 0, tchannel= 1; channel< gheader.NChannels; channel++ ){
						  GSRChannelHeaders *ch= &cheader[channel];
						  double calib0= ch->ChannelCoefficient/ch->SelectedGain;
						  double calib= ch->ChannelCoefficient/3276.8;

							if( sample[channel] ){
								snprintf( fbbuf, fblen, "%s [%s]", ch->ChannelName, ch->ChannelUnits );
								StringCheck(fbbuf, fblen, __FILE__, __LINE__);
								llist= Add_LabelsList( llist, &N, tchannel, fbbuf );
								for( i= 0, j= poffset; i< gheader.Samples; i++, j++ ){
									this_set->columns[tchannel][j]= sample[channel][i]* calib + ch->ChannelOffset;
								}
								  /* Finally, get rid of now unneeded memory: */
								xfree( sample[channel] );
								tchannel+= 1;
							}
						}
#if ADVANCED_STATS == 1
						this_set->N[*(state->spot)]= 1;
#endif

						*(state->spot)++;
						*(state->Spot)++;
						llist= Add_LabelsList( llist, &N, tchannel, "Events" );
					}
					xfree( sample );

					this_set->ColumnLabels= llist;

					{ XGStringList *EventList= NULL;
					  int nTop_event= 2;
						  /* Save a bit on memory here: import the events one at a time. */
						if( gheader.NEvents && *ImportFeedback ){
							if( (pragma_unlikely(ascanf_verbose) || scriptVerbose) ){
								fprintf( StdErr, "GSRio::import_GSR(%s): %d events: T=",
									the_file, gheader.NEvents
								);
								fflush(StdErr);
							}
							snprintf( fbbuf, fblen, "GSRio::import_GSR(%s): %d events:",
								the_file, gheader.NEvents
							);
							Sputs( fbbuf, &sinc );
						}
						for( event= 0; event< gheader.NEvents; event++ ){
							memset( &Event, 0, sizeof(Event) );
							if( fread( &Event, sizeof(Event), 1, stream)== 1 ){
								if( SwapEndian || EndianType!= 1 ){
									SwapEndian_int32( (int32_t*) &Event.EventPosition, 1 );
									SwapEndian_int32( (int32_t*) &Event.EventColour, 1 );
								}
								Event.EventText[ sizeof(Event.EventText)-1 ]= '\0';
								cleanup( Event.EventText );

								if( Event.EventText[0] ){
								  int nt;
									  /* This part is unchecked!!! */
									if( !XGStringList_FindItem( EventList, Event.EventText, &nt ) ){
										  /* previously unseen event type: store it in the list and attribute a new
										   \ number (should be its entry in the list) using our local counter.
										   */
										EventList= XGStringList_AddItem( EventList, Event.EventText );
										this_set->columns[NChannels+1][ Event.EventPosition ]= nTop_event;
										nTop_event+= 1;
									}
									else{
										  /* previously seen, XGStringList_FindItem will have given the entry. Add
										   \ 2 (as entry 0 corresponds to event type 2), and use that.
										   */
										this_set->columns[NChannels+1][ Event.EventPosition ]= nt+2;
									}
								}
								else{
									  /* Store a TOP event (one without event text associated) */
									this_set->columns[NChannels+1][ Event.EventPosition ]= 1;
								}

								if( *ImportFeedback ){
									if( (pragma_unlikely(ascanf_verbose) || scriptVerbose) ){
										fprintf( StdErr, " %g",
											Event.EventPosition* gheader.TResolution/gheader.TResolutionCorrection
										);
										if( Event.EventText[0] ){
											fprintf( StdErr, "[%s]", Event.EventText );
										}
									}
									if( event== 0 ){
										snprintf( fbbuf, fblen, " first at T=%g",
											Event.EventPosition* gheader.TResolution/gheader.TResolutionCorrection
										);
										Sputs( fbbuf, &sinc );
										if( Event.EventText[0] ){
											snprintf( fbbuf, fblen, "[%s]", Event.EventText );
											Sputs( fbbuf, &sinc );
										}
									}
									else if( event== gheader.NEvents-1 ){
										snprintf( fbbuf, fblen, " last at T=%g",
											Event.EventPosition* gheader.TResolution/gheader.TResolutionCorrection
										);
										Sputs( fbbuf, &sinc );
										if( Event.EventText[0] ){
											snprintf( fbbuf, fblen, "[%s]", Event.EventText );
											Sputs( fbbuf, &sinc );
										}
										Sputs( "\n", &sinc );
									}
								}

								ret+= 1;
							}
						}
						if( event && *ImportFeedback && (pragma_unlikely(ascanf_verbose) || scriptVerbose) ){
							fputs( "\n", StdErr );
						}
						if( EventList ){
							if( *ImportFeedback ){
							  XGStringList *list= EventList;
							  int nr= 2;
								snprintf( fbbuf, fblen, "GSRio::import_GSR(%s): non-TOP events, value=[event]:",
									the_file
								);
								Sputs( fbbuf, &sinc );
								if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
									fputs( fbbuf, StdErr );
									fflush(StdErr);
								}
								while( list ){
									snprintf( fbbuf, fblen, " %d=[%s]", nr, list->text );
									Sputs( fbbuf, &sinc );
									if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
										fputs( fbbuf, StdErr );
										fflush(StdErr);
									}
									list= list->next;
									nr+= 1;
								}
								Sputs( "\n", &sinc );
								if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
									fputs( "\n", StdErr );
								}
							}
							  /* Use list here ? */
							EventList= XGStringList_Delete( EventList );
						}
					}

					if( *Original_Segment_Duration ){
					  double *evcolumn= this_set->columns[NChannels+1];
					  double timebase= gheader.TResolution/gheader.TResolutionCorrection;
					  GSRlong start, finish, length;
					  int segment= 1;
					  /* Now store the segment number markers into the event column. This should be straightforward.
					   \ If there is an event marker at exactly the required location, this marker is removed, but
					   \ a warning is written on StdErr *and* in the setinfo.
					   \ Segment 1 will have a -1 on its first and last sample; segment n will have a -n there.
					   */
						start= 0;
						length= (GSRlong) ( *Original_Segment_Duration/ timebase + 0.5 );
						finish= length;
						do{
							if( evcolumn[start+poffset] ){
								snprintf( fbbuf, fblen,
									"GSRio::import_GSR(%s): segment %d start marker at t=%g hides event type=%g !\n",
									the_file, segment, start* timebase, evcolumn[start+poffset]
								);
								Sputs( fbbuf, &sinc );
								if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
									fputs( fbbuf, StdErr );
								}
							}
							evcolumn[start+poffset]= -segment;
							if( finish>= 0 && finish< gheader.Samples ){
								if( evcolumn[finish+poffset] ){
									snprintf( fbbuf, fblen,
										"GSRio::import_GSR(%s): segment %d end marker at t=%g hides event type=%g !\n",
										the_file, segment, finish* timebase, evcolumn[finish+poffset]
									);
									Sputs( fbbuf, &sinc );
									if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
										fputs( fbbuf, StdErr );
									}
								}
								evcolumn[finish+poffset]= -segment;
							}
							start= finish+1;
							finish= start+length;
							segment+= 1;
						} while( start>= 0 && start< gheader.Samples );
						if( *ImportFeedback ){
							snprintf( fbbuf, fblen, "GSRio::import_GSR(%s): marked %d segments in events column\n",
								the_file, segment-1
							);
							Sputs( fbbuf, &sinc );
							if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
								fputs( fbbuf, StdErr );
							}
						}
					}

					if( sinc.sinc.string ){
						xfree( this_set->set_info );
						this_set->set_info= sinc.sinc.string;
					}
					
					if( !this_set->setName ){
						this_set->setName= concat( the_file, " ", "%CY", NULL );
					}

				}
				else{
					fprintf( StdErr, "GSRio::import_GSR(%s): can't allocate memory for channel samples! (%s)\n",
						the_file, serror()
					);
				}

				xfree( cheader );
			}
			else{
				fprintf( StdErr, "GSRio::import_GSR(%s): can't allocate memory for channel headers! (%s)\n",
					the_file, serror()
				);
			}
		}
		else{
#ifndef READ_ATONCE
no_valid_gsr:;
#endif
			fprintf( StdErr, "GSRio::import_GSR(%s): this doesn't seem to be a GSR file (<magic>\"%s\"</magic>)\n",
				the_file, gheaderRaw.magic
			);
		}
	}
	else{
		fprintf( StdErr, "GSRio::import_GSR(%s): file not open or other read problem (%s)\n",
			the_file, serror()
		);
	}
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
		XGRAPH_FUNCTION(time_stamp_ptr, "time_stamp");
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( GSRio_Function, GSRio_Functions, "utils::initDyMod()" );
		initialised= True;
	}
	  /* Initialise the library hook. For now, we only provide the import routine. */
	GSRio_Handlers.type= DM_IO;
	GSRio_Handlers.import= import_GSR;
	theDyMod->libHook= (void*) &GSRio_Handlers;
	theDyMod->libname= XGstrdup( "DM-GSRio" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" I/O facilities for the \".gsr\" dataformat.\n"
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
	  int r= remove_ascanf_functions( GSRio_Function, GSRio_Functions, force );
		if( force || r== GSRio_Functions ){
			for( i= 0; i< GSRio_Functions; i++ ){
				GSRio_Function[i].dymod= NULL;
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
				r, GSRio_Functions
			);
		}
	}
	return(ret);
}

// see the explanation printed by wrong_dymod_loaded():
void initGSRio()
{
	wrong_dymod_loaded( "initGSRio()", "Python", "GSRio.so" );
}

// see the explanation printed by wrong_dymod_loaded():
void R_init_GSRio()
{
	wrong_dymod_loaded( "R_init_GSRio()", "R", "GSRio.so" );
}

