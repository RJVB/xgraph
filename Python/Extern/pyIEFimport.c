//IDENTIFY( "IEFio import XGraph library module for .IEF datafiles" );

#include <stdio.h>
#include <stdlib.h>

#define	StdErr	stderr

#include "SS.h"

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc). On Cygwin, the --export-all-symbols LINKER OPTION
   \ seems to play the same role.
   */

#include <float.h>

#include "ief.h"

IEFsensorSpecs
		CBFsensors[]= {
						{11, 1, 390, 1. * 360},	// steering angle in degrees, please
						{8, 1, 0, 0.0314},
						{8, 1, 0, 0.0392},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.066},	// angular speeds seem to be expressed in deg/s instead of rad/s!!!
						{16, 1, 2047, 0.066},
						{16, 1, 2047, 0.066},
						{16, 1, 0, 0.00122},
						{1, 1, 0, 1.}
		},
		VFRsensors[]= {
						{11, 1, 328, 1. * 360},
						{8, 1, 0, 0.031405},	// 60 tops for a wheel diameter of 60cm
						{8, 1, 0, 0.03298},		// 60 tops for a wheel diameter of 63cm
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.066},
						{16, 1, 2047, 0.066},
						{16, 1, 2047, 0.066},
						{16, 1, 0, 0.00122},
						{1, 1, 0, 1.}
		},
		ER5sensors[]= {
						{11, 1, 328, 1. * 360},
						/*
						 \ the Mod2 code I found suggested 27 tops in front and 42 in the rear, which doesn't correspond
						 \ to the recorded data, nor to the fact that FW brake disks are usually larger than rear disks
						 \ (and thus have more ventilation holes).
						 \ (the ER5 takes wheel revs from the brake disk perforations.)
						 \ 20090818: correction: 29 tops at the front, rear tops are taken from the pinion wheel teeth, so 44 ...
						 */
						{8, 1, 0, 1.77/29.},	// 190 cm for 9 tops (29 tops?! sloppy commenting habits...)
						{8, 1, 0, 1.88/44.},	// 190 cm for 44 tops
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.012208},
						{16, 1, 2047, 0.066},
						{16, 1, 2047, 0.066},
						{16, 1, 2047, 0.066},
						{16, 1, 0, 0.00122},
						{1, 1, 0, 1.}
		},
		RAWsensors[]= {
						{11, 1, 0, 1.},
						{8, 1, 0, 1.},
						{8, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{16, 1, 0, 1.},
						{1, 1, 0, 1.}
		},
		*theSensors= NULL;

// the various sensors, and their output after conversion to ISO units
char *IEFsensorLabels[]= { "Time [s]", "FrontWheel [tops]", "RearWheel [tops]",
	NULL, "Brake [?]", "Steer [deg]",
	"refX [m/s2]", "refY [m/s2]", "refZ [m/s2]",
	"linX [m/s2]", "linY [m/s2]", "linZ [m/s2]",
	"angX [deg/s]", "angY [deg/s]", "angZ [deg/s]",
	"Throttle [V]", "Time2 [s]",
	"fwDist [m]", "rwDist [m]",
	"fwSpeed [km/h]", "rwSpeed [km/h]",
};
char *RAWsensorLabels[]= { "Time [s]", "FrontWheel [tops]", "RearWheel [tops]",
	NULL, "Brake [tops]", "Steer [tops]",
	"refX [tops/s2]", "refY [tops/s2]", "refZ [tops/s2]",
	"linX [tops/s2]", "linY [tops/s2]", "linZ [tops/s2]",
	"angX [tops/s]", "angY [tops/s]", "angZ [tops/s]",
	"Throttle [tops]", "Time2 [s]",
	"fwDist [??]", "rwDist [??]",
	"fwSpeed [??/h]", "rwSpeed [??/h]",
};
char **theLabels= NULL;
static char *RemoteLab = "Remote [tops]", *RemoteLabRaw = "Remote [?]", *TTLLab = "TTL [?]";

IEFmobikeSpecs VFRspecs = {1.460, ((90.-25.5)/360.)*2.* M_PI, 1., 1024*4},
		CBFspecs = {1.483, ((90.-26.)/360.)*2.* M_PI, 1., 1024*4},
		ER5specs = {1.483, ((90.-26.)/360.)*2.* M_PI, 1., 1024*4},
		RAWspecs = {1, 1, 1., 1},
		*theSpecs= NULL;

char *KnownVehicle[]= { "RAWdata", "CBF1000", "VFR800", "ER5" };
int KnownVehicles= sizeof(KnownVehicle)/sizeof(char*);

#if 0
extern ascanf_Update_AnalogSensor_Stats( ASCB_ARGLIST );

// a list of variables that allow to parametrise the import process (and a single function to invoke the
// recalibration feature manually):
static ascanf_Function IEFio_Function[] = {
	{ "$IEF-Import-Vehicle-Selector", NULL, 2, _ascanf_variable,
		"$IEF-Import-Vehicle-Selector: selects the vehicle (motorcycle) on which the data to be imported.\n"
		" were recorded.\n"
		" 0:\tRaw sensor data, untransformed (and the time channel \"uncorrected\")\n"
		" 1:\tHonda CBF1000\n"
		" 2:\tHonda VFR800\n"
		" 3:\tKawasaki ER5\n"
		, 1, 0, 0, 0, 0, 0, 0.0
		// changes to this variable are processed through an internal ChangeHandler, IEFio_callback()
	},
	{ "$IEF-Import-Column-Selector", NULL, 2, _ascanf_variable,
		"$IEF-Import-Column-Selector: point this variable to an array specifying which columns to import.\n"
		" The array may be floating point or integer, but should enumerate the column numbers (i.e. do not\n"
		" use flags per column). Invalid values are silently ignored.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$IEF-Import-Feedback", NULL, 2, _ascanf_variable,
		"$IEF-Import-Feedback: shows and stores information about the import.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$IEF-Import-SavGolSpeed-Coeffs", NULL, 2, _ascanf_variable,
		"$IEF-Import-SavGolSpeed-Coeffs: if pointing to a 2-element array, determines filter half-width and order\n"
		" of the Savitzky-Golay filter used to calculate wheel speed from wheel distance\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$IEF-Standstill-Threshold", NULL, 2, _ascanf_variable,
		"$IEF-Standstill-Threshold: minimum duration (in s) under which periods in which both wheels do NOT"
		" turn are ignored. Stop-Start periods are thus at least this long (and are marked with vertical ULabels).\n"
		, 1, 0, 0, 0, 0, 0, 2.0
	},
	{ "$IEF-Recalibrate-AnalogSensors", NULL, 2, _ascanf_variable,
		"$IEF-Recalibrate-AnalogSensors: whether or not to calculate the average output of the analog sensors at"
		" standstill, and recalibrate the sensors' subsequent zero reference to that value.\n"
		" Must point to an array, where the first element is:\n"
		" 1: re-calibrate, readings during standstill are calibrated w.r.t. the previous stop\n"
		" 2: idem, readings during standstill are raw\n"
		" 3: idem, readings during standstill transit linearly from the previous to the new calibration\n"
		" 4: idem, readings during standstill are eliminated by setting them to the value of the second element\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$IEF-Import-Range", NULL, 2, _ascanf_variable,
		"$IEF-Import-Range: time interval over which to import data (of interest).\n"
		" Set to -Inf,Inf to import everything.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$IEF-Simple-Speed-Calculation", NULL, 2, _ascanf_variable,
		"$IEF-Simple-Speed-Calculation: if True, determines linear speed directly from the sensor data;\n"
		" if False, speed is calculated from wheel distance using $IEF-Import-SavGolSpeed-Coeffs.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$IEF-Correct-Incoherent-Time", NULL, 2, _ascanf_variable,
		"$IEF-Correct-Incoherent-Time: attempts to correct temporal incoherence (which assumes a fixed\n"
		" sampling frequency of 1kHz)\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$IEF-Export-TTL-Mask", NULL, 2, _ascanf_variable,
		"$IEF-Export-TTL-Mask: if unset, exports a single bit in column 4, the \"remote\" trigger as used\n"
		" in certain experiments. If set, export the mask of 5 TTL signals that is coded together with the\n"
		" handlebar (steer) position in a 16bit word.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "Recalibrate-AnalogSensors", ascanf_Update_AnalogSensor_Stats, 7, NOT_EOF,
		"Recalibrate-AnalogSensors[setnr,timecol,FWcol,RWcol,threshold,calibtype,&data]: recalibrate the data in the array\n"
		" <data> to the data in <set>, detecting periods of standstill longer than <threshold> seconds from sequences of\n"
		" constant travelled distance of the front AND rear wheels. Time data is expected in the <timecol> column; front\n"
		" and rear wheel revolutions or travelled distance in <fwcol> and <rwcol> respectively. <calibtype> can be set as\n"
		" explained for $IEF-Recalibrate-AnalogSensors.\n"
	},
};
static int IEFio_Functions= sizeof(IEFio_Function)/sizeof(ascanf_Function);

// the internal representation of the parameter list defined just above:
static ascanf_Function *afVehicleSelector= &IEFio_Function[0], *afStandstillRecalibrateAnalogSensors= NULL,
	*afImportRange= NULL;
static double *VehicleSelector= &IEFio_Function[0].value;
static double *ColumnSelector= &IEFio_Function[1].value;
static double *ImportFeedback= &IEFio_Function[2].value;
static double *SGparams= &IEFio_Function[3].value;
static double *StandstillDurationThreshold= &IEFio_Function[4].value;
static double *StandstillRecalibrateAnalogSensors= &IEFio_Function[5].value;
// static double *ImportRange= &IEFio_Function[6].value;
static double *ImportRange= NULL;
static double *SimpleSpeed= &IEFio_Function[7].value;
static double *CorrectIncoherentTime= &IEFio_Function[8].value;
static double *ExportTTLMask= &IEFio_Function[9].value;

#endif

// calculate wheel circumference from the tyre specifications:
void init_WheelConversions()
{
	// VFR: fw= 120/70 x 17" => diameter 17"*2.54 + 2 * 120mm * 70% = 59.98cm
	// there are 60 tops per revolution, so the distance travelled (in m) per top is:
	VFRsensors[c_fw].reading2ISO= (17 * 0.0254 + 2 * 0.12 * 0.7) * M_PI / 60;
	// VFR rw= 180/55 x 17"
	VFRsensors[c_rw].reading2ISO= (17 * 0.0254 + 2 * 0.18 * 0.55) * M_PI / 60;
	// CBF: fw= 120/70 x 17" => diameter 17"*2.54 + 2 * 120mm * 70% = 59.98cm
	// there are 60 tops per revolution, so the distance travelled (in m) per top is:
	CBFsensors[c_fw].reading2ISO= (17 * 0.0254 + 2 * 0.12 * 0.7) * M_PI / 60;
	// CBF rw= 160/60 x 17"
	CBFsensors[c_rw].reading2ISO= (17 * 0.0254 + 2 * 0.16 * 0.6) * M_PI / 60;
	// ER5: fw= 110/70 x 17" => diameter 17"*2.54 + 2 * 110mm * 70% 
	// there are 29 tops per revolution, so the distance travelled (in m) per top is:
	ER5sensors[c_fw].reading2ISO= (17 * 0.0254 + 2 * 0.11 * 0.7) * M_PI / 29;
	// ER5 rw= 130/70 x 17" for 44 tops
	ER5sensors[c_rw].reading2ISO= (17 * 0.0254 + 2 * 0.13 * 0.7) * M_PI / 44;
}

int SelectVehicle()
{ int r= 0;
	switch( ((int)(*VehicleSelector)) ){
		case 0:
			theSensors= RAWsensors;
			theSpecs= &RAWspecs;
			theLabels= RAWsensorLabels;
			break;
		case 1:
			theSensors= CBFsensors;
			theSpecs= &CBFspecs;
			theLabels= IEFsensorLabels;
			break;
		case 2:
			theSensors= VFRsensors;
			theSpecs= &VFRspecs;
			theLabels= IEFsensorLabels;
			break;
		case 3:
			theSensors= ER5sensors;
			theSpecs= &ER5specs;
			theLabels= IEFsensorLabels;
			break;
		default:
			r= 1;
			break;
	}
	if( *VehicleSelector>= KnownVehicles ){
		fprintf( StdErr, "##\n## Internal inconsistency concerning the number of known vehicles -- %d>%d\n##\n",
			(int) *VehicleSelector, KnownVehicles
		);
		*VehicleSelector= KnownVehicles-1;
	}
	init_WheelConversions();
	return(r);
}

// structure and function to convert 'tops' to speed and travelled distance
typedef struct TopBuf{
	uint8 prevTop;
	uint64 tops;
	double speed, distance, prevTime;
} TopBuf;

void top2Dist( uint8 rawtop, double time, TopBuf *mem, IEFsensorSpecs *sensor )
{ int top= (int) rawtop - (int) mem->prevTop;
	if( top< 0 ){
		top = 256 + top;
	}
	mem->tops+= top;
	mem->speed= ((top - sensor->offset) * sensor->reading2ISO) / (time - mem->prevTime);
	mem->distance= (mem->tops - sensor->offset) * sensor->reading2ISO;
	mem->prevTime= time;
	mem->prevTop= rawtop;
}

// periods of stationarity receive a graphic label at their start and end:
void Add_Stationarity_ULabel( int nr, double startTime, double endTime, int set_nr )
{ char *templ=
	"IDict[ SetULabel[-1,\"stop:%d\",{%g,0,%g,0},%d,\"VL\",\"Red\",1],"
	"SetULabel[-1,\"start:%d\",{%g,0,%g,0},%d,\"VL\",\"Green\",1] ] @";
  double dum;
  __ALLOCA( abuf, char, strlen(templ)+4*256+4*64, ablen );

	snprintf( abuf, ablen/sizeof(char), templ,
		nr, startTime, startTime, set_nr,
		nr, endTime, endTime, set_nr
	);
	new_param_now( abuf, &dum, 1 );
	GCA();
}

// samples with incoherent time receive a graphic label:
void Add_IncoherentTime_ULabel( double time, double time2, double corrTime, unsigned long idx, int set_nr )
{ char *templ=
	"IDict[ SetULabel[-1,\"Incoherent sample #%lu time(s) %g,%g (d=%g), should be approx. %g\",{%g,0,%g,0},%d,\"VL\",\"Purple\",1] ] @";
  double dum;
  __ALLOCA( abuf, char, strlen(templ)+6*256+64, ablen );
	
	snprintf( abuf, ablen/sizeof(char), templ,
		idx, time, time2, time2-time, corrTime,
		corrTime, corrTime, set_nr
	);
	new_param_now( abuf, &dum, 1 );
	GCA();
}

// statistics bins used for the automatic recalibration feature:
SimpleStats SS_refX, SS_refY, SS_refZ, SS_linX, SS_linY, SS_linZ, SS_angX, SS_angY, SS_angZ, SS_UAS;

void SS_Add_IEFData( SimpleStats *dst, double val)
{
	if( Check_Ignore_NaN(1,(val)) ){
		SS_Add_Data( dst, 1, val, 1);
	}
}

void Update_AnalogSensor_Stats( double **DataMatrix, short *tColumn, unsigned long statStart, unsigned long statEnd,
	ascanf_Function *af, double update_type,
	double currentTime, double statStartTime, double statStopTime, char *fbbuf, int fblen
)
{ // need to correct for current re-calibration values:
  double refXm= SS_refX.mean, refYm= SS_refY.mean, refZm= SS_refZ.mean,
		linXm= SS_linX.mean, linYm= SS_linY.mean, linZm= SS_linZ.mean,
		angXm= SS_angX.mean, angYm= SS_angY.mean, angZm= SS_angZ.mean,
		UASm= SS_UAS.mean;

	SS_Init_(SS_refX); SS_Init_(SS_refY); SS_Init_(SS_refZ);
	SS_Init_(SS_linX); SS_Init_(SS_linY); SS_Init_(SS_linZ);
	SS_Init_(SS_angX); SS_Init_(SS_angY); SS_Init_(SS_angZ);
	SS_Init_(SS_UAS);

	if( ASCANF_TRUE(update_type) ){
	  int i;
		for( i= statStart; i< statEnd; i++ ){
			if( af ){
				SS_Add_IEFData( &SS_UAS, ASCANF_ARRAY_ELEM(af,i)+ UASm );
				if( update_type== 2 ){
 					ASCANF_ARRAY_ELEM_OP(af,i, +=, UASm );
				}
			}
			else{
				if( tColumn[6]>= 0 ){
					// data currently stored in the set has already been calibrated, to the
					// previous "average zero" (which can be 0). So that value (refXm in this case)
					// has to be added in order to determine the true average zero.
					SS_Add_IEFData( &SS_refX, DataMatrix[tColumn[6]][i]+ refXm );
					if( update_type== 2 ){
						DataMatrix[tColumn[6]][i]+= refXm;
					}
				}
				if( tColumn[7]>= 0 ){
					SS_Add_IEFData( &SS_refY, DataMatrix[tColumn[7]][i]+ refYm );
					if( update_type== 2 ){
						DataMatrix[tColumn[7]][i]+= refYm;
					}
				}
				if( tColumn[8]>= 0 ){
					SS_Add_IEFData( &SS_refZ, DataMatrix[tColumn[8]][i]+ refZm );
					if( update_type== 2 ){
						DataMatrix[tColumn[8]][i]+= refZm;
					}
				}
				if( tColumn[9]>= 0 ){
					SS_Add_IEFData( &SS_linX, DataMatrix[tColumn[9]][i]+ linXm );
					if( update_type== 2 ){
						DataMatrix[tColumn[9]][i]+= linXm;
					}
				}
				if( tColumn[10]>= 0 ){
					SS_Add_IEFData( &SS_linY, DataMatrix[tColumn[10]][i]+ linYm );
					if( update_type== 2 ){
						DataMatrix[tColumn[10]][i]+= linYm;
					}
				}
				if( tColumn[11]>= 0 ){
					SS_Add_IEFData( &SS_linZ, DataMatrix[tColumn[11]][i]+ linZm );
					if( update_type== 2 ){
						DataMatrix[tColumn[11]][i]+= linZm;
					}
				}
				if( tColumn[12]>= 0 ){
					SS_Add_IEFData( &SS_angX, DataMatrix[tColumn[12]][i]+ angXm );
					if( update_type== 2 ){
						DataMatrix[tColumn[12]][i]+= angXm;
					}
				}
				if( tColumn[13]>= 0 ){
					SS_Add_IEFData( &SS_angY, DataMatrix[tColumn[13]][i]+ angYm );
					if( update_type== 2 ){
						DataMatrix[tColumn[13]][i]+= angYm;
					}
				}
				if( tColumn[14]>= 0 ){
					SS_Add_IEFData( &SS_angZ, DataMatrix[tColumn[14]][i]+ angZm );
					if( update_type== 2 ){
						DataMatrix[tColumn[14]][i]+= angZm;
					}
				}
			}
		}
		if( af ){
			SS_Mean_( SS_UAS );
			snprintf( fbbuf, fblen, "## %s: standstill from %g-%g (%gs); recalibrating analog sensor zeros from t=%gs with %g\n",
				af->name,
				statStartTime, statStopTime, statStopTime-statStartTime,
				currentTime, SS_UAS.mean
			);
		}
		else{
			snprintf( fbbuf, fblen,
				" IEFio::import_IEF(): standstill from %g-%g (%gs); recalibrating analog sensor zeros from t=%gs with:",
				statStartTime, statStopTime, statStopTime-statStartTime,
				currentTime
			);
			if( tColumn[6]>= 0 ){
				SS_Mean_( SS_refX );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_refX.mean );
			}
			if( tColumn[7]>= 0 ){
				SS_Mean_( SS_refY );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_refY.mean );
			}
			if( tColumn[8]>= 0 ){
				SS_Mean_( SS_refZ );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_refZ.mean );
			}
			if( tColumn[9]>= 0 ){
				SS_Mean_( SS_linX );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_linX.mean );
			}
			if( tColumn[10]>= 0 ){
				SS_Mean_( SS_linY );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_linY.mean );
			}
			if( tColumn[11]>= 0 ){
				SS_Mean_( SS_linZ );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_linZ.mean );
			}
			if( tColumn[12]>= 0 ){
				SS_Mean_( SS_angX );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_angX.mean );
			}
			if( tColumn[13]>= 0 ){
				SS_Mean_( SS_angY );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_angY.mean );
			}
			if( tColumn[14]>= 0 ){
				SS_Mean_( SS_angZ );
				snprintf( fbbuf, fblen, "%s %g", fbbuf, SS_angZ.mean );
			}
			strcat( fbbuf, "\n" );
		}
		StringCheck( fbbuf, fblen, __FILE__, __LINE__ );
		if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
			fputs( fbbuf, StdErr );
		}
		if( update_type== 3 || update_type== 4 ){
		  double nan;
			set_NaN(nan);
			for( i= statStart; i< statEnd; i++ ){
			  // rather horrible kludge for setting values to NaN: set weight to NaN and let it propagate through the calculation:
			  double weight= (update_type==3)? ((double)i-statStart) / ((double)statEnd-1-statStart) : nan;
				if( af ){
					if( update_type== 4 && afStandstillRecalibrateAnalogSensors->type== _ascanf_array ){
						ASCANF_ARRAY_ELEM_SET(af,i, StandstillRecalibrateAnalogSensors[1] );
					}
					else{
						ASCANF_ARRAY_ELEM_OP(af,i, +=, (UASm - SS_UAS.mean)*weight );
					}
				}
				else{
					if( update_type== 4 && afStandstillRecalibrateAnalogSensors->type== _ascanf_array ){
						if( tColumn[6]>= 0 ){
							DataMatrix[tColumn[6]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[7]>= 0 ){
							DataMatrix[tColumn[7]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[8]>= 0 ){
							DataMatrix[tColumn[8]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[9]>= 0 ){
							DataMatrix[tColumn[9]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[10]>= 0 ){
							DataMatrix[tColumn[10]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[11]>= 0 ){
							DataMatrix[tColumn[11]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[12]>= 0 ){
							DataMatrix[tColumn[12]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[13]>= 0 ){
							DataMatrix[tColumn[13]][i]= StandstillRecalibrateAnalogSensors[1];
						}
						if( tColumn[14]>= 0 ){
							DataMatrix[tColumn[14]][i]= StandstillRecalibrateAnalogSensors[1];
						}
					}
					else{
						if( tColumn[6]>= 0 ){
							DataMatrix[tColumn[6]][i]+= (refXm - SS_refX.mean) * weight;
						}
						if( tColumn[7]>= 0 ){
							DataMatrix[tColumn[7]][i]+= (refYm - SS_refY.mean) * weight;
						}
						if( tColumn[8]>= 0 ){
							DataMatrix[tColumn[8]][i]+= (refZm - SS_refZ.mean) * weight;
						}
						if( tColumn[9]>= 0 ){
							DataMatrix[tColumn[9]][i]+= (linXm - SS_linX.mean) * weight;
						}
						if( tColumn[10]>= 0 ){
							DataMatrix[tColumn[10]][i]+= (linYm - SS_linY.mean) * weight;
						}
						if( tColumn[11]>= 0 ){
							DataMatrix[tColumn[11]][i]+= (linZm - SS_linZ.mean) * weight;
						}
						if( tColumn[12]>= 0 ){
							DataMatrix[tColumn[12]][i]+= (angXm - SS_angX.mean) * weight;
						}
						if( tColumn[13]>= 0 ){
							DataMatrix[tColumn[13]][i]+= (angYm - SS_angY.mean) * weight;
						}
						if( tColumn[14]>= 0 ){
							DataMatrix[tColumn[14]][i]+= (angZm - SS_angZ.mean) * weight;
						}
					}
				}
			}
		}
	}
}

// the central function that does most of the work. It is called by XGraph when the *DM_IO* function is encountered
// receiving a pointer to the open file or stream, its name, number; the current DataSet number and pointer, and
// a pointer to a structure containing the state of XGraph's central ReadData() function (which have to be updated by us).
int import_IEF(FILE *stream, char *the_file, int filenr, int setNum, struct DataSet *this_set, ReadData_States *state )
{ IEFsamples DataBuf, DataRaw;
  IEFcommands DataComm;
  int ret= 0;
#ifdef DEBUG
  int break_poffset= 115874;
#endif

	if( stream ){
	  size_t nread= 0, MaxNChannels= IEF_CHANNELS + 2+2, NChannels= 0;
	  LabelsList *llist= this_set->ColumnLabels;
	  unsigned long poffset= this_set->numPoints;
	  char fbbuf[1024];
	  int fblen= sizeof(fbbuf)-1;
	  ascanf_Function *af= NULL;
	  __ALLOCA( ImportColumn, short, IEF_CHANNELS, IClen );
	  __ALLOCA( TargetColumn, short, MaxNChannels, TClen );
	  TopBuf frontwheel, rearwheel;
	  double currentTime, correctedTime, dt= 0, statStartTime, statStopTime, importStartTime, importEndTime;
	  uint32 dtN= 0;
	  int32 statStart= -1, statNr= 0;
	  SimpleStats SSdt, SSdtraw;
	  // 20090807: these used to be static vars inside the block handling incoherent time samples.
	  unsigned long pIncohTime= -1, nIncoh=0;
#ifdef USING_BITFIELDS
	  uint8 remote;
#endif

		memset( &DataBuf, 0, sizeof(DataBuf) );
		memset( &frontwheel, 0, sizeof(TopBuf) );
		memset( &rearwheel, 0, sizeof(TopBuf) );
		set_NaN(currentTime);
		statStopTime= currentTime;
		set_Inf(importStartTime,1);
		set_Inf(importEndTime,-1);

		SS_Init_(SSdt);
		if( *ImportFeedback > 1 ){
			SSdt.exact = 1;
		}
		SS_Init_(SSdtraw);

		if( *ExportTTLMask ){
			IEFsensorLabels[3] = TTLLab;
			RAWsensorLabels[3] = TTLLab;
		}
		else{
			IEFsensorLabels[3] = RemoteLab;
			RAWsensorLabels[3] = RemoteLabRaw;
		}

		{ int i, c;
			 // initialise the array that indicate which columns to import:
			if( *ColumnSelector && (af=
					parse_ascanf_address( *ColumnSelector, _ascanf_array, "IEFio::import_IEF()",
						ascanf_verbose, NULL))
			){
				memset( ImportColumn, 0, IClen );
				for( i= 0; i< af->N; i++ ){
					if( (c= ASCANF_ARRAY_ELEM(af,i))>= 0 && c< IEF_CHANNELS ){
						ImportColumn[c]= 1;
					}
				}
			}
			else{
				for( i= 0; i< IEF_CHANNELS; i++ ){
					ImportColumn[i]= 1;
				}
			}
			  // initialise the array that indicates which input column is stored in what target column,
			  // and count the effective number of columns that will be imported.
			memset( TargetColumn, -1, TClen );
			for( i= 0; i< IEF_CHANNELS; i++ ){
				if( ImportColumn[i] ){
					TargetColumn[i]= NChannels;
					NChannels+= 1;
				}
			}
			for( ; i< MaxNChannels; i++ ){
				TargetColumn[i]= NChannels;
				NChannels+= 1;
			}
		}

		if( *ImportFeedback ){
		  ALLOCA( buf, char, strlen(the_file)+256, blen );
			time_stamp( stream, the_file, buf, True, "\n" );
			snprintf( fbbuf, fblen,
				" IEFio::import_IEF(): reading on %d channels from"
				" %s",
				IEF_CHANNELS, buf
			);
			if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
				fputs( fbbuf, StdErr );
				if( af ){
					fprintf( StdErr, "\tColumnSelector: '%s'[%d]\n", af->name, af->N );
				}
			}
			GCA();
		}

		{ int N, i;
			if( this_set->ncols< NChannels ){
				if( this_set->numPoints ){
					this_set->columns= realloc_columns( this_set, NChannels );
				}
				else{
					this_set->ncols= NChannels;
				}
			}
			
			if( *ImportFeedback ){
				snprintf( fbbuf, fblen, " IEFio::import_IEF(%s): importing data recorded on a \"%s\":\n",
					the_file, KnownVehicle[(int)*VehicleSelector]
				);
				StringCheck( fbbuf, fblen, __FILE__, __LINE__ );
				if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
					fputs( fbbuf, StdErr );
				}
			}
			N= LabelsList_N( llist );
			for( i= 0; i< MaxNChannels; i++ ){
				if( TargetColumn[i]>= 0 && TargetColumn[i]< this_set->ncols ){
					llist= Add_LabelsList( llist, &N, TargetColumn[i], theLabels[i] );
					if( *ImportFeedback ){
						snprintf( fbbuf, fblen, " \"%s\"{%d}", theLabels[i], TargetColumn[i] );
						StringCheck( fbbuf, fblen, __FILE__, __LINE__ );
						if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
							fputs( fbbuf, StdErr );
						}
					}
				}
			}
			if( *ImportFeedback ){
				if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
					fputs( "\n", StdErr );
				}
			}
		}
		this_set->ColumnLabels= llist;

		SS_Init_(SS_refX); SS_Init_(SS_refY); SS_Init_(SS_refZ);
		SS_Init_(SS_linX); SS_Init_(SS_linY); SS_Init_(SS_linZ);
		SS_Init_(SS_angX); SS_Init_(SS_angY); SS_Init_(SS_angZ);

		new_param_now( NULL, NULL, 0 );

		  /* Read the sample data from file in a loop. We don't check for EOF or file error in the loop,
		   \ but perform it for all channels; thus, the user cannot avoid knowing that *no* data was read
		   \ if things went awry in the first channel.
		   */
		while( (nread= fread( &DataRaw, sizeof(DataRaw), 1, stream)== 1) ){

			  // make a copy of the data frame we just read, in order to keep a copy of the raw data around:
			DataBuf= DataRaw;

			if( SwapEndian || EndianType!= 1 ){
				/* We honour requests to correct the endianness, but we do know that this format is PC-native,
				 \ (little endian) so we also do the correction if it turns out to be necessary! Unasked.
				 */
				SwapEndian_int32( (int32_t*) &DataBuf.time, 1 );
				//SwapEndian_int16( &DataBuf.topsRW, 1 );
				//SwapEndian_int16( &DataBuf.topsFW, 1 );
				SwapEndian_int16( (int16_t*) &DataBuf.comm, 1 );
				SwapEndian_int16( (int16_t*) &DataBuf.ref, 3 );
				SwapEndian_int16( (int16_t*) &DataBuf.lin, 3 );
				SwapEndian_int16( (int16_t*) &DataBuf.ang, 3 );
				SwapEndian_int16( (int16_t*) &DataBuf.throttle, 1 );
				SwapEndian_int32( (int32_t*) &DataBuf.time2, 1 );
			}

			  // determine/correct the current time stamp:
			{ double newTime = DataBuf.time * 4e-6;
				if( !NaN(currentTime) ){
					dt= newTime - currentTime;
					dtN+= 1;
				}

				if( dtN ){
				  unsigned int timecheck;
					if( DataBuf.time2 < DataBuf.time ){
						timecheck = ((unsigned int) -1) - DataBuf.time - 1 + DataBuf.time2;
					}
					else{
						timecheck = DataBuf.time2 - DataBuf.time;
					}
					if( (timecheck<= 3 || timecheck>= 300) ){
						SS_Add_Data_( SSdtraw, 1, dt, 1.0 );
					}
					if( theSensors!= RAWsensors && *CorrectIncoherentTime && (timecheck<= 3 || timecheck>= 300) ){
					  double time2= DataBuf.time2 * 4e-6;
						correctedTime= (frontwheel.prevTime + rearwheel.prevTime)/2 + 250*4e-6;
						// 20090807: test...
						// correctedTime= newTime;
						snprintf( fbbuf, fblen, "Incoherent sample #%lu:%lu time(s) %g,%g (d=%g dt=%g), should be approx. %g\n",
							nIncoh, poffset, newTime, time2, time2-newTime, dt, correctedTime
						);
						StringCheck( fbbuf, fblen, __FILE__, __LINE__ );
						if( pIncohTime< 0 || poffset< pIncohTime || (poffset-pIncohTime)> 99 ){
							Add_IncoherentTime_ULabel( newTime, time2,
								correctedTime, poffset, this_set->set_nr
							);
							pIncohTime = poffset;
						}
						if( *ImportFeedback > 1 ){
//							if( pragma_unlikely(ascanf_verbose) || scriptVerbose )
							{
								fputs( fbbuf, StdErr );
							}
						}
						  // 20090618 ... correct dtSum too. Is actually necessary as dtSum is used in calculating speed.
//						dtSum -= DataBuf.time * 4e-6 - currentTime;
//						dtSum += correctedTime - currentTime;
						SS_Add_Data_( SSdt, 1, correctedTime - currentTime, 1.0 );
//						// 20090807: in the Mod2 code, the dt interval is in fact set to 0 (?!), so correctedTime=currentTime!
//						SS_Add_Data_( SSdt, 1, 0, 1.0 );
						currentTime = correctedTime;
						nIncoh += 1;
					}
					else{
						SS_Add_Data_( SSdt, 1, dt, 1 );
						correctedTime = currentTime = newTime;
					}
				}
				else{
					correctedTime = currentTime = newTime;
				}
			}

			 // allocate or expand the storage space:
			if( pragma_unlikely(!this_set->numPoints)
			    || *(state->Spot)>= this_set->numPoints || *(state->spot)>= this_set->numPoints
			){
				if( pragma_likely(this_set->numPoints) ){
					this_set->numPoints*= 2;
				}
				else{
					this_set->numPoints= 64;
				}
				realloc_points( this_set, this_set->numPoints, False );
			}

			  // process the data concerning the handlebars, brakes and the switch gear (binary sensors)
			{ uint16 frein, binaire, guidon;
#ifdef USING_BITFIELDS
				IEFbinaries ttl;
				
				ttl.dword = DataBuf.comm;
				DataComm = ttl.comm;
#endif
#ifdef DEBUG
				if( poffset== break_poffset ){
					break_poffset+= 1;
				}
#endif
				guidon= *((uint16*)&DataBuf.comm);
//				frein= (guidon >> 12) & 0x03;
				frein= (guidon >> 13) & 0x01;
// SE's code which I don't understand
//				{ uint16 gray= guidon & 0x07ff;
//					if( gray> 1023 ){
//						binaire= (uint16) ((int16)gray - 2048);
//					}
//					else{
//						binaire= gray;
//					}
//				}
// my code, guessed from the format description (20090626: SE does the same thing now):
				binaire= (guidon >> 15) & 0x01;
#ifndef USING_BITFIELDS
				DataComm.brake= frein;
				DataComm.remote= binaire;
				// guidon (handlebars) actually have 10bits + a sign bit??
				DataComm.steer= guidon & 0x07ff;
				// 20080724: interpretation of this field remains evasive - this is to make sure we cater for
				// 11bits signed integers?
				if( DataComm.steer> 1023 ){
					DataComm.steer -= 2048;
				}
				DataComm.TTLMask = (guidon >> 11) & 0x001f;
#else
				remote= (guidon >> 15) & 0x01;
#endif
			}

			  // we keep track of periods the wheels aren't turning, which can serve to recalibrate certain readings:
			if( dtN ){
				if( statStart>=0 ){
					if( DataBuf.topsFW != frontwheel.prevTop || DataBuf.topsRW != rearwheel.prevTop ){
					  double duration= (statStopTime= (frontwheel.prevTime + rearwheel.prevTime)/2) - statStartTime;
						if( duration>= *StandstillDurationThreshold ){
							if( !ImportRange /*afImportRange->type!= _ascanf_array*/
								|| (currentTime>= ImportRange[0] && currentTime<= ImportRange[1])
							){
								Add_Stationarity_ULabel( statNr, statStartTime, currentTime, this_set->set_nr );
							}
							statNr+= 1;
							Update_AnalogSensor_Stats( this_set->columns, TargetColumn, statStart, poffset,
								  NULL, *StandstillRecalibrateAnalogSensors, currentTime, statStartTime, statStopTime,
								  fbbuf, fblen
							);
						}
						statStart= -1;
					}
				}
				else{
					if( DataBuf.topsFW == frontwheel.prevTop && DataBuf.topsRW == rearwheel.prevTop ){
						statStart= poffset;
//						statStartTime= (frontwheel.prevTime + rearwheel.prevTime)/2;
//						if( statStartTime== statStopTime ){
//							statStartTime+= dtSum/dtN;
//						}
						statStartTime= currentTime;
					}
				}
			}
			
			  // convert wheel tops to rotations and thus covered distance:
			top2Dist( DataBuf.topsFW, currentTime, &frontwheel, &theSensors[c_fw] );
			top2Dist( DataBuf.topsRW, currentTime, &rearwheel, &theSensors[c_rw] );
			
			  // store the selected columns of the current sample, provided it falls within the selected
			  // temporal input range. Readings are converted from the raw value to the ISO value they represent.
			if( !ImportRange /*afImportRange->type!= _ascanf_array*/
				|| (currentTime>= ImportRange[0] && currentTime<= ImportRange[1])
			){
				if( currentTime< importStartTime ){
					importStartTime= currentTime;
				}
				if( currentTime> importEndTime ){
					importEndTime= currentTime;
				}

				if( TargetColumn[0]>= 0 ){
				  // time is in 250 tops of 4microsec each, so 250000 per second ; 1/250000 = 4e-6
					this_set->columns[0][poffset]= currentTime;
				}
				if( TargetColumn[1]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_fw];
					this_set->columns[1][poffset]= (DataBuf.topsFW - theSensor->offset) * theSensor->reading2ISO;
				}
				if( TargetColumn[2]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_rw];
					this_set->columns[2][poffset]= (DataBuf.topsRW - theSensor->offset) * theSensor->reading2ISO;
				}
#ifndef USING_BITFIELDS
				if( TargetColumn[3]>= 0 ){
					if( *ExportTTLMask ){
						this_set->columns[TargetColumn[3]][poffset]= DataComm.TTLMask;
					}
					else{
						this_set->columns[TargetColumn[3]][poffset]= DataComm.remote;
					}
				}
#else
				if( TargetColumn[3]>= 0 ){
					if( *ExportTTLMask ){
						this_set->columns[TargetColumn[3]][poffset]= ((IEFcommands2*)&DataComm)->TTLMask;
					}
					else{
						this_set->columns[TargetColumn[3]][poffset]= remote;
					}
				}
#endif
				if( TargetColumn[4]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_brake];
					this_set->columns[TargetColumn[4]][poffset]= (DataComm.brake - theSensor->offset) * theSensor->reading2ISO;
				}
				if( TargetColumn[5]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_steer];
					this_set->columns[TargetColumn[5]][poffset]= (DataComm.steer - theSensor->offset)
						* theSensor->reading2ISO
						/ theSpecs->steerSensorResolution * theSpecs->steerSensorGearing;
				}
				if( TargetColumn[6]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_refx];
					this_set->columns[TargetColumn[6]][poffset]= (DataBuf.ref.X - theSensor->offset) * theSensor->reading2ISO - SS_refX.mean;
				}
				if( TargetColumn[7]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_refy];
					this_set->columns[TargetColumn[7]][poffset]= (DataBuf.ref.Y - theSensor->offset) * theSensor->reading2ISO- SS_refY.mean;
				}
				if( TargetColumn[8]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_refz];
					this_set->columns[TargetColumn[8]][poffset]= (DataBuf.ref.Z - theSensor->offset) * theSensor->reading2ISO - SS_refZ.mean;
				}
				if( TargetColumn[9]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_ax];
					this_set->columns[TargetColumn[9]][poffset]= (DataBuf.lin.X - theSensor->offset) * theSensor->reading2ISO - SS_linX.mean;
				}
				if( TargetColumn[10]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_ay];
					this_set->columns[TargetColumn[10]][poffset]= (DataBuf.lin.Y - theSensor->offset) * theSensor->reading2ISO - SS_linY.mean;
				}
				if( TargetColumn[11]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_az];
					this_set->columns[TargetColumn[11]][poffset]= (DataBuf.lin.Z - theSensor->offset) * theSensor->reading2ISO - SS_linZ.mean;
				}
				if( TargetColumn[12]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_rx];
					this_set->columns[TargetColumn[12]][poffset]= (DataBuf.ang.X - theSensor->offset) * theSensor->reading2ISO - SS_angX.mean;
				}
				if( TargetColumn[13]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_ry];
					this_set->columns[TargetColumn[13]][poffset]= (DataBuf.ang.Y - theSensor->offset) * theSensor->reading2ISO - SS_angY.mean;
				}
				if( TargetColumn[14]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_rz];
					this_set->columns[TargetColumn[14]][poffset]= (DataBuf.ang.Z - theSensor->offset) * theSensor->reading2ISO - SS_angZ.mean;
				}
				if( TargetColumn[15]>= 0 ){
				  IEFsensorSpecs *theSensor= &theSensors[c_throttle];
					this_set->columns[TargetColumn[15]][poffset]= (DataBuf.throttle - theSensor->offset) * theSensor->reading2ISO;
				}
				if( TargetColumn[16]>= 0 ){
				  // time is in 250 tops of 4microsec each, so 250000 per second ; 1/250000 = 4e-6
					this_set->columns[TargetColumn[16]][poffset]= DataBuf.time2 * 4e-6;
				}
				if( TargetColumn[17]>= 0 ){
					this_set->columns[TargetColumn[17]][poffset]= frontwheel.distance;
				}
				if( TargetColumn[18]>= 0 ){
					this_set->columns[TargetColumn[18]][poffset]= rearwheel.distance;
				}
				if( *SimpleSpeed ){
					if( TargetColumn[19]>= 0 ){
						this_set->columns[TargetColumn[19]][poffset]= frontwheel.speed * 3.6;
					}
					if( TargetColumn[20]>= 0 ){
						this_set->columns[TargetColumn[20]][poffset]= rearwheel.speed * 3.6;
					}
				}

#if ADVANCED_STATS == 1
				this_set->N[poffset]= 1;
#endif
				
				*(state->spot)+= 1;
				*(state->Spot)+= 1;
				poffset+= 1;
			}

		}

		GCA();
		if( nread!= 1 && !feof(stream) ){
			fprintf( StdErr, "IEFio::import_IEF(%s): read problem reading data (%s)\n",
				    the_file, serror()
			);
		}

		 // reallocate the set's storage space so that it contains only the samples actually imported.
		this_set->numPoints= *(state->spot);
		realloc_points( this_set, this_set->numPoints, False );

		  // calculate speed by taking the derivative of covered distance using a Savitzky Golay filter with
		  // half-width and order parameters that can be specified by the user. Calls into xgraph's own ascanf
		  // scripting functionality.
		{ ascanf_Function *af;
			if( *SGparams
				&& (af= parse_ascanf_address( *SGparams, _ascanf_array, "IEFio::import_IEF()", ascanf_verbose, NULL))
				&& af->N== 2
			){ int aae= ascanf_arg_error, lb1, lb2, deriv= 1;
			   double dum, timebase= SS_Mean_(SSdt), tb;
			   char *buf1= NULL, *Command1Template=
					"IDict[ DCL[importIEFSGCoeffs,1,0], DCL[importIEFdata,1,0], "
							"SavGolayCoeffs[&importIEFSGCoeffs,%s,%s,%d],"
							"DCL[importIEFTimeBase,%g],"
							"$AllowSimpleArrayOps[1]"
#ifdef DEBUGMORE
							", verbose[importIEFSGCoeffs]"
#endif
					"] @";
			   char *buf2, *Command2Template=
					"IDict[ LinkArray2DataColumn[&importIEFdata,%d,%d], DataColumn2Array[&importIEFdata,%d,%d],"
#ifdef DEBUGMORE
					"verbose[importIEFdata],"
#endif
					"system.time[convolve[&importIEFdata,&importIEFSGCoeffs,0,0],"
								"mul[&importIEFdata,mul[3.6,importIEFTimeBase]],]"
#ifdef DEBUGMORE
					", verbose[importIEFdata]"
#endif
					"] @";
				if( (buf1= (char*) calloc( (lb1=strlen(Command1Template)+3*256), sizeof(char)))
				    && (buf2= (char*) calloc( (lb2=strlen(Command2Template)+6*64), sizeof(char)))
				){
					  // the convolution result must be multiplied by deriv * AvTimeStep^-deriv to have the right scale;
					  // with deriv==1 ; add a factor 3.6 to go to km/h (done in the Command2Template command above):
					tb= ((deriv>0)? deriv / pow(timebase,deriv) : 1);
					snprintf( buf1, lb1, Command1Template,
						ad2str(ASCANF_ARRAY_ELEM(af,0), d3str_format, 0 )
						, ad2str(ASCANF_ARRAY_ELEM(af,1), d3str_format, 0 )
						, deriv, tb
					);
					new_param_now( buf1, &dum, 1 );
				}
				else{
					ascanf_arg_error= 1;
				}
				if( !*SimpleSpeed ){
					if( !ascanf_arg_error && TargetColumn[17] >= 0 && TargetColumn[19] >= 0 ){
					  int aae2= ascanf_arg_error, sN=setNumber;
						snprintf( buf2, lb2, Command2Template, 
							setNum, TargetColumn[19]
							, setNum, TargetColumn[17]
						);
						if( setNum == setNumber ){
							setNumber+= 1;
						}
						new_param_now( buf2, &dum, 1 );
						ascanf_arg_error= aae2;
						setNumber= sN;
					}
					if( !ascanf_arg_error && TargetColumn[18] >= 0 && TargetColumn[20] >= 0 ){
					  int aae2= ascanf_arg_error, sN=setNumber;
//						snprintf( buf2, lb2, Command2Template, setNum, TargetColumn[18], tb, setNum, TargetColumn[20] );
						snprintf( buf2, lb2, Command2Template, 
							setNum, TargetColumn[20]
							, setNum, TargetColumn[18]
						);
						if( setNum == setNumber ){
							setNumber+= 1;
						}
						new_param_now( buf2, &dum, 1 );
						ascanf_arg_error= aae2;
						setNumber= sN;
					}
					  // clean up
//					new_param_now( "IDict[ Delete[importIEFSGCoeffs], Delete[importIEFdata] ] @", &dum, 1 );
					new_param_now( "IDict[ Delete[importIEFdata] ] @", &dum, 1 );
				}
				ascanf_arg_error= aae;
				xfree(buf1);
				xfree(buf2);
			}
		}

		if( *ImportFeedback ){
			// determine a useful form of feedback
#if 0
			{ int i;
			  SimpleStats rSS, SS;
				SS_Init_(rSS);
				SS_Init_(SS);
				for( i= 0; i< nread; i++ ){
					SS_Add_Data_( rSS, 1, sample[channel][i], 1.0 );
					SS_Add_Data_( SS, 1,
					    sample[channel][i]* calib + ch->ChannelOffset,
					    1.0
					);
				}
			}
#endif
			SS_St_Dev_(SSdt);
			if( SSdt.stdv< 0 ){
				SSdt.stdv = 0;
			}
			snprintf( fbbuf, fblen, " IEFio::import_IEF(%s): av. time base: %gs±%gs over time range %g-%gs"
				, the_file,
				, SSdt.mean, SSdt.stdv
				, importStartTime, importEndTime
			);
			if( statNr ){
				snprintf( fbbuf, fblen, "%s; %d periods at standstill of at least %gs",
					fbbuf, statNr, *StandstillDurationThreshold
				);
				if( ASCANF_TRUE(*StandstillRecalibrateAnalogSensors) ){
					snprintf( fbbuf, fblen, "%s (analog sensors were recalibrated during these, type %g)",
						fbbuf, *StandstillRecalibrateAnalogSensors
					);
				}
			}
			if( SSdtraw.count ){
				snprintf( fbbuf, fblen, "%s (uncorrected time base: %gs±%gs)",
					fbbuf, SS_Mean_(SSdtraw), SS_St_Dev_(SSdtraw)
				);
			}
			strcat( fbbuf, "\n" );
			StringCheck( fbbuf, fblen, __FILE__, __LINE__ );
			if( pragma_unlikely(ascanf_verbose) || scriptVerbose ){
				fputs( fbbuf, StdErr );
			}
		}
		
		if( !this_set->setName ){
			this_set->setName= concat( the_file, " ", "%CY", NULL );
		}
		xfree(SSdt.sample);
	}
	else{
		fprintf( StdErr, "IEFio::import_IEF(%s): file not open or other read problem (%s)\n",
			the_file, serror()
		);
	}
	GCA();
	return( ret );
}

#if 0
static int initialised= False;
static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= IEFio_Function;
  static char called= 0;
  int i;
  char buf[64];

	afVehicleSelector->accessHandler= &internal_AHandler;
	for( i= 0; i< IEFio_Functions; i++, af++ ){
		if( !called ){
			if( af->name ){
				af->name= XGstrdup( af->name );
			}
			else{
				sprintf( buf, "Function-%d", i );
				af->name= XGstrdup( buf );
			}
			af->old_value= af->value;
			if( strcmp( af->name, "$IEF-Recalibrate-AnalogSensors" )== 0 ){
				afStandstillRecalibrateAnalogSensors= af;
			}
			else if( strcmp( af->name, "$IEF-Import-Range" )== 0 ){
				afImportRange= af;
			}
			if( af->usage ){
				af->usage= XGstrdup( af->usage );
			}
			ascanf_CheckFunction(af);
			if( af->function!= ascanf_Variable ){
				set_NaN(af->value);
			}
			if( label ){
				af->label= XGstrdup( af->label );
			}
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

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
#ifdef XG_DYMOD_IMPORT_MAIN
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
		XGRAPH_FUNCTION(register_VariableNames_ptr, "register_VariableNames");
#endif
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, new->name, new->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		{ ascanf_Function *af= afStandstillRecalibrateAnalogSensors;
			if( (afStandstillRecalibrateAnalogSensors= get_VariableWithName( "$IEF-Recalibrate-AnalogSensors", False )) ){
				af= afStandstillRecalibrateAnalogSensors;
				Resize_ascanf_Array( af, 2, NULL );
				if( af->N!= 2 || !af->array ){
					af->type= _ascanf_variable;
					xfree(af->array);
				}
				else{
					af->array[0]= af->old_value;
					StandstillRecalibrateAnalogSensors= af->array;
				}
			}
			else{
				afStandstillRecalibrateAnalogSensors= af;
			}
			  // initialise $IEF-Import-Range, making it a 2-element array of doubles:
			af= afImportRange;
			if( (afImportRange= get_VariableWithName( "$IEF-Import-Range", False )) ){
				af= afImportRange;
				Resize_ascanf_Array( af, 2, NULL );
				if( af->N!= 2 || !af->array ){
					af->type= _ascanf_variable;
					ImportRange = NULL;
					xfree(af->array);
				}
				else{
					af->array[0]= af->old_value;
					ImportRange= af->array;
					set_Inf(ImportRange[0], -1);
					set_Inf(ImportRange[1], 1);
				}
			}
			else{
				afImportRange= af;
			}
		}
		SelectVehicle();
		initialised= True;
	}
}


// see the explanation printed by wrong_dymod_loaded():
void R_init_IEFio()
{
	wrong_dymod_loaded( "R_init_IEFio()", "R", "IEFio.so" );
}
#endif

