/*
 * Hardcopy Devices
 *
 * This file contains the basic output device table.  The hardcopy
 * dialog is automatically constructed from this table.
 */

#include "config.h"
IDENTIFY( "Supported devices table" );

#include <stdio.h>

#include "xgout.h"
#include "hard_devices.h"

#include "copyright.h"

extern int hpglInit();
extern int psInit();
extern int idrawInit();
extern int SpreadSheetDump();
extern int CricketDump(), XGraphDump(), DumpCommand();

#ifdef IDRAW_DUMP
/* some bogus 11.3 compatibility:	*/
#define D_DOCU	0x01
int _idrawInit( file, width, height, orient, tf, ts, lf, ls, af, as, outInfo, errmsg, initFile)
FILE *file;				/* Output file            */
int width, height;		/* In microns             */
int orient;
char *tf, *af, *lf;			/* Title and axis font    */
double ts, as, ls;			/* Title and axis size    */
xgOut *outInfo;			/* Returned device info   */
char errmsg[ERRBUFSIZE];	/* Returned error message */
int initFile;
{
	return( idrawInit( file, width, height, tf, ts,
				lf, ls, af, as, D_DOCU, outInfo, errmsg, initFile
			)
	);
}
#endif

struct hard_dev __hard_devices[] = {
    {	"Postscript",
		psInit,
		"lpr -P%s",
		"xgraph.ps",
		"lps40",
		19, 26.5,
		"Palatino-Bold",
		15.5,
		"Palatino-Bold",
		12,
		"Palatino-Bold",
		10,
		"Helvetica",
		12.0
	},
	{	"SpreadSheet",
		SpreadSheetDump,
		"",
		"xgraph.csv",
		"/dev/null",
		19, 26.5,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0
	},
	{	"Cricket",
		CricketDump,
		"",
		"xgraph.cg",
		"/dev/null",
		19, 26.5,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0
	},
#ifdef HPGL_DUMP
     {	"HPGL",
		hpglInit,
		"lpr -P%s",
		"xgraph.hpgl",
		"paper",
		19, 26.5,
		"1",
		14.0,
		"1",
		14.0,
		"1",
		14.0,
		"1",
		12.0
	},
#endif
#ifdef IDRAW_DUMP
    {	"Idraw", _idrawInit,
		"cat > /usr/tmp/idraw.tmp.ps; lpr -P%s /usr/tmp/idraw.tmp.ps&",
		"~/.clipboard",
		"lps40",
		19, 26.5,
		"Palatino-Bold",
		15.5,
		"Palatino-Bold",
		12,
		"Palatino-Bold",
		10,
		"Helvetica",
		12.0
	},
#endif
	{	"XGraph",
		XGraphDump,
		"",
		"xgraph.xg",
		"/dev/null",
		19, 26.5,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0
	},
	{	"Command",
		DumpCommand,
		"",
		"xgraph.sh",
		"/dev/null",
		19, 26.5,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0,
		"",
		-1.0
	},
};

struct hard_dev *hard_devices= NULL;

int hard_count = sizeof(__hard_devices)/sizeof(struct hard_dev);

  /* Initialise the global hard_Devices structure. This (mainly) recopies
   \ some fields that are dynamic strings, but have been initialised (above)
   \ with (global,) static strings.
   */
void Init_Hard_Devices()
{ int i;
  char *c;

	if( !hard_devices ){
		if( !(hard_devices= (struct hard_dev*) calloc( hard_count, sizeof(struct hard_dev) )) ){
			hard_devices= __hard_devices;
		}
	}
	if( hard_devices ){
		if( hard_devices!= __hard_devices ){
			memcpy( hard_devices, __hard_devices, hard_count* sizeof(struct hard_dev) );
		}
		for( i= 0; i< hard_count; i++ ){
			c= hard_devices[i].dev_file;
			hard_devices[i].dev_spec= NULL;
			hard_devices[i].dev_file= NULL;
			stralloccpy( &hard_devices[i].dev_file, c, MFNAME-1 );
			c= hard_devices[i].dev_printer;
			hard_devices[i].dev_printer= NULL;
			stralloccpy( &hard_devices[i].dev_printer, c, MFNAME-1 );
		}
	}
}

  /* Copies one hard_devices structure into another, taking care to
   \ (re)allocate the dynamic strings it contains.
   */
Hard_Devices *Copy_Hard_Devices( Hard_Devices *dest, Hard_Devices *src )
{ int i;
	if( dest && src ){
		for( i= 0; i< hard_count; i++ ){
		  char *df= dest[i].dev_file, *dp= dest[i].dev_printer, *ds= dest[i].dev_spec;
			memcpy( &dest[i], &src[i], sizeof(Hard_Devices) );
			dest[i].dev_file= df;
			dest[i].dev_printer= dp;
			dest[i].dev_spec= ds;
			stralloccpy( &dest[i].dev_file, src[i].dev_file, MFNAME- 1 );
			stralloccpy( &dest[i].dev_printer, src[i].dev_printer, MFNAME- 1 );
			stralloccpy( &dest[i].dev_spec, src[i].dev_spec, MFNAME- 1 );
		}
	}
	return( dest );
}

