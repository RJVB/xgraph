#ifndef _READDATA_H
#define _READDATA_H

/* 20020506: implementation of a more flexible input format that is less dependent on empty lines
 \ as terminators. Selected commands can have a closing form that delineates the scope.
 \ An additional starting asterisk will activate the mechanism:
 \ E.g. (transparent expansion; *COMMAND* syntax would remain possible and valid):
 \ **EVAL*
 \ 	DEPROC-noEval[foo,progn[
 \		....
 \	] ] @
 \ !EVAL!
 \ XML-like syntax would also be possible, but this would require a change to the whole syntax:
 \ <EVAL>
 \	blablabla
 \ </EVAL>
 \ This will of course require a completely new ReadData() routine, whereas the old should be preserved
 \ to allow old files to be read. It could be placed in a DyMod that gets loaded automatically whenever
 \ a keyword misses at the head of the file that identifies it as new style input.
 \ _XGraphDump will have to be rewritten too.
 */

typedef enum XGCommand {
	ic_ENDIAN= 0,
	  /* ..etc.. */
	ic_UNKNOWN, XGCommands
} XGCommand;

typedef struct InputCommand{
	char *opcode;			/* an opcode of the form COMMAND, e.g. DATA_PROCESS -- thus without the asterisks */
	short active, has_EOC;	/* have we seen the COMMAND; will we ever expect a closing form (End of Command)? */
	unsigned long hash, EOC_hash;
	int (*handler)( DataSet *this_set, char *optbuf, char *buffer, char *filename, int sub_div, double data[ASCANF_DATA_COLUMNS], Boolean *DF_bin_active, Boolean *LineDumped );
} InputCommand;

/* A collection of state variables used inside ReadData(), and necessary/useful for calling functions like AddPoint()
 \ from handlers called by ReadData().
 */
typedef struct ReadData_States{
	int *spot, *Spot;
	int sub_div, line_count;
	struct FileLinePos *flpos;
	struct Process *ReadData_proc;
} ReadData_States;

/* The library hook structure that contains pointers to the handlers provided by a DM_IO module.
 \ The type field should be DM_IO
 \ In addition, ascanf variables can be provided via the usual mechanism.
 */
typedef struct DM_IO_Handler{
	DyModTypes type;
	  /* Handler for importing data from the indicated file into the indicated DataSet. <stream> will be open,
	   \ and should not be closed by this handler!!
	   \ The return value is not currently used, but should reflect a measure for the quantity of data read successfully.
	   */
	int (*import)(FILE *stream, char *the_file, int filenr, int Sets, struct DataSet *this_set, ReadData_States *state );
	  /* Handler for dumping data. This should follow the conventions for the output routines registered via the hard_devices
	   \ mechanism. Cf. XGraphDump().
	   */
	int (*export)( FILE *fp, int width, int height, int orient,
		char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
		LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
	);
} DM_IO_Handler;

#endif
