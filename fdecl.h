/* 20001203 (C) RJVB.
 \ All function declarations as they exist now. To be kept uptodate: see the 20001203 note in TODOHIST.
 */

#ifndef _FDECL_H
#define _FDECL_H

#include <math.h>
#include <limits.h>

#include "xtb/xtb.h"
#include "xgout.h"
#include "new_ps.h"
#include "DataSet.h"

#include "SS.h"
#include "Sinc.h"
#include "ascanf.h"

extern FILE *StdErr;

#include "xgerrno.h"

#include <string.h>
#include <strings.h>

#include "hard_devices.h"

#if defined(__APPLE__) 
#	if !defined(NATIVE_SINCOS)
#		include "AppleVecLib.h"
#	endif
#endif

// 20101105: the _xfree and _xfree_setitem prototypes were moved to xfree.h. Include that file if it hasn't yet been included,
// but do not define the xfree and xfree_setitem macros as that never happened in fdecl.h
#ifndef _XFREE_H
#	define NDEF_XFREE
#	define NDEF_XFREE_SETITEM
#	include "xfree.h"
#	undef NDEF_XFREE
#	undef NDEF_XFREE_SETITEM
#endif

// #ifdef __cplusplus
// extern "C" {
// #endif

/* Function declarations for ascanfc*.c:	*/

  /* part of the input buffer <s> passed to fascanf() that remains unparsed. For single expression lists, this will
   \ be the empty string (e.g. when s="e1,e2,e3"). However, when s="e1,e2 e3", this variable will point to the remaining
   \ part " e3". Thus, multiple expression(list)s can be specified in a single string.
   */
extern char *fascanf_unparsed_remaining;

extern int fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **form );
extern int compiled_fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **Form);
extern int Destroy_Form( struct Compiled_Form **form );
extern int Ascanf_AllocMem( int elems );
extern int show_ascanf_functions( FILE *fp, char *prefix, int do_bold, int lines );
extern long ascanf_hash( char *name, unsigned int *hash_len);
extern int Resize_ascanf_Array( ascanf_Function *af, int N, double *result );
extern void Check_linkedArray(ascanf_Function *af);
extern int Check_Form_Dependencies( struct Compiled_Form **Form );
extern int ascanf_CheckFunction( struct ascanf_Function *af );
extern int add_ascanf_functions( ascanf_Function *array, int n, char *caller);
extern int add_ascanf_functions_with_autoload( ascanf_Function *array, int n, char* libname, char *caller);
extern int remove_ascanf_functions( ascanf_Function *array, int n, int force );
extern int Copy_preExisting_Variable_and_Delete( ascanf_Function *af, char *label );
extern int ascanf_frame_has_expr;
extern int ascanf_arguments;
extern char *ascanf_emsg;
extern char *ad2str( double d, const char *Format , char **Buf );
extern double normal_rand( int i, double av, double stdv );
extern char *TBARprogress_header;
extern int ascanf_escape;
extern double *ascanf_escape_value;
#ifndef linux
	extern char *strcasestr( const char *a,  const char *b);
#endif
extern ascanf_Function *Procedure_From_Code( void *code );
extern int Check_ascanf_SS();
extern int Check_ascanf_SAS();
extern char *grl_MatchVarNames( char *string, int state );
extern char *getFileName( char *path );
extern char *get_fdName( int fd );
extern ascanf_Function *Create_Internal_ascanfString( char *string, int *level );
extern ascanf_Function *Create_ascanfString( char *string, int *level );
extern ascanf_Function *init_Static_StringPointer( ascanf_Function *af, char *AFname );

extern Window ascanf_window;
extern LocalWin *ActiveWin, *LastDrawnWin, *StubWindow_ptr, *HO_PreviousWin_ptr;
extern LocalWin *InitWindow;

/* Function declarations for ascanfc2.c	*/
extern double *ascanf_getDataColumn( int set_nr, int col_nr, int *N );

/* Function declarations for dialog.c	*/

extern char XG_PS_NUp_buf[];
extern void CloseHO_Dialog( xtb_frame *dial);
extern xtb_hret HO_dev_fun(Window win, int previous, int current, xtb_data info);
extern xtb_hret HO_ok_fun( Window win, int bval, xtb_data info);
extern int Handle_HO_Event( XEvent *evt, int *handled, xtb_hret *xtb_return, int *level, int handle_others );
extern int Isdigit( unsigned int ch );
extern int StringCheck( char *s, int maxlen, char *file, int line );
extern void _CloseHO_Dialog( xtb_frame *dial, Boolean destroy);
extern void _do_message( char *mesg, int err);
extern void do_error( char *err);
extern void do_message(  char *mesg);
extern int ho_dialog( Window theWindow, LocalWin *win_info, char *prog, xtb_data cookie, char *title, char *in_title);
extern void set_HO_printit_win();
extern xtb_hret SimpleFileDialog_h( Window win, int bval, xtb_data info );
extern xtb_hret SimpleFileDialog2_h( Window win, int bval, xtb_data sourcedest_window );

/* Function declarations for dialog_s.c	*/

extern void CloseSD_Dialog( xtb_frame *dial );
extern int Data_SN_Label( char *number );
extern void Data_SN_Number( char *number );
extern char *Data_fileName();
extern char *GetLabelNr( int nr);
extern UserLabel *GetULabelNr( int nr);
extern int Handle_SD_Event( XEvent *evt, int *handled, xtb_hret *xtb_return, int *level, int handle_others );
extern char *LegendorTitle(int data_sn_number, int mode);
extern int SD_get_errb();
extern xtb_hret SD_help_fnc(Window win, int bval, xtb_data info);
extern xtb_hret SD_option_hist(Window win, int bval, xtb_data info);
extern xtb_hret SD_process_hist(Window win, int bval, xtb_data info);
extern xtb_hret SD_quit_fun( Window win, int val, xtb_data info);
extern xtb_hret SD_selectfun(Window win, int bval, xtb_data info);
extern int SD_set_errb( int *type);
extern xtb_hret SD_set_info(Window win, int bval, xtb_data info);
extern xtb_hret SD_set_bardimensions(Window win, int bval, xtb_data info);
extern xtb_hret SD_snl_fun(Window win, int ch, char *text, xtb_data val);
extern int SD_strlen(const char *s);
extern void _CloseSD_Dialog( xtb_frame *dial, Boolean destroy );
extern int barFlag_Value();
extern char *d3str(char *buf,char *format, double val);
extern xtb_hret display_ascanf_variables_h(Window win, int bval, xtb_data info);
extern int draw_set_Value();
extern int end_arrow_Value();
extern int find_fileName_max_AND_legend_len();
extern int floating_Value();
extern int format_SD_Dialog( xtb_frame *frame, int discard );
extern void get_data_legend_buf();
extern int get_error_type( LocalWin *wi, int snr );
extern int highlight_Value();
extern int markFlag_Value();
extern int mark_set_Value();
extern int noLines_Value();
extern int overwrite_marks_Value();
extern int pixelMarks_Value();
extern int points_added_Value();
extern int raw_display_Value();
extern int set_changeables(int do_it,int allow_auto_redraw);
extern int set_error_type( LocalWin *wi, int snr, int *type, int no_fit );
extern int settings_dialog(Window theWindow, LocalWin *win_info, char *prog, xtb_data cookie, char *title, char *in_title);
extern int show_legend_Value();
extern int show_llines_Value();
extern xtb_hret slide_f( Window win, int pos, double val, xtb_data info);
extern xtb_hret snn_slide_f( Window win, int pos, double val, xtb_data info);
extern int start_arrow_Value();
extern int update_SD_size();
extern int use_error_Value();
extern int SimpleEdit( char *text, Sinc *result, char *errbuf );
extern xtb_hret SimpleEdit_h(Window win, int bval, xtb_data info);
extern int Parse_vectorPars( char *buffer, DataSet *this_set, int global_copy, int *Changed, char *caller );


/* Function declarations for fascanf.c	*/

extern Sinc *SSputc( int c, Sinc *sinc );
extern Sinc *SSputs( char *text, Sinc *sinc );
extern int Sflush( Sinc *sinc );
extern int SincAllowExpansion( Sinc *sinc );
extern Sinc *Sinc_base( Sinc *sinc, long base );
extern Sinc *Sinc_file( Sinc *sinc, FILE *file, long cnt, long base );
extern Sinc *Sinc_string( Sinc *sinc, char *string, long cnt, long base );
extern Sinc *Sinc_string_behaviour( Sinc *sinc, char *string, long cnt, long base, SincString behaviour );
extern int Sputc( int c, Sinc *sinc );
extern int Sputs( char *text, Sinc *sinc );
extern int _EndJoin( char *buf, char *first, int dlen, int instring );
extern int _StartJoin( char *buf, char *first, int dlen, int instring );
extern char *ascanf_index( char *s, char dum, int *instring);
extern int fascanf2( int *n, char *s, double *a, int separator);
#undef GetEnv
#undef SetEnv
extern char* GetEnv( char *n);
extern char* SetEnv( char *n, char *v);
  /* Generic string comparison function: returns True for a match.
   \ If a and b and b starts with 'RE^' and ends with '$', re_comp/re_exec are used to
   \ perform a regular expression matching.
   */
extern int streq( char *a, char *b, int n );
extern char *ReadString( char *buffer, int len, FILE *fp, int *line, 
	  /* Start joining mode when this method returns true. When it returns 2, really only exit this mode
	   \ when EndJoining says so.
	   */
	DEFMETHOD( StartJoining, (char *buf, char *first_nspace, int len, int instring), int ),
	DEFMETHOD( EndJoining, (char *buf, char *first_nspace, int len, int instring), int )
);
extern int Use_ReadLine;
extern char *ReadLine( char *buffer, int len, char *prompt, int *line, 
	DEFMETHOD( StartJoining, (char *buf, char *first_nspace, int len, int instring), int ),
	DEFMETHOD( EndJoining, (char *buf, char *first_nspace, int len, int instring), int )
);


/* Function declarations for hard_devices.c	*/

extern Hard_Devices *Copy_Hard_Devices( Hard_Devices *dest, Hard_Devices *src );
extern void Init_Hard_Devices();
extern int _idrawInit();

/* Function declarations for main.c	*/

extern void CheckEndianness();
extern Window LocalWin_window( LocalWin *wi );
#ifndef _MAIN_C
	extern LocalWin *aWindow(LocalWin *wi);
	extern int XG_error_box( LocalWin **wi, char *title, char *mesg, ... );
#endif
extern void *XSynchronise( Display *disp, Bool onoff );
extern void testrfftw(char *fname, int linenr, int cnt);
extern FILE *NullDevice;

/* Function declarations for matherr.c	*/

extern double Entier(double x);
extern int XGstrcasecmp(const char    *s1, const char     *s2);
extern int XGstrcmp(const char *s1, const char *s2);
extern int XGstrncasecmp(const char *s1, const char *s2, size_t n);
extern int XGstrncmp(const  char *s1, const     char *s2, size_t n);
extern int d2str_printhex;
extern char *d2str( double d, const char *Format , char *buf );
extern double dcmp( double b, double a, double prec);
extern double dcmp2( double b, double a, double prec);
extern int matherr(struct exception *x);
extern char *matherr_mark( char *mark );
extern void matherr_report();
extern int q_Permute( void *a, int n, int size, int quick);

/* Function declarations for new_ps.c	*/

extern int CustomFont_psHeight( CustomFont *cf );
extern int CustomFont_psWidth( CustomFont *cf );
extern void PS_set_ClearHere( char *state );
extern char *PSconvert_colour( FILE *fp, Pixel pixel);
extern void PSprint_string( FILE *fp, char *header, char *wrapheader, char *trailer, char *string, int is_comment );
extern int Write_ps_comment( FILE *fp );
extern int psInit( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
);
extern void gsResetTextWidths( LocalWin *wi, int destroy );
extern double gsTextWidth( LocalWin *wi, char *text, int style, CustomFont *cfont);
extern int gsTextWidthBatch( LocalWin *wi );
extern void psMessage( void *state, char *message);
extern int psSilent( char *user_state, Boolean silent);
extern double ps_MarkSize_X( struct userInfo *ui, double setnr );
extern double ps_MarkSize_Y( struct userInfo *ui, double setnr );
extern int rd(double dbl);
extern double Font_Width_Estimator;

/* Function declarations for regex.c	*/

extern char *xg_re_comp(const char *pat);
extern int xg_re_exec( const char *lp);
extern char *xg_re_exec_len(char *lp);

/* Function declarations for xgInput.c	*/

typedef struct FileLinePos{
	FILE *stream;
	long last_fpos;
	int last_line_count;
} FileLinePos;

extern int AddEllipse( DataSet **this_set, double x, double y, double rx, double ry, int points, double skew,
	int *spot, int *Spot, double data[4], int column[4],
	char *filename, int sub_div, int line_count, FileLinePos *flpos, char *buffer, 
	Process *proc
);
extern int AddPoint( DataSet **This_Set, int *spot, int *Spot, int numcoords, double *Data, int column[4],
	char *filename, int sub_div, int line_count, FileLinePos *flpos, char *buffer, Process *proc
);
extern int AllocBinaryFields( int nc, char *caller );
extern int Check_Option( int (*compare_fun)(), char *arg, char *check, int len);
extern int CleanUp_Sets();
extern void Destroy_Set( DataSet *this_set, int relinking );
extern LocalWindows *Find_WindowListEntry_String( char *expr, LocalWin **wi1, LocalWin **wi2 );
extern int GetColorDefault( char *value, Pixel *tempPixel, char *def_res_name, char *def_col_name, Pixel *def_pix );
extern int IncludeFile( LocalWin *theWin_Info, FILE *strm, char *Fname, int redraw_event, char *skip_to_label );
extern int interactive_IncludeFile(LocalWin *wi, char *msg, char *ret_fn);
extern int ReadScriptFile( LocalWin *wi );
extern int IdentifyStreamFormat( char *fname, FILE **strm, int *is_pipe );
extern int IsLabel( char *optbuf );
extern int LinkSet2( DataSet *this_set, int set_link );
extern int ListKeyParamExpressions( FILE *fp, int newline );
extern int NewSet( LocalWin *wi, DataSet **this_set, int spot );
extern int ParseArgs( int argc, char *argv[]);
extern int ParseArgsString( char *string );
extern int ParseArgsString2( char *optbuf, int set_nr );
extern int ParseArgsString3( char *argbuf, int set_nr );
extern int ParseInputString( LocalWin *wi, char *input );
extern double WaitForEvent( LocalWin *wi, int type, char *msg, char *caller );
extern char *strrstr(const char *a, const char *b);
extern char *XGbasename( char *path );
extern char *XGdirname( char *path );

#if !defined(__APPLE__)
#	define basename(p)	XGbasename(p)
#	define dirname(p)	XGdirname(p)
#endif

extern char *SubstituteOpcodes( char *source, VA_DCL );
extern int ReadData(FILE *stream, char *the_file, int filenr);
extern char *ReadData_AllocBuffer( char *buffer, int length, char *the_file );
extern int IOImport_Data_Direct( const char *libname, char *fname, DataSet *this_set );
extern int IOImport_Data( const char *libname, char *fname );
extern int ShiftDataSet( int dsn, int dirn, int to_extreme, int do_redraw );
extern int ShiftDataSets_Drawn( LocalWin *lwi, int dirn, int extreme, int do_redraw );
extern char *String_ParseVarNames( char *instring, char *opcode_start, int opcode_terminator, int include_varname, int no_cook, char *caller );
extern int SwapDataSet( int set_a, int set_b, int do_redraw );
extern double *SwapEndian_double( double *data, int N);
extern float *SwapEndian_float( float *data, int N);
extern int *SwapEndian_int( int *data, int N);
extern int32_t *SwapEndian_int32( int32_t *data, int N);
extern int16_t *SwapEndian_int16( int16_t *data, int N);
extern int XGStartJoin( char *buf, char *first, int dlen, int instring );
extern int XGEndJoin( char *buf, char *first, int dlen, int instring );
extern XGStringList *XGStringList_FindItem( XGStringList *list, char *text, int *item );
extern XGStringList *XGStringList_AddItem( XGStringList *list, char *text );
extern XGStringList *XGStringList_Pop( XGStringList *list );
extern XGStringList *XGStringList_PopLast( XGStringList *list );
extern XGStringList *XGStringList_Delete( XGStringList *list );
extern void *XGcalloc( size_t n, size_t s);
extern double **XGrealloc_2d_doubles( double **cur_columns, int ncols, int nlines, int cur_ncols, int cur_nlines, char *caller );
extern char *add_option_hist( char *expression );
extern int argerror( char *err, char *val);
extern int change_stdfile( char *newfile, FILE *stdfile );
extern DataSet *find_NewSet( LocalWin *wi, int *idx );
extern int interactive_parse_string( LocalWin *wi, char *expr, int tilen, int maxlen, char *message, int modal, double verbose );
extern int next_gen_include_file( char *command, char *fname );
extern void option_hist(LocalWin *wi);
extern int realloc_FileNames( LocalWin *wi, int offset, int N, char *caller );
extern int realloc_LocalWinList_data( LocalWin *wi, int start, int N );
extern void realloc_Xsegments();
extern double **realloc_columns( DataSet *this_set, int ncols );
extern void realloc_points( DataSet *this_set, int allocSize, int force );
extern int realloc_sets( LocalWin *wi, int offset, int N, char *caller );
extern void set_Columns( DataSet *this_set);

extern LabelsList *ColumnLabels;
DEFUN( *Find_LabelsList, (LabelsList *llist, int column), LabelsList);
DEFUN( *Find_LabelsListLabel, (LabelsList *llist, int column), char);
DEFUN( Find_LabelsListColumn, ( LabelsList *llist, char *label, int partial, int withcase), int );
DEFUN( *Add_LabelsList, (LabelsList *current_LList, int *current_N, int column, char *label), LabelsList);
DEFUN( *Copy_LabelsList, (LabelsList *dest, LabelsList *llist), LabelsList );
DEFUN( *Free_LabelsList, (LabelsList *llist), LabelsList);
DEFUN( LabelsList_N, (LabelsList *llist), int);
DEFUN( *Parse_SetLabelsList, ( LabelsList *llist, char *labels, int separator, int nCI, int *ColumnInclude ), LabelsList );
DEFUN( Evaluate_ExpressionList, ( LocalWin *wi, XGStringList **Exprs, int dealloc, char *descr ), int );

DEFUN( XG_SimpleConsole, (), void );

extern int scriptVerbose;

/* Function declarations for xgX.c	*/

extern void _arc_X( Display *disp, Window win, GC gc, int x, int y, int rx, int ry, double la, double ha );
extern void AddXClip( LocalWin *wi, int x, int y, int w, int h);
extern int CustomFont_height_X( CustomFont *cf );
extern int CustomFont_width_X( CustomFont *cf );
extern void Default_Intensity_Colours();
extern XFontStruct *Find_greekFont( char *name, char *greek );
extern int GetCMapColor( char *Name, Pixel *pix, Colormap colourmap );
extern void FreeColor( Pixel *pix, char **cname );
extern void Free_CustomFont( CustomFont *cf );
extern int GetColor( char *Name, Pixel *pix );
extern char *GetFont( XGFontStruct *Font, char *resource_name, char *default_font, long size, int bold, int use_remembered );
extern int GetPointSize( char *fname, int *pxsize, int *ptsize, int *xres, int *yres );
extern CustomFont *Init_CustomFont( char *xfn, char *axfn, char *psfn, double pssize, int psreencode );
extern int Intensity_Colours( char *exp );
extern int New_XGFont( long which, char *font_name );
extern int ReallocColours(Boolean do_redraw);
extern void RecolourCursors();
extern void SetWindows_Cursor( Cursor curs );
extern void SetXClip( LocalWin *wi, int x, int y, int w, int h);
extern void Set_X( LocalWin *new_info, xgOut *out_info );
extern void UnGetFont( XGFontStruct *Font, char *resource_name);
extern int Update_greekFonts(long which);
extern int XErrHandler( Display *disp, XErrorEvent *evt );
extern int XFatalHandler( Display *disp);
extern XGFontStruct *XGFont( long which, int *best_crit );
extern int XGFontWidth( LocalWin *wi, int style, char *text, int *width, int *height, CustomFont *cfont,
	XCharStruct *bb, double *fontScale);
int XGFontWidth_Lines( LocalWin *wi, int FontNR, char *text, char letter, int *width, int *theight, int *height,
	CustomFont *cfont, XCharStruct *bb, double *scale );
extern int XGTextWidth( LocalWin *wi, char *text, int style, CustomFont *cfont);
extern int XG_DisplayHeightMM( Display *disp, int screen );
extern int XG_DisplayWidthMM( Display *disp, int screen );
extern int XG_DisplayXRes( Display *disp, int screen );
extern int XG_DisplayYRes( Display *disp, int screen );
extern void XG_choose_visual();
extern GC X_CreateGC(Display *disp, Window win, unsigned long gcmask, XGCValues *gcvals, char *fnc, char *__file__, int __line__);
extern int Synchro_State;
extern void* X_Synchro(LocalWin *wi);
extern int X_ps_MarkSize_X( double setnr);
extern int X_ps_MarkSize_Y( double setnr);
extern int X_ps_Marks( char *user_state, int ps_marks );
extern int X_silenced( LocalWin *wi );
extern double _X_ps_MarkSize_X( double setnr);
extern double _X_ps_MarkSize_Y( double setnr);
extern void _arc_X( Display *disp, Window win, GC gc, int x, int y, int rx, int ry, double la, double ha );
extern void arc_X(char *user_state, int x, int y, int rx, int ry, double La, double Ha, double Width,
	int style, int lappr, int colour, Pixel pixval);
extern void close_X(char *user_state);
extern void init_X( char *user_state);
extern XRectangle *rect_diag2xywh( int x1, int y1, int x2, int y2 );
extern XRectangle *rect_xsegs2xywh( int ns, XSegment *segs );
extern XRectangle *rect_xywh( int x, int y, int width, int height );
extern int silence_X( char *user_state, int silent);

/* Function declarations for xgraph.c	*/


extern double LastActionDetails[5];
extern int QuietErrors, quiet_error_count;
extern int StartUp;
extern int Num_Windows;

extern void AdaptWindowSize( LocalWin *wi, Window win, int w, int h );
extern ValCategory *Add_ValCat( ValCategory *VCat, int *current_N, double value, char *category );
extern void Add_mStats( LocalWin *wi );
extern int Adorned_Points( DataSet *this_set );
extern int CShrinkArea( LocalWin *wi, /* double _loX, double _loY, double _hiX, double _hiY,	*/
	double ulX, double ulY, int ulx, int uly, int lrx, int lry, int uhptc
);
extern void ChangeCrossGC( GC *CrossGC);
extern void Check_Columns(DataSet *this_set);
extern int Check_Process_Dependencies( LocalWin *wi );
extern int ClipWindow( LocalWin *wi, DataSet *this_set, int floating,
	double *sx1, double *sy1, double *sx2, double *sy2, int *mark_inside1, int *mark_inside2, int *clipcode1, int *clipcode2
);
extern void CollectPointStats( LocalWin *wi, DataSet *this_set, int pnt_nr, double sx1, double sy1, double sx3, double sy3, double sx4, double sy4 );
extern LocalWin *CopyFlags( LocalWin *dest, LocalWin *src );
extern void DrawCCross( LocalWin *lwi, Boolean erase, int curX, int curY, char *label );
extern void DrawCCrosses( LocalWin *wi, XEvent *evt, double cx, double cy, int curX, int curY, char *label, char *caller );
extern void DrawData(LocalWin *wi, Boolean bounds_only);
extern int DrawData_process(LocalWin *wi, DataSet *this_set, double data[2][4], int subindex,
	int nr, int ncoords,
	double *sx1, double *sy1,
	double *sx2, double *sy2,
	double *sx3, double *sy3, double *sx4, double *sy4,
	double *sx5, double *sy5, double *sx6, double *sy6
);
extern int AddAxisValue( LocalWin *wi, AxisValues *av, double val );
extern int AxisValueCurrentLabelled( LocalWin *wi, AxisValues *av, int label );
extern int CompactAxisValues( LocalWin *wi, AxisValues *av );
extern int DrawGridAndAxis( LocalWin *wi, int doIt );

#ifdef ULabelTypes
	extern char *ULabelTypeNames[UL_types+1];
	extern ULabelTypes Parse_ULabelType( char ULtype[2] );
	extern UserLabel *Add_UserLabel( LocalWin *wi, char *labeltext, double x1, double y1, double x2, double y2,
		int point_label, DataSet *point_label_set, int point_l_nr, double point_l_x, double point_l_y,
		ULabelTypes type,
		int allow_name_trans, unsigned int mask_rtn_pressed, unsigned int mask_rtn_released,
		int immediate
	);
#else
	extern char *ULabelTypeNames[];
#endif
extern UserLabel *Install_ULabels( UserLabel *dst, UserLabel *src, int copy );
extern int Copy_ULabels( LocalWin *dst, LocalWin *src );
extern int Delete_ULabels( LocalWin *wi );
extern void DrawULabels( LocalWin *wi, int pass, int doit, int *prev_silent, void *dimension_data );

extern int DrawLegend( LocalWin *wi, int doit, void *dimension_data );
extern int DrawLegend2( LocalWin *wi, int doit, void *dimension_data );
extern int DrawIntensityLegend( LocalWin *wi, Boolean doit );

extern char *ParseTitlestringOpcodes( LocalWin *wi, int idx, char *title, char **parsed_end );
extern int DrawTitle( LocalWin *wi, int draw);
extern int DrawWindow( LocalWin *wi);
extern void Draw_Bar( LocalWin *wi, XRectangle *rec, XSegment *line, double barPixels, int barType, int LStyle,
	DataSet *this_set, int set_nr, int pnt_nr, int lwidth, int lstyle, int olwidth, int olstyle,
	double minIntense, double maxIntense, double scaleIntense, Pixel colorx, int respix
);
extern void Draw_ErrorBar( LocalWin *wi, DataSet* this_set, int set_nr, int pX_set_nr, int X_set_nr, int pnt_nr, int first,
	double ebarPixels, int LStyle,
	double aspect
);
extern int Draw_Process(LocalWin *wi, int before );
extern void Draw_valueMark( LocalWin *wi, DataSet *this_set, int pnt_nr,
	short sx, short sy, int colour, Pixel pixval
);
extern ValCategory *Find_ValCat( ValCategory *vcat, double val, ValCategory **low, ValCategory **high );
extern int Fit_After_Draw( LocalWin *wi, char *old_Wname );
extern ValCategory *Free_ValCat( ValCategory *vcat );
extern char *Get_YAverageSorting(LocalWin *wi);
extern inline int LINEWIDTH(LocalWin *wi, int set);
extern double HL_WIDTH(double w);
extern int HandleMouse( char *progname, XButtonPressedEvent *evt, LocalWin *wi, LocalWin **New_Info, Cursor *cur);
extern int Handle_An_Event( int level, int CheckFirst, char *caller, Window win, long mask);
extern int Handle_An_Events( int level, int CheckFirst, char *caller, Window win, long mask);
extern int Handle_MaybeLockedWindows(Bool flush);
extern void HighlightSegment( LocalWin *wi, int idx, int nsegs, XSegment *segs, double width, int LStyle);
extern void InitSets();
extern void Initialise_Sets( int start, int end );
extern int LegendLineWidth(LocalWin *wi, int idx );
extern void MarkerSizes( LocalWin *wi, int idx, int ps_mark_scale, int *mw, int *mh );
extern psUserInfo *PS_STATE(LocalWin *wi );
extern int RedrawAgain( LocalWin *wi );
extern int RedrawNow( LocalWin *wi );
extern Status ExposeEvent( LocalWin *wi );
extern int RedrawSet( int set_nr, Boolean doit );
extern void Retrieve_IntensityColour( LocalWin *wi, DataSet *this_set, double value,
	double minIntense, double maxIntense, double scaleIntense, Pixel *colorx, int *respix
);
extern double RoundUp( LocalWin *wi, double val );
extern char *LocalWinRepr( LocalWin *wi, char *buf );
extern void SetWindowTitle( LocalWin *wi, double time );
extern void ShowAllSets( LocalWin *lwi );
extern void SwapSets( LocalWin *lwi );
extern void TitleMessage( LocalWin *wi, char *msg );
extern int TransformCompute( LocalWin *wi, Boolean warn_about_size );
extern char *ULabel_pixelCName( UserLabel *ul, int *type );
extern char *ULabel_pixelCName2( UserLabel *ul, int *type );
extern Pixel ULabel_pixelValue( UserLabel *ul, Pixel *txtPixel );
extern int UpdateWindowSettings( LocalWin *wi , Boolean is_primary, Boolean dealloc );
extern int ValCat_N( ValCategory *vcat );
extern double WindowAspect( LocalWin *wi );
extern char *XG_GetString( LocalWin *wi, char *text, int maxlen, Boolean do_events );
extern void *XGalloca(void **ptr, int items, int *alloced_items, int size, char *name);
extern double XGpow( double x, double y );
extern int _DiscardedPoint( LocalWin *wi, DataSet *set, int idx);
extern int _Handle_An_Event( XEvent *theEvent, int level, int CheckFirst, char *caller);
extern char *_strcpy( char *d, char *s);
extern int ascanf_ValCat_any( double *args, double *result, ValCategory *ValCat, char *descr, char *caller );
extern void check_marked_hlt( LocalWin *wi, Boolean *all_marked, Boolean *all_hlt, Boolean *none_marked, Boolean *none_hlt );
extern LocalWin *check_wi( LocalWin **wi, char *caller );
extern char *clipcode( int code);
extern void cycle_drawn_sets( LocalWin *wi, int sets );
extern int cycle_highlight_sets( LocalWin *wi, int sets );
extern void cycle_plot_only_file( LocalWin *wi, int files );
extern void cycle_plot_only_group( LocalWin *wi, int groups );
extern int cycle_plot_only_set( LocalWin *wi, int sets);
extern int do_TRANSFORM( LocalWin *lwi, int spot, int nrx, int nry, double *xvec, double *ldxvec, double *hdxvec,
	double *yvec, double *ldyvec, double *hdyvec, int is_bounds, int just_doit);
extern int do_transform( LocalWin *wi, char *filename, double line_count, char *buffer, int *spot_ok, DataSet *this_set,
	double *xvec, double *ldxvec, double *hdxvec, double *yvec, double *ldyvec, double *hdyvec,
	double *xvec_1, double *yvec_1, int use_log_zero, int spot, double xscale, double yscale, double dyscale, int is_bounds,
	int data_nr, Boolean just_doit
);

#define DRAW_SET(wi,draw_set_array,idx,retval) {\
 extern int DrawAllSets; \
	if( (idx)>= setNumber ){ \
		(retval)= 0; \
	} \
	else if( AllSets[(idx)].numPoints<= 0 ){ \
		(retval)= 0; \
	} \
	else if( DrawAllSets || \
		((wi)->AlwaysDrawHighlighted && (wi)->legend_line && (wi)->legend_line[(idx)].highlight) \
	){ \
		(retval)= 1; \
	} \
	else if( (wi)->ctr_A ){ \
		(retval)=( (draw_set_array)[(idx)] ); \
	} \
	else{ \
		(retval)=( (draw_set_array)[(idx)]> 0 ); \
	} \
}

extern int draw_set( LocalWin *wi, int idx);
extern int drawingOrder( LocalWin *wi, int idx);
extern void files_and_groups( LocalWin *wi, int *fn, int *grps );
extern double initGrid( LocalWin *wi, double low, double step, int logFlag, int sqrtFlag, AxisName axis, int polar );
extern xtb_hret label_func( Window win, int bval, xtb_data info);
extern XSegment *make_arrow_point1( LocalWin *wi, DataSet *this_set, double xp, double yp, double ang, double alen, double aspect );
extern void make_sized_marker( LocalWin *wi, DataSet *this_set, double X, double Y, double *sx3, double *sy3, double *sx4, double *sy4,
	double size
);
extern void make_vector( LocalWin *wi, DataSet *this_set, double X, double Y, double *sx3, double *sy3, double *sx4, double *sy4,
	double orn, double vlength
);
extern GC msgGC(LocalWin *wi);
extern int new_tr_LABEL( char *d, char *s );
extern void reset_Scalers(char *msg);
extern char *splitmodestring(LocalWin *wi);
extern double stepGrid( double factor);
extern UserLabel *update_LinkedLabel( LocalWin *wi, UserLabel *new_label, DataSet *point_label_set, int point_l_nr,
	Boolean short_label
);
extern int xgraph( int argc, char *argv[] );

/* Function declarations for xgsupport.c	*/

extern char *CheckPrefsDir(char *name);
extern char *PrefsDir;
extern int AXsprintf( char *str, char *format, double val);
extern char *Add_Comment( char *comment, int left_align );
extern char *Add_SincList( Sinc *List, char *string, int left_align );
extern void AlterGeoBounds( LocalWin *wi, int absflag, double *_loX, double *_loY, double *_hiX, double *_hiY );
extern int Average( LocalWin *wi, int *av_sets, char *filename, int sub_div, int line_count, char *buffer, Process *proc,
	Boolean use_transformed, Boolean XYAveraging, char *YAv_Sort, Boolean add_Interpolations, Boolean ScattEllipse
);
extern int BoxFilter_Undo( LocalWin *wi );
extern int CheckProcessUpdate( LocalWin *wi, int only_drawn, int always, int show_redraw );
extern void CleanUp();
extern char *Collect_Arguments( LocalWin *wi, char *cbuf, int Len );
extern LocalWin *ConsWindow( LocalWin *wi);
extern int CricketDump( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
);
extern double *curvelen_with_discarded;
extern double CurveLen( LocalWin *wi, int idx, int pnr, int update, int signed, double *lengths );
#ifdef TR_CURVE_LEN
extern double tr_CurveLen( LocalWin *wi, int idx, int pnr, int update, int signed, double *lengths );
#endif
extern double ErrorLen( LocalWin *wi, int idx, int pnr, int update );
extern void DelWindowTransform( Transform *transform );
extern void DelWindowProcess( Process *process );
extern int DelWindow( Window win, LocalWin *wi);
extern void XGIconify( LocalWin *wi );
extern int DiscardPoint( LocalWin *wi, DataSet *this_set, int pnt_nr, int dval );
extern int DiscardPoints_Box( LocalWin *wi, double _loX, double _loY, double _hiX, double _hiY, int dval );
extern int DoSettings( Window win, LocalWin *wi);
int DumpCommand( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
);
extern int DumpDiscarded_Points( FILE *fp, LocalWin *wi, DataSet *this_set, int startlen, int linelen, char *trailer );
extern int DumpFILES( LocalWin *wi );
extern void Dump_CustomFont( FILE *fp, CustomFont *cf, char *header );
extern void Dump_handler( int sig);
extern int Exitting;
extern void ExitProgramme(int ret);
extern double FTest( double var1, int n1, double var2, int n2, double *f );
extern int FilterPoints_Box( LocalWin *wi, char *fname,
	double _loX, double _loY, double _hiX, double _hiY, DataSet *this_set, int this_point
);
extern void set_Find_Point_precision( double x, double y, double *ox, double *oy );
extern int Find_Point( LocalWin *wi, double *x, double *y, DataSet **set_rtn, int do_labels, UserLabel **ulabel_rtn,
	Boolean verbose, Boolean use_sqrt, Boolean use_x, Boolean use_y
);
extern int Fit_XBounds( LocalWin *wi, Boolean redraw );
extern int Fit_XYBounds( LocalWin *wi, Boolean redraw );
extern int Fit_YBounds( LocalWin *wi, Boolean redraw );
extern char *FontName( XGFontStruct *f);

extern double Gonio_Base_Value, Gonio_Base_Value_2, Gonio_Base_Value_4;
extern double Units_per_Radian, Gonio_Base_Offset;
extern double Gonio_Base( LocalWin *wi, double base, double offset);
#define Gonio(fun,x)	(fun(((x)+Gonio_Base_Offset)/Units_per_Radian))
#define InvGonio(fun,x)	((fun(x)*Units_per_Radian-Gonio_Base_Offset))
#define Sin(x) Gonio(sin,x)
#define Cos(x) Gonio(cos,x)
#define Tan(x) Gonio(tan,x)
#define ArcSin(x) InvGonio(asin,x)
#define ArcCos(x) InvGonio(acos,x)
#define ArcTan(wi,x,y) (_atan3(wi,x,y)*Units_per_Radian-Gonio_Base_Offset)
#define Atan2(x,y) (atan2(x,y)*Units_per_Radian-Gonio_Base_Offset)

#define Gonio2(fun,x)	(fun(((x))/Units_per_Radian))
#define InvGonio2(fun,x)	((fun(x)*Units_per_Radian))
#define Sin2(x) Gonio2(sin,x)
#define Cos2(x) Gonio2(cos,x)

extern int INSIDE(LocalWin *wi, double _loX, double _hiX, double l, double h);
extern int Interpolate( LocalWin *wi, DataSet *this_set, double tx, int *pnr, double *xp, double *yp, double *ep, double *np, Boolean use_transformed );
extern void LWG_printf( FILE *fp, char *ind, LocalWinGeo *wig);
extern Window NewWindow( char *progname, LocalWin **New_Info, double _lowX, double _lowY, double _lowpX, double _lowpY,
					double _hinY, double _upX, double _upY, double asp,
					LocalWin *parent, double xscale, double yscale, double dyscale, int add_padding
);
extern int Num_Mapped();
extern void PIPE_handler( int sig);
extern int PrintWindow( Window win, LocalWin *wi);
extern int ReadDefaults();
extern double Reform_X( LocalWin *wi, double x, double y);
extern double Reform_Y( LocalWin *wi, double y, double x);
extern char *DisplayDirectory( char *target, int len );
extern int RememberFont( Display *disp, XGFontStruct *font, char *font_name);
extern int RememberedFont( Display *disp, XGFontStruct *font, char **rfont_name);
extern LocalWin *RemoveWindow( LocalWin *wi );
extern void Restart(LocalWin *wi, FILE *showfp );
extern void Restart_handler( int sig);
extern Pixel ReversePixel(Pixel *pixValue);
extern int SCREENX(LocalWin *ws, double userX);
extern int SCREENXDIM(LocalWin *ws, double userX);
extern int SCREENY(LocalWin *ws, double userY);
extern int SCREENYDIM(LocalWin *ws, double userY);
extern void SelectXInput( LocalWin *wi );
extern char *SetColourName( DataSet *this_set );
extern double _fig_dist( LocalWin *wi, DataSet *set1, DataSet *set2, double *SumDist, int *K, int raw, int orn_handling );
extern char *ShowLegends( LocalWin *wi, int PopUp, int this_one );
extern int Show_Ridges( LocalWin *wi, DataSet *ridge_set );
extern int Show_Ridges2( LocalWin *wi, DataSet *ridge_set );
extern void Show_Stats(FILE *fp, char *label, SimpleStats *SSx, SimpleStats *SSy, SimpleStats *SSe, SimpleStats *SS_sy, SimpleStats *SS_SY);
extern int SpreadSheetDump( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
);
extern int Sprint_string( Sinc *fp, char *header, char *wrapstr, char *trailer, char *string );
extern int Sprint_string_string( Sinc *fp, char *header, char *trailer, char *string, int *instring );
extern LocalWin *StackWindows(int direction);
extern double TTest( double mean1, double var1, int n1, double mean2, double var2, int n2, double *t );
extern double TTest_uneq( double mean1, double var1, int n1, double mean2, double var2, int n2, double *t );
extern LocalWin *TileWindows(int direction, int horvert );
extern LocalWin *Tile_Files(LocalWin *wi, Boolean rescale);
extern LocalWin *Tile_Groups(LocalWin *wi, Boolean rescale);
extern double Trans_X( LocalWin *wi, double x);
extern double Trans_XY( LocalWin *wi, double *x, double *y, int is_bounds);
extern double Trans_Y( LocalWin *wi, double y, int is_bounds);
extern double Trans_YX( LocalWin *wi, double *y, double *x, int is_bounds);
extern int Update_LMAXBUFSIZE( int update, char *errmsg );
extern int WINSIDE( LocalWin *wi, double l, double h);
extern char *WriteValue( LocalWin *wi,
	char *str, double val, double val2,
	int exp, int logFlag, int sqrtFlag,
	AxisName axis, int use_real_value, double step_size, int len);
extern char *XGFetchName( LocalWin *wi);
extern char *XG_GetDefault( Display *disp, char *Prog_Name, char *name );
extern int XG_XSync( Display *disp, Bool discard );
extern int XG_sleep_once( double time, Boolean wait_now );
extern int XGraphDump( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
);
extern void Sprint_LabelsList( Sinc *sink, LabelsList *llist, char *header );
extern void Sprint_SetLabelsList( Sinc *sink, LabelsList *llist, char *header, char *trailer );

/* Declared in config.h:
extern void *_XGrealloc( void** ptr, size_t n, char *name, char *size);
 */

extern char *XGrindex( const char *s, char c);
extern char *XGstrcpy( char *dest, const char *src );
extern char *XGstrncpy( char *dest, const char *src, size_t n );
extern char *XGstrdup( const char *c );
extern char *XGstrdup2( char *c , char *c2 );
extern char *XGtempnam( const char *dir, const char *prefix );
extern char *raw_XLABEL(LocalWin *wi );
extern char *XLABEL(LocalWin *wi );
extern double X_Value(LocalWin *wi, double x);
extern char *raw_YLABEL(LocalWin *wi );
extern char *YLABEL(LocalWin *wi );
extern int RawDisplay( LocalWin *wi, int value );
extern double Y_Value(LocalWin *wi, double y);
extern LocalWin *ZoomWindow_PS_Size( LocalWin *wi, int set_aspect, double aspect, Boolean raise );
extern void _DumpSetHeaders( FILE *fp, LocalWin *wi, DataSet *this_set, int *points, int *NumObs, Boolean newline /* , int empty */ );
extern void _Dump_Arg0_Info( LocalWin *wi, FILE *fp, char **pinfo, int add_hist );
extern Pixmap _XCreateBitmapFromData(Display *dis, Drawable win, char *mark_bits, unsigned int mark_w, unsigned int mark_h);
extern void _XGDump_MoreArguments( FILE *fp, LocalWin *wi );
extern char *dd2str( LocalWin *wi, double d, const char *Format , char **Buf );
extern int _XGraphDump( LocalWin *wi, FILE *fp, char errmsg[ERRBUFSIZE] );
extern double _atan3( LocalWin *wi, double x, double y);
extern int _int_MAX( double a, double b);
extern char *add_comment( char *comment );
extern char *add_process_hist( char *expression );
extern ascanf_Function *ascanf_ParseColour( double args, char **name_ret, int strict, char *caller );
extern double atan3( double x, double y);
extern double betacf(double a, double b, double x);
extern double betai(double a, double b, double x);
extern double *clean_param_scratch();
extern char *cleanup( char *T );
extern double cus_log10(double x);
extern void close_pager();
extern char *concat( char *first, VA_DCL );
extern char *concat2( char *string, VA_DCL );
extern void cont_handler( int sig );
extern int count_char( char *string, char c);
extern double cus_log10X(LocalWin *wi, double x);
extern double cus_log10Y( LocalWin *wi, double y);
extern double cus_pow( double x, double p);
extern double cus_powX( LocalWin *wi, double x);
extern double cus_powY( LocalWin *wi, double x);
extern double cus_pow_y_pow_xX( LocalWin *wi,  double y, double x);
extern double cus_pow_y_pow_xY( LocalWin *wi, double y, double x);
extern double cus_sqr( double x);
extern double cus_sqrt( double x);
extern xtb_hret del_func( Window win, int bval, xtb_data info);
extern int display_ascanf_statbins( LocalWin *wi );
extern int display_ascanf_variables( Window win, Boolean show_info, Boolean show_dollars, char **search );
extern int do_hardcopy(char *prog, void *info, int (*init_fun)(), char *dev_spec, char *file_or_dev, int append,
		double *maxheight, double *maxwidth, int orientation,
		char *ti_fam, double ti_size, char *le_fam, double le_size, char *la_fam, double la_size, char *ax_fam, double ax_size
);
extern double gammln(double xx);
extern double get_radix(char *arg, double *radix, char *radixVal );
extern xtb_hret hcpy_func( Window win, int bval, xtb_data info);
extern char *help_fnc_selected;
extern int help_fnc(LocalWin *wi, Boolean refresh, Boolean showMiss );
extern int increment_height( LocalWin *wi, char *text, char letter);
extern xtb_hret info_func( Window win, int bval, xtb_data info);
extern char *ascanf_string( char *string, int *take_usage );
extern int interactive_param_now( LocalWin *wi, char *expr, int tilen, int maxlen, char *message, double *x, int modal, int verbose );
extern int interactive_param_now_xwin( Window win, char *expr, int tilen, int maxlen, char *message, double *x,
	int modal, double verbose, int AllWin
);
extern void kill_caller_process();
extern int fascanf_eval( int *n, char *expression, double *vals, double *data, int *column, int compile );
extern int new_param_now( char *exprbuf, double *val, int N);
extern int new_param_now_allwin( char *exprbuf, double *val, int N);
extern int new_process_BoxFilter_process( LocalWin *theWin_Info, int which );
extern int new_process_Cross_fromwin_process( LocalWin *theWin_Info );
extern int new_process_data_after( LocalWin *theWin_Info );
extern int new_process_data_before( LocalWin *theWin_Info );
extern int new_process_data_finish( LocalWin *theWin_Info );
extern int new_process_data_init( LocalWin *theWin_Info );
extern int new_process_data_process( LocalWin *theWin_Info );
extern int new_process_draw_after( LocalWin *theWin_Info );
extern int new_process_draw_before( LocalWin *theWin_Info );
extern int new_process_dump_after( LocalWin *theWin_Info );
extern int new_process_dump_before( LocalWin *theWin_Info );
extern int new_process_enter_raw_after( LocalWin *theWin_Info );
extern int new_process_leave_raw_after( LocalWin *theWin_Info );
extern int new_process_set_process( LocalWin *theWin_Info, DataSet *this_set );
extern int new_transform_x_process( LocalWin *new_info);
extern int new_transform_y_process( LocalWin *new_info);
extern double nlog10X( LocalWin *wi, double x);
extern double nlog10Y( LocalWin *wi, double x);
extern void notify( int sig);
extern double Calculate_SetOverlap( LocalWin *wi, DataSet *set1, DataSet *set2, SimpleStats *O1, SimpleStats *O2, double *weight, int *overlap_type, int all_vectors );
extern double overlap( LocalWin *wi );
extern char *parse_codes( char *T );
extern char *parse_seconds(double seconds, char *buf);
extern char *parse_varname_opcode( char *opcode, int end_mark, char **end_found, int internal_too );
extern int print_string( FILE *fp, char *header, char *wrapstr, char *trailer, char *string );
extern int print_string2( FILE *fp, char *header, char *trailer, char *string, int instring );
extern int print_string_string( FILE *fp, char *header, char *trailer, char *string, int *instring );
extern int sprint_string2( char **target, char *header, char *trailer, char *string, int instring );
extern int sprint_string_string( char **target, char *header, char *trailer, char *string, int *instring );
extern void process_hist(Window win);
extern xtb_hret process_hist_h(Window win, int bval, xtb_data info);
extern int rd_dbl(char *name);
extern int rd_flag(char *name);
extern int rd_font(char *name, char **font_name);
extern int rd_int(char *name);
extern int rd_pix(char *name);
extern int rd_str(char *name);
extern int realloc_LocalWin_data( LocalWin *new_info, int n);
extern int realloc_WinDiscard( LocalWin *wi, int n );
extern xtb_hret settings_func( Window win, int bval, xtb_data info);
extern int sort_ascanf_Functions( ascanf_Function *a, ascanf_Function *b );
extern double sqr(double x);
extern xtb_hret ssht_func( Window win, int bval, xtb_data info);
extern char *stralloccpy( char **dest, char *src, int minlen );
extern char *strcpalloc( char **dest, int *alloclen, const char *src );
extern int stricmp( const char *a,  const char *b);
extern char *substitute( char *s, int c, int n);
extern char *tildeExpand(char *out, const char *in);
extern char *time_stamp( FILE *fp, char *name, char *buf, int verbose, char *postfix);
extern char *today();
extern double wilog10( LocalWin *wi, double x );
extern int Whereis( char *Prog_Name, char *us, int len );
/* extern xtb_hret xtb_LocalWin_h( XEvent *evt, LocalWin *wi );	*/
extern xtb_hret xtb_LocalWin_h( XEvent *evt, xtb_registry_info *info );

/* Functions in xgPen.c:	*/
extern void CollectPenPosStats( LocalWin *wi, XGPenPosition *pos );
extern int Outside_PenText( LocalWin *wi, int *redo );
extern void DrawPen( LocalWin *wi, XGPen *Pen );
extern void DrawPenSegments( LocalWin *wi, XGPen *Pen, int nondrawn_s, int linestyle, double lineWidth, int pixvalue, Pixel pixelValue );
extern int FlushPenSegments( LocalWin *wi, XGPen *Pen, int *nondrawn_s,
	int *linestyle, double *lineWidth, int *pixvalue, Pixel *pixelValue,
	XGPenPosition *pos, XSegment **xseg, int CheckFirst );
extern int AddPenPosition( XGPen *Pen, XGPenPosition *vals, unsigned int mask );
extern XGPenPosition *Alloc_PenPositions( XGPen *Pen, int N );
extern XGPen *CheckPens( LocalWin *wi, int pen_nr );
extern int PenReset( XGPen *pen, int dealloc );
extern int xgPenMoveTo( int argc, double arg0, double arg1, double arg2, double *result );
extern int xgPenLineTo( int argc, int checkArray, double arg0, double arg1, double arg2, double *result );

/* generic function(s) in dymod.c : */
extern void *XG_dlopen( char **name, int flags, char **error_mesg );

#ifndef MOD
#	define MOD(v,d)  ( (v) >= 0  ? (v) % (d) : (d) - ( (-(v))%(d) ) )
#endif
#ifndef ABS
#	define ABS(v)    ((v)<0?-(v):(v))
#endif
#ifndef FABS
#	define FABS(v)   (((v)<0.0)?-(v):(v))
#endif
#ifndef SIGN
#	define SIGN(x)		(((x)<0)?-1:1)
#endif
#define ODD(x)		((x) & 1)

#ifndef SQR
#	define SQR(x)	((x)?((double)(x)*(x)):0.0)
#endif

#ifndef MAX
#	define MAX(a,b)                (((a)>(b))?(a):(b))
#endif

#ifndef MIN
#	define MIN(a,b)                (((a)<(b))?(a):(b))
#endif

#if defined(__GNUC__)
/* IMIN: version of MIN for integers that won't evaluate its arguments more than once (as a macro would).
 \ With gcc, we can do that with an inline function, removing the need for static cache variables.
 \ There are probably other compilers that honour the inline keyword....
 */
#	undef IMIN
inline static int IMIN(int m, int n)
{
	return( MIN(m,n) );
}
#else
#	ifndef IMIN
static int mi1,mi2;
#		define IMIN(m,n)	(mi1=(m),mi2=(n),(mi1<mi2)? mi1 : mi2)
#	endif
#endif

#define MAXp(a,b)				(((a)&&(a)>(b))?(a):(b))	/* maximal non-zero	*/
#define MINp(a,b)				(((a)&&(a)<(b))?(a):(b))	/* minimal non-zero	*/
#ifndef SWAP
#	define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}
#endif

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif
#ifndef CLIP_EXPR
#	define CLIP_EXPR(var,expr,low,high)	if(((var)=(expr))<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif

#if defined(linux)
		/* In principle, a sincos() routine is (much) faster than 2 separate calls to sin() and cos().
		 \ So use it when the hardware and software provide it!
		 \ It is available on LinuxX86 (in the FPU, so also under other Unixes),
		 \ on SGIs, it seems not to be there, but it can be had via Performer (pfSinCos())
		 \ On other machines, I don't know.
		 */
#	define SinCos(a,s,c)	sincos( ((a)+Gonio_Base_Offset)/Units_per_Radian,(s),(c))
#	define NATIVE_SINCOS	1

#elif defined(sgi) && defined(__PR_H__)

#	define SinCos(a,s,c)	{float fs, fc; pfSinCos( (float)((a)+Gonio_Base_Offset)/Units_per_Radian,&fs,&fc); *(s)=fs,*(c)=fc;}
#	define sincos(a,s,c)	{float fs, fc; pfSinCos( (float)(a),&fs,&fc); *(s)=fs,*(c)=fc;}
#	define NATIVE_SINCOS	2

#elif defined(__APPLE__) 
#	if !defined(NATIVE_SINCOS)
		static inline void SinCos(double a, double *s, double *c)
		{ const int nn = 1;
		  const double aa = (a + Gonio_Base_Offset) / Units_per_Radian;
// 		  extern void vvsincos( double *, double *, const double *, const int *);
			vvsincos( s, c, &aa, &nn );
		}

		static inline void sincos(double a, double *s, double *c)
		{ const int nn = 1;
			vvsincos( s, c, &a, &nn );
		}
#		define NATIVE_SINCOS	1
#	endif
#else

#	define SinCos(a,s,c)	*(s)=Sin((a)),*(c)=Cos((a))
#	define sincos(a,s,c)	*(s)=sin((a)),*(c)=cos((a))
#	define NATIVE_SINCOS	0
#endif

#ifndef degrees
#	define degrees(a)			((a)*57.2957795130823229)
#endif
#ifndef radians
#	define radians(a)			((a)/57.2957795130823229)
#endif

#define sindeg(a)			sin(radians(a))
#define cosdeg(a)			cos(radians(a))
#define tandeg(a)			tan(radians(a))

// #if defined(__MACH__) || defined(__CYGWIN__)
#	ifndef MAXINT
#		define MAXINT INT_MAX
#	endif
#	ifndef MAXSHORT
#		define MAXSHORT SHRT_MAX
#	endif
// #endif

// #ifdef __cplusplus
// }
// #endif


#endif
