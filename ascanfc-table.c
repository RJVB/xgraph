#define _ASC_TABLE_C

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <math.h>

#include "dymod.h"

#include "cpu.h"

#include "copyright.h"

#include "ascanf.h"
#include "compiled_ascanf.h"
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"
#include "XGPen.h"

IDENTIFY( "Builtin ascanfc function table" );

extern double IntensityRGBValues[7];
extern double printf_ValuesPrinted[2];
extern double LastActionDetails[5];
extern double ascanf_PointerPos[2], ascanf_ArgList[1], ascanf_elapsed_values[4];
extern double HandlerParameters[8];

#include "ascanfc-table.h"

ascanf_Function vars_ascanf_Functions[]= {
	{ "DCL", ascanf_DeclareVariable, AMAXARGS, NOT_EOF_OR_RETURN, "DCL[name] or DCL[name,expr]: declare a label or (modify) a variable."},
	{ "$CurrentSet", ascanf_Variable, 2, _ascanf_variable, "$CurrentSet: the current setNumber (variable)"},
	{ "$setNumber", ascanf_Variable, 2, _ascanf_variable, "$setNumber: the total number of sets (variable)"},
	{ "$numPoints", ascanf_Variable, 2, _ascanf_variable, "$numPoints: the number of points in the current set (variable)"},
	{ "$counter", ascanf_Variable, 2, _ascanf_variable, "$counter: the current pointnumber (variable)"},
	{ "$Counter", ascanf_Variable, 2, _ascanf_variable, "$Counter: the current pointnumber including rejected points (variable)"},
	{ "$loop", ascanf_Variable, 2, _ascanf_variable, "$loop: the current loopcounter (variable)"},
	{ "$DATA{0}", ascanf_Variable, 2, _ascanf_variable,
		"$DATA{0} or $DATA{0}[expr]: this is a variable that is a shortcut for the initial value of DATA[0]\n"
		" (it is not updated to reflect the current (processed) value of the X co-ordinate",
		0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$DATA{1}", ascanf_Variable, 2, _ascanf_variable,
		"$DATA{1} or $DATA{1}[expr]: this is a variable that is a shortcut for the initial value of DATA[1]", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$DATA{2}", ascanf_Variable, 2, _ascanf_variable,
		"$DATA{2} or $DATA{2}[expr]: this is a variable that is a shortcut for the initial value of DATA[2]", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$DATA{3}", ascanf_Variable, 2, _ascanf_variable,
		"$DATA{3} or $DATA{3}[expr]: this is a variable that is a shortcut for the initial value of DATA[3]", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$index", ascanf_Variable, 2, _ascanf_variable, "$index (of expression-element)" },
	{ "$self", ascanf_Variable, 2, _ascanf_variable, "$self (initial value = parameter range variable)" },
	{ "$current", ascanf_Variable, 2, _ascanf_variable, "$current: current value" },
	{ "$curve_len-with-discarded", ascanf_Variable, 2, _ascanf_variable,
		"$curve_len-with-discarded: 0(def): curve_len doesn't included discarded points\n"
		"                           1: includes discard[1]'ed points\n"
		"                           2: includes all points ",
		1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$disable_SET_PROCESS", ascanf_Variable, 2, _ascanf_variable,
		"$disable_SET_PROCESS don't evaluate the individual *SET_PROCESS*es ", 1, 0, 0, 0, 0, 0.0
	},
	{ "$SET_PROCESS_last", ascanf_Variable, 2, _ascanf_variable,
		"$SET_PROCESS_last when to evaluate the individual *SET_PROCESS*es:\n"
		" <0: after the *DATA_BEFORE* and before the *DATA_PROCESS* statement\n"
		"  0: after the *DATA_PROCESS* and before the *DATA_AFTER* statement\n"
		" >0: after the *DATA_AFTER* statement and before the *DATA_FINISH* statement\n",
		1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$SAS_converts_angle", ascanf_Variable, 2, _ascanf_variable,
		"$SAS_converts_angle: whether or not the angular statistics (SAS_Add) do conversion of angles-to-be-added\n"
		" onto a <-radix/2,radix/2] interval ", 1, 0, 0, 0, 0, 0,
		  /* This one defaults to 0, which is NOT the original behaviour!!! */
		0.0
	},
	{ "$SAS_converts_result", ascanf_Variable, 2, _ascanf_variable,
		"$SAS_converts_result: whether or not the angular statistics (SAS_Mean) do conversion of the resulting average angle\n"
		" onto a <-radix/2,radix/2] interval ", 1, 0, 0, 0, 0, 0,
		  /* This one defaults to 1 */
		1.0
	},
	{ "$SAS_exact", ascanf_Variable, 2, _ascanf_variable,
		"$SAS_exact: whether or not the angular statistics (SAS_Mean) are exact, using sines and cosines, and storing all samples",
		1, 0, 0, 0, 0, 0,
		  /* This one defaults to 0 */
		0.0
	},
	{ "$SS_exact", ascanf_Variable, 2, _ascanf_variable,
		"$SS_exact: whether or not the statistics (SS_Mean) stores all samples",
		1, 0, 0, 0, 0, 0,
		  /* This one defaults to 0 */
		0.0
	},
	{ "$SAS_Ignore_NaN", ascanf_Variable, 2, _ascanf_variable,
		"$SAS_Ignore_NaN: whether or not the angular statistics (SAS_Mean) ignore NaN \"values\" being added\n"
		" When set to 2, ignore Inf values also.\n"
		, 1, 0, 0, 0, 0, 0,
		  /* This one defaults to 0 */
		0.0
	},
	{ "$SS_Ignore_NaN", ascanf_Variable, 2, _ascanf_variable,
		"$SS_Ignore_NaN: whether or not the angular statistics (SS_Mean) ignore NaN \"values\" being added\n"
		" When set to 2, ignore Inf values also.\n"
		, 1, 0, 0, 0, 0, 0,
		  /* This one defaults to 0 */
		0.0
	},
	{ "$SS_Empty_Value", ascanf_Variable, 2, _ascanf_variable,
		"$SS_Empty_Value: the value SS_Mean and SAS_Mean, SS_Min, SS_max, etc. return when the bin is empty."
		, 1, 0, 0, 0, 0, 0,
		  /* This one defaults to 0 */
		0.0
	},

	{ "$verbose", ascanf_Variable, 2, _ascanf_variable,
		"controls verbose output, like the verbose[] function or -fascanf_verbose CLA ", 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$popup_verbose", ascanf_Variable, 2, _ascanf_variable,
		"makes popup[expr] behave like popup[verbose[expr]] ", 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$compile_verbose", ascanf_Variable, 2, _ascanf_variable,
		"Switches to verbose mode while compiling. For values >1, messages are given during destruction, too.\n"
		" Also forces verbose mode while compiling inside an IDict[] block.\n"
		" Setting this to a negative value has the (surprise) reverse effect of suppressing all(most) all\n"
		" compilation feedback, including that of verbose[] blocks being compiled.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$IDict", ascanf_Variable, 2, _ascanf_variable,
		"controls global access to the internal dictionary, like the IDict[] function", 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$ReDo_DRAW_BEFORE", ascanf_Variable, 2, _ascanf_variable,
		"set if the *DRAW_BEFORE* statements contain necessary resets to be executed before doing\n"
		" any other processing - e.g. *DATA_PROCESS* statements are executed more than once\n"
		" under certain conditions (re-initialisation) ", 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$Really_DRAW_AFTER", ascanf_Variable, 2, _ascanf_variable,
		"set if the *DRAW_AFTER* statements should be executed as the really really last\n"
		" thing in a window redraw\n", 1, 0, 0, 0, 0, 0, 0.0
	},

	{ "$SS_T_Value", ascanf_Variable, 2, _ascanf_variable, "$SS_T_Value: T value of the last executed T test"},
	{ "$SS_F_Value", ascanf_Variable, 2, _ascanf_variable, "$SS_F_Value: F value of the last executed F test"},

	{ "$IntensityColours", ascanf_Variable, 2, _ascanf_variable, "$IntensityColours: the number of (possible) Intensity Colours"},
	{ "$IntensityRGB", ascanf_Variable, 7, _ascanf_array,
		"$IntensityRGB[7]: the RGB intensities of the last colour allocation or query (GetRGB,GetIntensityRGB).\n"
		" The 4th element is the pixel value; the last 3 the exact (requested) RGB values, or -1\n",
		0, 0, 0, 0, 0, 0, 0.0, &IntensityRGBValues[0], NULL, NULL, NULL, NULL, 0, 0, 7 },

	{ "$ActiveWinWidth", ascanf_Variable, 2, _ascanf_variable, "$ActiveWinWidth: the current effective (plotting) width of the active window" },
	{ "$ActiveWinHeight", ascanf_Variable, 2, _ascanf_variable, "$ActiveWinHeight: the current effective (plotting) height of the active window" },
	{ "$ReCalculate", ascanf_Variable, 2, _ascanf_variable,
		"$ReCalculate: ignore an eventual QuickMode setting (use_transformed) of the first window to be redrawn.\n"
		" The redraw resets this variable to 0!",
		1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$VariableInit", ascanf_Variable, 2, _ascanf_variable,
		"$VariableInit: how a variable is initialised if declared by DCL[foo] (i.e. without assigning a value);\n"
		" 0: initialise at 0\n"
		" >0: initialise with the number of so declared variables\n"
		" <0: initialise with a random number [0,1> (useful for checking dependence on initial variable values ;-))"
		" NaN: initialise with NaN\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$ExitOnError", ascanf_Variable, 2, _ascanf_variable,
		"$ExitOnError: >0: stop reading the current file if ascanf (and some other) errors are encountered\n"
		" <0: stop reading upon errors during compilation.\n",
		1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$UseConstantsLists", ascanf_Variable, 2, _ascanf_variable,
		"Activates some compiler optimisation which reduces overhead on expressions\n"
		" with constants/variables of not more than 2 levels deep. ",
		1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$AllowSimpleArrayOps", ascanf_Variable, 2, _ascanf_variable,
		"whether or not a number of \"simple\" functions check if they are applied to two or more arrays.\n"
		" Currently, these include: add[], sub[], relsub[], mul[], div[], len[], pow[], pow*[], =[] and !=[]\n"
		" Example: mul[&foo,&bar,&boo]: multiply bar with boo storing in foo;\n"
		"          mul[&foo,&bar,n]: multiply all in bar with constant n, storing in foo\n"
		"          mul[&foo,n]: multiply all in foo with constant n, storing in foo\n"
		"          add[&foo], mul[&foo]: sum and product over foo (only these!)\n"
		" When $AllowSimpleArrayObs==1, these functions return the sum of the destination array (except for the last form)\n"
		" When $AllowSimpleArrayObs==2, the pointer to the destination array is returned instead.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$AlwaysUpdateAutoArrays", ascanf_Variable, 2, _ascanf_variable,
		"Whether or not automatic arrays with variables or functions (e.g. {v1,v2,add[v1,v2]} are always\n"
		" updated to reflect the current referenced values, whenever an argument passed to a function is\n"
		" checked to be a generic or an array pointer. If not set, updating takes place only when the array\n"
		" is referenced directly or when it is passed to nDindex[].\n",
		1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$CurrentScreenWidth", ascanf_Variable, 2, _ascanf_variable,
		"$CurrentScreenWidth: this is a variable that contains the current screen's width", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$CurrentScreenHeight", ascanf_Variable, 2, _ascanf_variable,
		"$CurrentScreenHeight: this is a variable that contains the current screen's height", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$ReadBufVal", ascanf_Variable, 2, _ascanf_variable,
		"$ReadBufVal: this variable can contain the current value in the read buffer\n"
		" or NaN when it does not contain a number. The read buffer is a global buffer\n"
		" that can be \"filled\" by typing characters into a graph window. It can be reset\n"
		" with the Escape key and the ClearReadBuffer[] function\n", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$PointerPos", ascanf_Variable, 2, _ascanf_array,
		"$PointerPos[2]: the pointer's X,Y co-ordinates as last updated by a call to QueryPointer[]",
		0, 0, 0, 0, 0, 0, 0.0, &ascanf_PointerPos[0], NULL, NULL, NULL, NULL, 0, 0, 2
	},
	{ "$Sets-Reordered", ascanf_Variable, 2, _ascanf_variable,
		"$Sets-Reordered: set, for each window, when a reordering of sets has taken place,\n"
		" so that arrays storing set-specific information can be recalculated.\n"
		" This variable is reset to 0 after a full window redraw\n", 0, 0, 0, 0, 0, 0, 0.0
	},
	{ "$SyncedAnimation", ascanf_Variable, 2, _ascanf_variable,
		"$SyncedAnimation: sync to the X server after all to-be-animated windows have\n"
		" been redrawn once, and before continuing with the next redraw.\n"
		" This way, each window does not have to perform the sync itself (e.g. in a *DRAW_AFTER*)\n",
		1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$AllowSomeCompilingInitialisations", ascanf_Variable, 2, _ascanf_variable,
		"$AllowSomeCompilingInitialisations: allow initialisation of variables while they are being\n"
		" declared while the expression is being compiled (e.g. compile[ DCL[theVar,<exp>] ]). The\n"
		" compiler evaluates <exp>, but many functions don't return the expected values while compiling\n"
		" (e.g. to avoid side-effects). Constant expressions, however, (numbers, other variables) should\n"
		" be OK. A warning will be issued when <exp> is not a list consisting of only constants (when\n"
		" $UseConstantsLists is true).\n",
		1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$", ascanf_Variable, 2, _ascanf_array,
		"$[<n>]: the argumentlist passed to the current frame\n"
		" For use in procedures, elsewhere its values won't make much sense.\n"
		" In procedures called as accesshandler, this array contains the arguments passed\n"
		" to the entity that was accessed. (Also called $ArgList.)\n",
		0, 0, 0, 0, 0, 0, 0.0, &ascanf_ArgList[0], NULL, NULL, NULL, NULL, 0, 0, 0
	},
	{ "$case", ascanf_Variable, 2, _ascanf_variable, "$case: the currently active/appropriate case of a switch[]"},
	{ "$AllowGammaCorrection", ascanf_Variable, 2, _ascanf_variable,
		"$AllowGammaCorrection: determines the behaviour of the colour allocation. Normally, when this variable is set,\n"
		" XGraph allocates colour specified with RGB triplets instead of with names through the X rgbi:ri/gi/bi intensity\n"
		" specification. This way, the gamma correction currently defined for the display is applied before the colour is\n"
		" allocated. This will probably guarantee that the visual effect is closest to what is intended, but it can\n"
		" introduce colour deviations, especially visible when a grayscale is intended.\n"
		" Therefore, when this variable is unset, XGraph will use the rgb:rrrr/gggg/bbbb specification, which does not apply\n"
		" the gamma correction.\n"
		" Also see the manpage section on colours.\n",
		1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$gsTextWidthBatch", ascanf_Variable, 2, _ascanf_variable,
		"$gsTextWidthBatch: stores/sets the value of the -gs_twidth_batch option. When set,\n"
		" the PostScript width of all currently relevant strings is determined by printing\n"
		" to /dev/null, each time the hardcopy dialog is invoked. For the cost of printing once\n"
		" more, this can dramatically increase the speed of an \"effective\" printout.\n",
		1, 0, 0, 0, 0, 0, 0.0
	},
	  /* The following *must* not have a usage string by default!	*/
	{ "$VariableLabel", ascanf_Variable, 2, _ascanf_variable, NULL, 0, 0, 0, 0, 0, 0, 0.0, },
	  /* The following *must* not have a usage string by default!	*/
	{ "$ValuePrintFormat", ascanf_Variable, 2, _ascanf_variable, NULL, 0, 0, 0, 0, 0, 0, 0.0, },
	{ "$ValuesPrinted", ascanf_Variable, 2, _ascanf_array,
		"$ValuesPrinted[2]: values actually and/or theoretically printed by printd[].\n"
		"$ValuesPrinted[0]=actually printed, $ValuesPrinted[1]=number of %s fields in the format string.",
		0, 0, 0, 0, 0, 0, 0.0, &printf_ValuesPrinted[0], NULL, NULL, NULL, NULL, 0, 0, 2
	},
	  /* When updating LastActionDetails, don't forget the ->N field (the last before the closing brace)!	*/
	{ "$PrintNaNCode", ascanf_Variable, 2, _ascanf_variable, 
		"$PrintNaNCode: whether or not to print the code identifying the type of a NaN when printing one"
		, 0, 0, 0, 0, 0, 0, 0.0, },
	{ "$LastActionDetails", ascanf_Variable, 5, _ascanf_array,
		"$LastActionDetails[5]: Some details about the last interactive action:\n"
		" Element [0] contains the action mask:\n"
		"   0= no data available;\n"
		"   1= data displacement; \\d\\X, \\d\\Y, <Npoints>, <N-whole-sets>\n"
		"   1.1= undo data displacement; -\\d\\X, -\\d\\Y, <Npoints>, <N-whole-sets>\n"
		,0, 0, 0, 0, 0, 0, 0.0, &LastActionDetails[0], NULL, NULL, NULL, NULL, 0, 0, 5
	},
	  /* $XGOutput *must* not have a usage string by default!	*/
	{ "$XGOutput", ascanf_Variable, 2, _ascanf_variable, NULL, },
	{ "$elapsed", ascanf_Variable, 2, _ascanf_array,
		"$elapsed[4]: timing values determined by the last call to the elapsed[] or time[] functions.\n"
		"$elapsed[0]=real time, $elapsed[1,2]=user,system time (should be about the same if no other processes run).\n"
		"$elapsed[3]=estimate of the number of (floating point) operations: this must be set by the called expression!\n"
		, 0, 0, 0, 0, 0, 0, 0.0, &ascanf_elapsed_values[0], NULL, NULL, NULL, NULL, 0, 0, 4
	},
	{ "$Dprint-file", ascanf_Variable, 2, _ascanf_variable, NULL,
		1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$DataWin-before-Processing", ascanf_Variable, 2, _ascanf_variable,
		"$DataWin-before-Processing: when to apply an active DataWindow (see ActiveWinDataWin[]):\n"
		" True: before any processes/transformations are applied\n"
		" False: after any processes/transformations have been performed\n"
		" (Should be false in case the calculation results can depend on points outside the window...)\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$AllowProcedureLocals", ascanf_Variable, 2, _ascanf_variable,
		"whether or internal (IDict) variables defined in a procedure are local to that procedure.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$Find_Point-exhaustive", ascanf_Variable, 2, _ascanf_variable,
		"whether the internal Find_Point heeds the precision specifiers.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$AllowArrayExpansion", ascanf_Variable, 2, _ascanf_variable,
		"whether or not arrays are auto-expanded as necessary (storing a value in a non-existent cell)\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},

/* end of variablelist	*/

	{ "nop", ascanf_nop, AMAXARGS, NOT_EOF_OR_RETURN, "nop or nop[...]: does nothing" },
#if 0
	{ "ADR", ascanf_Address, 1, NOT_EOF,
		"ADR[name]: dynamic equivalent to &name ."
	},
#endif
	{ "register-Variable-Names", ascanf_registerVarNames, 1, NOT_EOF_OR_RETURN,
		"register-Variable-Names: returns whether or not a registry of variable names is being kept\n"
		" register-Variable-Names[yesno]: (de)activates that maintaining of a variable names registry;\n"
		" the old setting is returned.\n"
		" This feature is used by certain DyMods to access variables with a special name; the IO_Import modules\n"
		" require it to be set to check for pre-existing definitions of the variables they export, and retrieve their values.\n"
	},
	{ "ParentsName", ascanf_ParentsName, 0, NOT_EOF,
		"ParentsName: returns the name of the calling (parent) function/procedure when this is known. Uses an internal string buffer."
	},
	{ "GrandParentsName", ascanf_GrandParentsName, 0, NOT_EOF,
		"GrandParentsName: returns the name of the parent's parent, when this is known. Uses an internal string buffer."
	},
	{ "ProcedureName", ascanf_procedureName, 1, NOT_EOF,
		"ProcedureName: returns the name of the procedure that is the nearest on the calling stack.\n"
		" ProcedureName[level]: returns procedure <level> (0== nearest) on the calling stack.\n"
		" ProcedureName[&procedure]: returns the name of the procedure pointed to.\n"
		" Cycles through 10 internal string buffers\n"
	},
	{ "ProcedureCode", ascanf_procedureCode, 1, NOT_EOF,
		"ProcedureCode: returns the code of the procedure that is the nearest on the calling stack.\n"
		" ProcedureCode[level]: returns procedure <level> (0== nearest) on the calling stack.\n"
		" ProcedureCode[&procedure]: returns the code of the procedure pointed to.\n"
		" Cycles through 10 internal string buffers\n"
		" Returns 0 if no code is available (for whatever reason).\n"
	},
	{ "Declare", ascanf_DeclareVariable, AMAXARGS, NOT_EOF_OR_RETURN, "Declare[name] or Declare[name,expr]: declare a label or (modify) a variable."},
	{ "DEPROC-noEval", ascanf_DeclareProcedure, AMAXARGS, NOT_EOF_OR_RETURN,
		"DEPROC-noEval[name,expr[,description]]: declare a procedure without evaluating it.", 0
	},
	{ "DEPROC", ascanf_DeclareProcedure, AMAXARGS, NOT_EOF_OR_RETURN,
		"DEPROC[name,expr[,description]]: (declare and) evaluate a procedure.", 1
	},
	{ "DEPROC*", ascanf_DeclareProcedure, AMAXARGS, NOT_EOF_OR_RETURN,
		"DEPROC*[name,expr[,description]]: (declare and) immediately evaluate a procedure.", 1
	},
	{ "ASKPROC", ascanf_DeclareProcedure, AMAXARGS, NOT_EOF_OR_RETURN,
		"ASKPROC[name,expr[,description]]: (declare and) evaluate a procedure that posts\n"
		" its code in a dialog for the user to edit/confirm or cancel.\n"
		" The accepted or specified code is (re)compiled and replaces the original.\n"
	},
	{ "EditPROC", ascanf_EditProcedure, AMAXARGS, NOT_EOF_OR_RETURN,
		"EditPROC[name]: calls up an editor to edit the procedure `name'. The editor is taken from the\n"
		" env.variable XG_EDITOR; if absent, `xterm -e vi' is used. After the editor quits (or forks!),\n"
		" the resulting code is read back in.\n"
	},
	{ "Delete", ascanf_DeleteVariable, 1, NOT_EOF, "Delete[name]: delete a label or variable.\n"
		" Delete[$AllDefined] deletes all variables/procedures not starting with a '$'\n"
		" Delete[$UnUsedVars] deletes all unused variables\n"
		" Delete[$UnUsed] deletes all unused variables *and* procedures\n"
		" Delete[$InternalUnUsed] deletes all unused internal variables\n"
		" Delete[$Label=<label>] deletes all variables declared with $VariableLabel==<label>\n"
		" Delete[$UnUsedLabel=<label>] deletes all unused variables declared with $VariableLabel==<label>\n"
		" These global actions take place immediately, compiling or not; deletion of a single\n"
		" variable is not done at compile-time, but at the 1st evaluation (as expected)\n"
	},
	{ "QSortSet", ascanf_QSortSet, 2, NOT_EOF,
		"QSortSet[<set_nr>,<compar_proc>]: quick sort the indicated set. <compar_proc> must point to a procedure that\n"
		" accepts/expects 2 arguments in the '$' array. The 2nd argument is the point number (index in the set's\n"
		" data arrays) for which a sorting value is to be returned.\n"
		" To sort for X value, one can simply use a procedure that returns DataVal[$[0],xcol[$[0]],$[1]]\n"
	},
	{ "QSortSet2", ascanf_QSortSet2, 2, NOT_EOF,
		"QSortSet2[<set_nr>,<compar_proc>]: quick sort the indicated set. <compar_proc> must point to a procedure that\n"
		" accepts/expects 3 arguments in the '$' array. These arguments are 1) the set number and 2,3) the point numbers\n"
		" (index in the set's data arrays) for which a sorting comparison is to be returned.\n"
		" To sort for X value, one can simply use a procedure that returns\n"
		" sub[ DataVal[$[0],xcol[$[0]],$[2]], DataVal[$[0],xcol[$[0]], $[1]] ] @\n"
	},
	{ "QSortArray", ascanf_QSortArray, 4, NOT_EOF_OR_RETURN,
		"QSortArray[<targ_p>,<compar_proc>,<val_A>,<val_B>]: quick sort the array pointed to by <targ_p> (see qsort(2)).\n"
		" A pointer to a procedure comparing 2 of targ_p's elements should be provided in <compar_proc>; the 2\n"
		" variables it compares can be passed as pointers in <val_A> and <val_B> (for old-style procedures that don't accept\n"
		" arguments). QSortArray[] returns the number of comparisons made to sort the array.\n"
		" NB: <compar_proc> may also be a pointer to a function that accepts 2 arguments; passing &sub will do an increment sort.\n"
	},
	{ "ArrayFind", ascanf_ArrayFind, 5, NOT_EOF,
		"ArrayFind[&array,value,&low_idx,&high_idx[,tolerance]]: finds the two values in <array> that are adjacent (or equal) to\n"
		" <value>, using bisection. Returns the indices of the adjacent elements in &low_idx and &high_idx, and True upon success,\n"
		" 0 upon failure (in which case the values of idx pointers are *not* changed).\n"
		" If <value> is in array, its index will be returned in high_idx when it is the last element and in\n"
		" low_idx otherwise; the return value is -1 when array[low_idx] is closest to <value> and 1 otherwise (0 on error).\n"
		" The optional tolerance argument is only used in the final (internal) stage: if specified, the lower bound lidx will be\n"
		" accepted when either array[lidx]<=value OR when array[lidx]==value+-tolerance/2.\n"
		" This only works/makes sense for sorted arrays!\n"
	},
	{ "SetArraySize", ascanf_SetArraySize, AMAXARGS, NOT_EOF_OR_RETURN,
		"SetArraySize[<targ_p>,<size>[,v1[,v2[,...vn]]]]: set array pointed to by <targ_p> to new size <size> (-1 for current size),\n"
		" and if additional values are passed, initialise element 0 to <v1>, 1 to <v2>, etc; the last vn is stored in the remaining elements\n"
		" If <targ_p> is a scalar, it is CONVERTED to an array! The reverse is currently not possible.\n"
	},
	{ "CopyArray", ascanf_cp_array, 6, NOT_EOF_OR_RETURN,
		"CopyArray[dest_p,src_p[,dup[,start,end[,nan_handling]]]]: copy from <src_p> to <dest_p>\n"
		" - over MIN(dest_p->N,src_p->N) elements when dup==False or omitted\n"
		" - making dest_p a duplicate of src_p when dup==True.\n"
		" When specified, <start> and <end> define the part of <src_p> to use.\n"
	},
/* 	{ "AutoCropArray", ascanf_AutoCropArray, AMAXARGS, NOT_EOF_OR_RETURN,	*/
/* 		"AutoCropArray[&array1[,&array2,...]]: takes any number of arrays, and \"crops\" constant values from the start\n"	*/
/* 		" and end. Useful to automatically remove the leading and trailing DC components from time-series\n"	*/
/* 	},	*/
	{ "AutoCropArray", ascanf_AutoCropArray, AMAXARGS, NOT_EOF_OR_RETURN,
		"AutoCropArray[&array1,&mask1[,&array2,&mask2,...]]: takes any number of pairs of arrays, and \"crops\" constant\n"
		" values from the start and end. Useful to automatically remove the leading and trailing DC components from time-series\n"
		" The <mask> argument can be an array-pointer sized to the target array's original dimension that will contain\n"
		" that will have ones for the retained values, and zeroes for the cropped elements.\n"
		" Returns the original index of the first 'uncropped' element in the last array (=> 0 means no initial cropping)\n"
	},
	{ "FixArray", ascanf_FixAutoArray, AMAXARGS, NOT_EOF_OR_RETURN,
		"FixArray[&array1[,&array2,...]]: takes any number of arrays, and \"fixates\" their elements. Without this call,\n"
		" any array that has been (auto-)defined as e.g. {v1,v2,add[v1,v2]} will at each access be updated to reflect the\n"
		" current values it references (here v1 and v2). Passing a pointer to the array to FixArray, or defining\n"
		" FixArray[ {v1,v2,add[v1,v2]} ] will prevent this from happening; i.e. the array will reflect the referenced values\n"
		" at the time of declaration (or the last access). FixArray doesn't do any updating itself. It returns either the\n"
		" first argument it was passed, or else the value set by return[].\n"
	},
	{ "Defined?", ascanf_DefinedVariable, 1, NOT_EOF,
		"Defined?[name]: in uncompiled expressions, check whether <name> is defined.\n"
		" Returns -1 for defined internal variables.\n"
	},

	{ "discard", ascanf_Discard, 1, NOT_EOF, "discard[exp]: discard the datapoint (if exp==True; exp==-1 undiscards during runtime)"},
	{ "Discard", ascanf_Discard3, AMAXARGS, NOT_EOF_OR_RETURN,
		"Discard[rexp,dexp,l,h[,dexp2,l2,h2,..]]: discard the datapoint (if dexp==True; dexp==-1 undiscards during runtime)\n"
		"\twhen the current pointnumber is >= <l> and <= <h>. Returns <rexp>."
	},
	{ "DiscardPoint0", ascanf_DiscardedPoint, 3, NOT_EOF_OR_RETURN,
		"DiscardPoint0[set,pnr[,val]]: show the discardstatus of, or discard a datapoint, like discard[] does;\n"
		" if val==True; val==-1 undiscards during runtime.\n Returns the previous setting."
	},
	{ "DiscardPoint", ascanf_Discard2, 3, NOT_EOF_OR_RETURN,
		"DiscardPoint[<setnr>,<pnr>,exp]: like DiscardPoint0, but does a \"hard\" discard\n"
		" when called outside a drawing cycle (e.g. in a *EVAL* statement).\n"
	},
	{ "exit", ascanf_Exit, 1, NOT_EOF, "exit or exit[exp]: skip remainder of the current file (if exp==True)"},
	{ "setNumber", ascanf_SetNumber, 1, NOT_EOF,
		"setNumber or setNumber[<exp>]: the current setNumber or total number of sets\n"
		" If <exp> is a stringpointer, setName and (if relevant) set Title are stored in it.\n"
		" Otherwise, it is interpreted as a Boolean flag to return the total number of sets.\n"
	},
	{ "counter", ascanf_Count, 1, NOT_EOF,
		"counter or counter[exp]: the current pointnumber or the number including rejected points"
	},
	{ "numPoints", ascanf_NumPoints, 2, NOT_EOF_OR_RETURN,
		"numPoints or numPoints[setnr[,N]]: maximum number of points or number of points in set <setnr>\n"
		" It <N> is given, set the number of points in set <setnr> to <N>\n"
	},
#ifdef ADVANCED_STATS
	{ "NumObs", ascanf_NumObs, 2, NOT_EOF_OR_RETURN,
		"NumObs[setnr[,index]]: average number or number of observations per point in set <setnr>"
#else
	{ "NumObs", ascanf_NumObs, 1, NOT_EOF_OR_RETURN,
		"NumObs[setnr]: average number of observations per point in set <setnr>"
#endif
	},
	{ "pointVisible", ascanf_pointVisible, 3, NOT_EOF_OR_RETURN,
		"pointVisible[setnr[,index[,new|&dest_p]]: whether point <index> of set <setnr> is visible (drawn)\n"
		" in the currently active window. When <dest_p> points to an array, all the visibility info for the\n"
		" requested set is copied into it. Otherwise, when this argument is not an array pointer, it is taken\n"
		" as the temporary new visibility value for the given point. This is mainly intended for use in \n"
		" BOX_FILTER applications. The value returned is the visibility value at the time of invocation.\n"
	},

	{ "DATA", ascanf_data, 2, NOT_EOF_OR_RETURN,
		"DATA[n[,expr]]: set DATA[n] to expr, or get DATA[n]: n=0,1,2,3 (x,y,e,l) direct datapoint manipulation\n"
	},
	{ "DataVal", ascanf_DataVal, 4, NOT_EOF_OR_RETURN,
		"DataVal[set,col,index[,newval]]: (new) value #<index> in column <col> of set <set>"
	},
	{ "VAL", ascanf_val, 2, NOT_EOF_OR_RETURN,
		"VAL[n[,m]]: get the corresponding (x,y,e) (n=0,1,2) value in the <m>th set from the current\n"
		"\t VAL[1,10] = yval[add[setNumber,10],counter]\n"
	},
	{ "COLUMN", ascanf_column, 2, NOT_EOF_OR_RETURN,
		"COLUMN[n[,expr]]: set COLUMN[n] to expr, or get COLUMN[n]: n=0..3 direct datapoint order manipulation\n"
	},
	{ "DataChanged", ascanf_DataChanged, 2, NOT_EOF_OR_RETURN,
		"DataChanged: whether there was a change in X, Y or Error between the previous and the current point\n"
		" DataChanged[set,point]: specifies the set and the pointnumber to check for changes with it predecessor\n"
	},
	{ "ColumnSame", ascanf_ColumnSame, 5, NOT_EOF_OR_RETURN,
		"ColumnSame[set,col,n[,pnt[,val]]]: whether set <set>'s column <col> stays at the current or specified value for this and the next <n> points.\n"
		" Returns -1 if n had to be truncated but no change was observed. n< 0 means scan previous points.\n"
		" <pnt> specifies starting point: default is the current point ($Counter)\n"
		" <val> specifies the reference value: default is the current column value\n"
	},
	{ "AddDataPoints", ascanf_AddDataPoints, AMAXARGS, NOT_EOF_OR_RETURN,
		"AddDataPoints[<set#>,v1,v2[,v3,...vn]]: add data to set <set#>\n"
	},
	{ "LinkArray2DataColumn", ascanf_LinkArray2DataColumn, 3, NOT_EOF_OR_RETURN,
		"LinkArray2DataColumn[dest_p,set,col]: link set's <set> column <col> to <dest_p> which\n"
		" should be an array.\n"
	},
	{ "DataColumn2Array", ascanf_DataColumn2Array, 9, NOT_EOF_OR_RETURN,
		"DataColumn2Array[dest_p,set,col[,start[,end_incl[,offset[,pad=0[,padlow[,padhigh]]]]]]]: copy set's <set> column <col> into <dest_p> which\n"
		" must be a pointer to an array. <start> and <end_incl> specify (inclusive) source start and end of copying (end_incl==-1\n"
		" to copy until last); <offset> specifies starting point in <dest_p> which will be expanded/shrunk to the correct size\n"
		" pad,padlow,padhigh: pad begin and/or end according to the SavGolayInit conventions* (pad==-1 is of course undefined).\n"
		" Padding starts at <offset>, so the 1st copied datapoint is at offset+pad; default padding values are those at\n"
		" <start> and <end_incl>.\n"
		" <start>,<end> may also be given as <&Visible>[,getVisible], with Visible an array. This means that only points will\n"
		" be returned that are visible in the currently active window (no active window => no visible points!). The <Visible>\n"
		" array will then contain the indices of those points. If getVisible==0, then the points currently referenced in Visible\n"
		" will be retrieved.\n"
		" * see the fourconv module.\n"
	},
	{ "Array2DataColumn", ascanf_Array2DataColumn, 6, NOT_EOF_OR_RETURN,
		"Array2DataColumn[set,col,src_p[,start[,end_incl[,offset]]]]: copy into set's <set> column <col> from <src_p> which\n"
		" must be a pointer to an array. <start> and <end_incl> specify (inclusive) source start and end of copying (end_incl==-1\n"
		" to copy until last); <offset> specifies starting point in <set> which will be expanded as necessary\n"
	},
	{ "Set2Arrays", ascanf_Set2Arrays, 13, NOT_EOF_OR_RETURN,
		"Set2Arrays[set,raw?,&xvals,&yvals[,&evals,[&lvals,[&Nvals,[,start[,end_incl[,offset[,pad=0[,padlow[,padhigh]]]]]]]]]]:\n"
		" Copy the <set>'s user-defined columns (X,Y,E,Length and/or N) into the designated arrays, some of which are\n"
		" optional. <start> and <end_incl> specify (inclusive) source start and end of copying (end_incl==-1\n"
		" to copy until last); <offset> specifies the starting point in the target arrays which will be expanded/shrunk\n"
		" to the correct size.\n"
		" pad,padlow,padhigh: pad begin and/or end according to the SavGolayInit conventions (pad==-1 is of course undefined).\n"
		" Padding starts at <offset>, so the 1st copied datapoint is at offset+pad; default padding values are those at\n"
		" <start> and <end_incl>.\n"
		" If <raw?> is True, uses the raw, unprocessed values; otherwise, processed values are used.\n"
	},

	{ "NewSet", ascanf_NewSet, 3, NOT_EOF_OR_RETURN, "NewSet[[points[,columns[,link2]]]]: \n"
		" add a new set, optionally specifying number of points & columns\n"
		" When link2 is defined and >=0, the new set points (links) to the specified set's data;\n"
		" in this case, the other arguments are ignored.\n"
	},
	{ "DestroySet", ascanf_DestroySet, 1, NOT_EOF_OR_RETURN,
		"DestroySet[set]: destroy a set. Returns 1 if a set was actually destroyed."
	},

	{ "ncols", ascanf_ncols, 2, NOT_EOF, "ncols[[n[,new]]]: the number of columns of the current [nth] dataset; n=-1 returns max columns" },
	{ "xcol", ascanf_xcol, 4, NOT_EOF_OR_RETURN,
		"xcol[[n[,new-xcol[,partial,withcase]]]: the (new) number of the X column of the current [nth] dataset; n<0 for all sets\n" 
		" <new-xcol> can be a number, or a string, in which case it will be looked up in either the set's or the window's\n"
		" ColumnLabel database, using <partial> matching and/or case-insensitive string matching.\n"
	},
	{ "ycol", ascanf_ycol, 4, NOT_EOF_OR_RETURN,
		"ycol[[n[,new-ycol[,partial,withcase]]]: the (new) number of the Y column of the current [nth] dataset; n<0 for all sets\n"
		" see the help for the xcol[] function.\n"
	},
	{ "ecol", ascanf_ecol, 4, NOT_EOF_OR_RETURN,
		"ecol[[n[,new-ecol[,partial,withcase]]]: the (new) number of the E column of the current [nth] dataset; n<0 for all sets\n"
		" see the help for the xcol[] function.\n"
	},
	{ "lcol", ascanf_lcol, 4, NOT_EOF_OR_RETURN,
		"lcol[[n[,new-lcol[,partial,withcase]]]: the (new) number of the vectorLength column of the current [nth] dataset; n<0 for all sets\n"
		" see the help for the xcol[] function.\n"
	},
	{ "Ncol", ascanf_Ncol, 4, NOT_EOF_OR_RETURN,
		"Ncol[[n[,new-Ncol[,partial,withcase]]]: the (new) number of the N column of the current [nth] dataset; n<0 for all sets\n"
		" see the help for the xcol[] function.\n"
	},
	{ "LabelledColumn", ascanf_LabelledColumn, 4, NOT_EOF_OR_RETURN,
		"LabelledColumn[n[,pattern[,partial,withcase]]: return the n'th dataset's column matching <pattern>\n" 
	},
	{ "ProcessSet", ascanf_ProcessSet, 2, NOT_EOF_OR_RETURN,
		"ProcessSet[<setnr>[,<pnt_nr>]]: perform the DRAW_BEFORE, DATA_xxxx and DRAW_AFTER processing of the currently\n"
		" active window. When pnt_nr is missing or equals -1, do all points of the specified set(s).\n"
		" Updates the curve_len and possibly tr_curve_len data.\n"
		" When setnr==-1, do all sets. NB: calling this routine in QuickMode will likely cause a messed\n"
		" up display after the next redraw!\n"
	},
	{ "BoxFilter", ascanf_BoxFilter, 7, NOT_EOF_OR_RETURN,
		"BoxFilter[`fname, lowX, lowY, highX, highY[, setNR[, pntNR]]]: apply a BoxFilter file.\n"
		" When fname==0, a filename is requested interactively as when invoked through the GUI.\n"
		" It is possible to specify a set on which to apply the filtering exclusively, possibly\n"
		" limited to a single specified point number (applying to a full, single set is only possible\n"
		" through this function, not through the GUI).\n"
	},
	{ "tr_xval", ascanf_xvec, 2, NOT_EOF, "tr_xval[setNumber,index]" },
	{ "tr_yval", ascanf_yvec, 2, NOT_EOF, "tr_yval[setNumber,index]" },
	{ "tr_error", ascanf_errvec, 2, NOT_EOF, "tr_error[setNumber,index]" },
	{ "tr_lval", ascanf_lvec, 2, NOT_EOF, "tr_lval[setNumber,index]" },
	{ "xval", ascanf_xval, 2, NOT_EOF, "xval[setNumber,index]" },
	{ "yval", ascanf_yval, 2, NOT_EOF, "yval[setNumber,index]" },
	{ "error", ascanf_error, 2, NOT_EOF, "error[setNumber,index]" },
	{ "lval", ascanf_lval, 2, NOT_EOF, "lval[setNumber,index]" },
	{ "Nval", ascanf_Nval, 2, NOT_EOF, "Nval[setNumber,index]" },
	{ "tr_xmin", ascanf_tr_xmin, 1, NOT_EOF, "tr_xmin[setNumber]" },
	{ "tr_ymin", ascanf_tr_ymin, 1, NOT_EOF, "tr_ymin[setNumber]" },
	{ "tr_errmin", ascanf_tr_errmin, 1, NOT_EOF, "tr_errmin[setNumber]" },
	{ "tr_xmax", ascanf_tr_xmax, 1, NOT_EOF, "tr_xmax[setNumber]" },
	{ "tr_ymax", ascanf_tr_ymax, 1, NOT_EOF, "tr_ymax[setNumber]" },
	{ "tr_errmax", ascanf_tr_errmax, 1, NOT_EOF, "tr_errmax[setNumber]" },
	{ "xmin", ascanf_xmin, 1, NOT_EOF, "xmin[setNumber]" },
	{ "ymin", ascanf_ymin, 1, NOT_EOF, "ymin[setNumber]" },
	{ "errmin", ascanf_errmin, 1, NOT_EOF, "errmin[setNumber]" },
	{ "xmax", ascanf_xmax, 1, NOT_EOF, "xmax[setNumber]" },
	{ "ymax", ascanf_ymax, 1, NOT_EOF, "ymax[setNumber]" },
	{ "errmax", ascanf_errmax, 1, NOT_EOF, "errmax[setNumber]" },
	{ "tr_curve_len", ascanf_tr_curve_len, 5, NOT_EOF, "tr_curve_len[[n[,i[,signed[,&results[,&lengths]]]]]]: the length along the current [nth] dataset\n"
		" after transformations, or idem at point <i>\n"
		" See curve_len[].\n"
	},
	{ "curve_len", ascanf_curve_len, 5, NOT_EOF, "curve_len[[n[,i[,signed[,&results[,&lengths]]]]]]: the length (X,Y) along the current [nth] dataset\n"
		" (in the current window), or idem at point <i>\n"
		" If <signed> is True, the calculation is done preserving the sign of the segment lengths, which are assumed to be\n"
		" negative if dx<0 (signed==1) or dy<0 (signed==-1), or (if signed==2), when the rotation from the 1st to the second\n"
		" point is clockwise (and all points should --approximately-- lie on a circle).\n"
		" If signed==3, the set is translated such that its centroid (avX,avY) is at (0,0), and then each segment is rotated\n"
		" such that the 1st point is on the positive X ax; clockw. rotation (neg. length) is then when the 2nd point is in the\n"
		" 3rd or 4th quadrant.\n"
		" If &results points to an array, it will contain the lengths upto <i>.\n"
		" If &lengths points to a doubles array, it will contain the individual segment lengths (with the 1st element being 0)\n"
	},
	{ "curve_len_arrays", ascanf_curve_len_arrays, 6, NOT_EOF_OR_RETURN,
		"curve_len_arrays[&xvals,&yvals[,i[,signed[,&results[,&lengths]]]]]: the length (X,Y) along the\n"
		" trajectory defined by the points in xvals and yvals. See curve_len[] for the other arguments.\n"
	},
	{ "error_len", ascanf_error_len, 2, NOT_EOF, "error_len[[n[,i]]]: the sum over the absolute differences\n"
		" in error OR the total change in orientation (in vector mode) of the current [nth] dataset\n"
		" (in the current window), or idem at point <i>\n"
	},
	{ "LZx", ascanf_LZx, 0, NOT_EOF, "LZx: the log_zero_x value"},
	{ "LZy", ascanf_LZy, 0, NOT_EOF, "LZy: the log_zero_y value"},
	{ "index", ascanf_Index, 0, NOT_EOF, "index (of expression-element)" },
	{ "self", ascanf_self, 0, NOT_EOF, "self (initial value = parameter range variable)" },
	{ "split", ascanf_split, 1, NOT_EOF,
		"while reading data: split: new dataset or split[1]: new dataset after current point; returns self\n"
		" runtime: split[<exp>] split *at* current point when <exp> = True\n"
	},
	{ "current", ascanf_current, 0, NOT_EOF, "current: current value" },
	{ "elapsed", ascanf_elapsed, 1, NOT_EOF, "elapsed: since last call (real time) or elapsed[x] (user time)"},
	{ "time", ascanf_time, 1, NOT_EOF, "time: since first call (real time) or time[x] (user time)"},
	{ "system.time", ascanf_systemtime, AMAXARGS, NOT_EOF_OR_RETURN,
		"system.time[expression]: returns the \"system\" time needed to evaluate <expression>.\n"
		" This is independent of the timer used by elapsed[], but sets $elapsed[0-2] in the same fashion.\n"
		" In compiled mode, the name of the first expression is printed, for clarity.\n"
		" If the expression sets $elapsed[3] to an estimate of the number of operations performed,\n"
		" the number of ops. per second will be printed too ($elapsed[3] will be reset!).\n"
	},
	{ "system.time.p", ascanf_systemtime2, AMAXARGS, NOT_EOF_OR_RETURN,
		"system.time.p[expression]: identical to system.time except that in compiled mode, it prints\n"
		" the name of the parent expression. It can thus be used instead of e.g. progn as the toplevel\n"
		" expression 'scope' in procedure, printing the procedure name with the timing data.\n"
	},
	{ "no.system.time", ascanf_systemtime_silent, AMAXARGS, NOT_EOF_OR_RETURN,
		"no.system.time[expression]: silent version of system.time[] and system.time.p[]\n"
	},
	{ "sleep", ascanf_sleep_once, AMAXARGS, NOT_EOF_OR_RETURN, "sleep[sec[,exprs]]: wait for <sec> seconds" },
	{ "usleep", ascanf_usleep, AMAXARGS, NOT_EOF_OR_RETURN, "usleep[microseconds]: wait for <microseconds>" },
	{ "intertimer", ascanf_set_interval, AMAXARGS, NOT_EOF_OR_RETURN,
		"intertimer[init,interv[,exprs]]: set an interval timer: init. delay <init>, interval <interv>" },
	{ "gettimeofday", ascanf_gettimeofday, 0, NOT_EOF,
		"gettimeofday: time since the Epoch: see gettimeofday(3). Typically, this gives microsecond\n"
		" resolution. In time-critical applications, this routine can be used as a lowlevel timer\n"
		" with a baseline (the 1st sample it returns).\n"
	},
#ifdef FFTW_CYCLES_PER_SEC
	{ "HRTimer", ascanf_HRTimer, 0, NOT_EOF,
		"HRTimer: interface to a lowlevel, cpu timer (the time register on the Pentium/PowerPC, representing uptime).\n"
		" Should be the fastest and highest-resolution timer available.\n"
	},
#else
	{ "HRTimer", ascanf_gettimeofday, 0, NOT_EOF,
		"HRTimer: normally, an interface to a lowlevel, cpu timer (the time register on the Pentium/PowerPC)\n"
		" None is available on this system, or none has been configured. HRTimer is an alias for\n"
		" gettimeofday.\n"
	},
#endif
	{ "YearDay", ascanf_YearDay, 3, NOT_EOF_OR_RETURN,
		"YearDay[[day,month,year]]: returns the number of days since Jan, 1st.\n"
		" When no arguments are passed, this applies to today, otherwise the specified date\n"
		" is used. A value of NaN means use today's value. <year> must be fully specified (i.e. 99!=1999).\n"
		" The first day is day 1, in contrast to the tm_yday field of mktime(2). For an invalid day,\n"
		" the function returns NaN.\n"
	},
	{ "Restart", ascanf_restart, 0, NOT_EOF, "Restart: restarts the program by \"raising\" a SIGUSR2" },
	{ "add", ascanf_add, AMAXARGS, NOT_EOF_OR_RETURN, "add[x,y[,..]] or add[&dest_array,&array1,&array2]" },
	{ "sub", ascanf_sub, AMAXARGS, NOT_EOF_OR_RETURN, "sub[x,y[,..]] or sub[&dest_array,&array1,&array2]" },
	{ "relsub", ascanf_relsub, AMAXARGS, NOT_EOF_OR_RETURN,
		"relsub[x,y[,relat[,absol]]] or relsub[&dest_array,&array1,&array2[,relat[,absol]]]\n"
		" calculates x-y or (x-y)/y when relat==True; takes absolute value when absol==True\n"
	},
	{ "min", ascanf_sub, AMAXARGS, NOT_EOF_OR_RETURN, "min[x,y[,..]] or min[&dest_array,&array1,&array2]" },
	{ "mul", ascanf_mul, AMAXARGS, NOT_EOF_OR_RETURN, "mul[x,y[,..]] or mul[&dest_array,&array1,&array2]" },
	{ "div", ascanf_div, AMAXARGS, NOT_EOF_OR_RETURN, "div[x,y[,..]] or div[&dest_array,&array1,&array2]" },
	{ "zero_div_zero", ascanf_zero_div_zero, 1, NOT_EOF, "zero_div_zero or zero_div_zero[new_val]: define value of 0/0" },
	{ "randomise", ascanf_randomise, 2, NOT_EOF, "randomise or randomise[low,high]: randomises first" },
	{ "ran", ascanf_random, 3, NOT_EOF,
		"ran or ran[low,high[,cond]]: the optional <cond> argument can be:\n"
		" <1: impose this as the minimal fractional difference of the <low>,<high> range\n"
		"     between the generated and the previous value\n"
		" >=1: perform this many polls of the generator before returning a value\n"
		" <=-1: perform between 0 and this many polls of the generator before returning a value\n"
	},
#ifdef linux
	{ "kran", ascanf_krandom, 3, NOT_EOF,
		"kran or kran[low,high[,cond]]; identical to the ran routine, but uses the near-perfect though slow /dev/urandom device\n"
	},
#else
	{ "kran", ascanf_krandom, 3, NOT_EOF,
		"kran or kran[low,high[,cond]]; identical to the ran routine; different (much better but slower) on linux systems\n"
	},
#endif
	{ "pi", ascanf_pi, AMAXARGS, NOT_EOF_OR_RETURN, "pi or pi[mulfac[,fac[,fac]]]" },
	{ "pow", ascanf_pow, 3, NOT_EOF_OR_RETURN, "pow[x,y] or pow[&result,&x,&y].\n"
		" When x==0.5, sqrt() is used, for x==1/3, cbrt().\n"
	},
	{ "pow*", ascanf_pow_, 2, NOT_EOF_OR_RETURN, "pow*[x,y]: pow*[-x,y] = -pow*[x,y]" },
	{ "fac", ascanf_fac, 2, NOT_EOF_OR_RETURN, "fac[x,dx]" },
	{ "radians", ascanf_radians, 1, NOT_EOF, "radians[x]" },
	{ "degrees", ascanf_degrees, 1, NOT_EOF, "degrees[x]" },
	{ "sin", ascanf_sin, 2, NOT_EOF_OR_RETURN, "sin[x[,base]]" },
	{ "cos", ascanf_cos, 2, NOT_EOF_OR_RETURN, "cos[x[,base]]" },
	{ "tan", ascanf_tan, 2, NOT_EOF_OR_RETURN, "tan[x[,base]]" },
	{ "sincos", ascanf_sincos, 5, NOT_EOF_OR_RETURN,
		"sincos[x,base,&s,&c[,&t]]: computes base <base> sin and cos of <x>,\n"
		" and stores results in &s and &c (and the tan in &t if supplied).\n"
		" x, s, c and t can all be (double) arrays\n"
	},
	{ "asin", ascanf_asin, 2, NOT_EOF_OR_RETURN, "asin[x[,base]]" },
	{ "acos", ascanf_acos, 2, NOT_EOF_OR_RETURN, "acos[x[,base]]" },
	{ "atan", ascanf_atan, 2, NOT_EOF_OR_RETURN, "atan[x[,base]]" },
	{ "atan2", ascanf_atan2, 3, NOT_EOF_OR_RETURN, "atan2[x,y[,base]] atan2(x,y)==atan(y/x)" },
	{ "atan3", ascanf_atan3, 3, NOT_EOF_OR_RETURN, "atan3[x,y[,base]] atan2(x,y) in [0,2PI]" },
	{ "cnv_angle1", ascanf_cnv_angle1, 2, NOT_EOF_OR_RETURN, "cnv_angle1[x[,base]]: x in <-base/2,base/2]" },
	{ "cnv_angle2", ascanf_cnv_angle2, 2, NOT_EOF_OR_RETURN, "cnv_angle2[x[,base]]: x in [0,base]" },
	{ "arg", ascanf_arg, 4, NOT_EOF_OR_RETURN, "arg[x,y[,base[,offset]]] angle to (x,y) in 0..2PI (base)" },
	{ "angdiff", ascanf_angdiff, 3, NOT_EOF_OR_RETURN, "angdiff[a,b[,base]] difference between 2 angles in 0..2PI (base)" },
	{ "UnwrapAngles", ascanf_UnwrapAngles, 2, NOT_EOF_OR_RETURN, "UnwrapAngles[<angles_p>[,radix]]: remove the jumps due to circularity (at e.g. +-180 or 0/360) from an array of consecutive angles\n"
		" Returns -1 upon error, or else the number of wraps made\n"
	},
	{ "len", ascanf_len, AMAXARGS, NOT_EOF_OR_RETURN,
		"len[x,y[,z[,a...]]]: distance to (x,y) or (x,y,z) or ...\n"
		" Can be called with 3 arrays: len[&result, &x, &y]\n"
	},
	{ "exp", ascanf_exp, 1, NOT_EOF, "exp[x]" },
	{ "erf", ascanf_erf, 1, NOT_EOF_OR_RETURN, "erf[x]" },
	{ "1-erf", ascanf_erfc, 1, NOT_EOF_OR_RETURN, "1-erf[x]" },
	{ "log", ascanf_log, 2, NOT_EOF_OR_RETURN, "log[x,base]" },
	{ "abs", ascanf_abs, 1, NOT_EOF_OR_RETURN, "abs[x]" },
	{ "sign", ascanf_sign, 1, NOT_EOF_OR_RETURN, "sign[x]" },
	{ "Ent", ascanf_Ent, 1, NOT_EOF_OR_RETURN, "Ent[x]: Entier function" },
	{ "RoundOff", ascanf_RoundOff, 3, NOT_EOF_OR_RETURN,
		"RoundOff[x[,precision[,round]]]: round off <x> to the given precision (nr. of decimals). The <round>\n"
		" argument defaults to 0.5 and determines at which point rounding up occurs.\n"
		" Thus: div[ floor[ add[ mul[x,pow[10,precision]], round] ], pow[10,precision] ] when x>=0\n"
		" Thus: div[ ceil[ sub[ mul[x,pow[10,precision]], round] ], pow[10,precision] ] when x>0\n"
	},
	{ "gRoundOff", ascanf_gRoundOff, 1, NOT_EOF_OR_RETURN, "gRoundOff[x]: printf <x> using the global format (Settings Dialog) and sscanf it back" },
	{ "floor", ascanf_floor, 1, NOT_EOF_OR_RETURN, "floor[x]" },
	{ "ceil", ascanf_ceil, 1, NOT_EOF_OR_RETURN, "ceil[x]" },
	{ "cmp", ascanf_dcmp, 3, NOT_EOF_OR_RETURN, "cmp[x,y[,precision]]: check x against y with precision <precision> (fraction of y)" },
#if defined(__ppc__)
	{ "fmod", ascanf_fmod2, 2, NOT_EOF_OR_RETURN,
		"fmod[x,y]: calculates x - y * sign[x/y] * floor[|x/y|]\n"
		" On this platform, this is faster than the native fmod() routine, itself used by fmod2[].\n"
	},
	{ "fmod2", ascanf_fmod, 2, NOT_EOF_OR_RETURN, "fmod2[x,y]: uses native fmod() routine; fmod2[x,0]== x" },
#else
	{ "fmod", ascanf_fmod, 2, NOT_EOF_OR_RETURN, "fmod[x,y]: fmod[x,0]== x" },
	{ "fmod2", ascanf_fmod2, 2, NOT_EOF_OR_RETURN,
		"fmod2[x,y]: calculates x - y * sign[x/y] * floor[|x/y|]\n"
		" On some platforms, this is faster than the native fmod[] routine, and hence fmod and fmod2 are interchanged there.\n"
	},
#endif
	{ "fmod3", ascanf_fmod3, 2, NOT_EOF_OR_RETURN, "fmod3[x,y]: calculates ??" },
	{ "ifelse2", ascanf_if2, AMAXARGS, NOT_EOF_OR_RETURN, "ifelse2[expr,val1,[else-val:0]] - all arguments are evaluated" },
	{ "ifelse", ascanf_if, AMAXARGS, NOT_EOF_OR_RETURN, "ifelse[expr,val1,[else-val:0]] - lazy evaluation" },
	{ "switch0", ascanf_switch0, AMAXARGS, NOT_EOF_OR_RETURN,
		"switch[expr,case1,expr1[,case2,expr2,..,..[,default_expr]]] returns evaluated {exprN} for which {expr}=={caseN},\n"
		" or else {default_expr}\n"
		" (original version of switch[]; may be slightly faster.)\n"
	},
	{ "switch", ascanf_switch, AMAXARGS, NOT_EOF_OR_RETURN,
		"switch[expr,case1,expr1[,case2,expr2,..,..[,default_expr]]] returns evaluated {exprN} for which {expr}=={caseN},\n"
		" or else {default_expr}\n"
		" If <expr> is not an arraypointer, either of the case arguments may be an arraypointer, and checks will be made\n"
		" for {expr} in {caseN values} (this allows multiple case:s as in C).\n"
	},
	{ "&", ascanf_land, 2, NOT_EOF_OR_RETURN, "&[x,y] bitwise and" },
	{ "|", ascanf_lor, 2, NOT_EOF_OR_RETURN, "|[x,y] bitwise or" },
	{ "^", ascanf_lxor, 2, NOT_EOF_OR_RETURN, "^[x,y] bitwise xor" },
	{ "~", ascanf_lnot, 1, NOT_EOF_OR_RETURN, "~[x] bitwise not" },
	{ "<<", ascanf_shl, 2, NOT_EOF_OR_RETURN, "<<[x,n] bitwise shift left" },
	{ ">>", ascanf_shr, 2, NOT_EOF_OR_RETURN, ">>[x,n] bitwise shift right" },
	{ "=", ascanf_eq, 3, NOT_EOF_OR_RETURN, "=[x,y[,precision]]: precision==-1 means use machine precision (DBL_EPSILON)\n" },
/* 	{ "!=", ascanf_neq, 2, NOT_EOF_OR_RETURN, "!=[x,y]" },	*/
	{ ">=", ascanf_ge, 3, NOT_EOF_OR_RETURN, ">=[x,y[,precision]]" },
	{ "<=", ascanf_le, 3, NOT_EOF_OR_RETURN, "<=[x,y[,precision]]" },
	{ ">", ascanf_gt, 2, NOT_EOF_OR_RETURN, ">[x,y]" },
	{ "<", ascanf_lt, 2, NOT_EOF_OR_RETURN, "<[x,y]" },
	{ "AND", ascanf_and, AMAXARGS, NOT_EOF_OR_RETURN, "AND[x,y[,..]] (boolean)" },
	{ "OR", ascanf_or, AMAXARGS, NOT_EOF_OR_RETURN, "OR[x,y[,..]] (boolean)" },
	{ "XOR", ascanf_xor, 2, NOT_EOF_OR_RETURN, "XOR[x,y] (\"boolean\")" },
	{ "NOT", ascanf_not, 1, NOT_EOF_OR_RETURN, "NOT[x] (boolean)" },
	{ "IsBitSet", ascanf_bitset, 2, NOT_EOF_OR_RETURN, "IsBitSet[x,bit] (boolean); up to 64 bits supported" },
	{ "MIN", ascanf_MIN, AMAXARGS, NOT_EOF_OR_RETURN,
		"MIN[x,y[,..]]: passing a pointer to an array equals passing all the array's arguments."
	},
	{ "MAX", ascanf_MAX, AMAXARGS, NOT_EOF_OR_RETURN,
		"MAX[x,y[,..]]: passing a pointer to an array equals passing all the array's arguments."
	},
	{ "ABSMIN", ascanf_ABSMIN, AMAXARGS, NOT_EOF_OR_RETURN,
		"ABSMIN[x,y[,..]]: as MIN, but compares the absolute values while preserving sign."
	},
	{ "ABSMAX", ascanf_ABSMAX, AMAXARGS, NOT_EOF_OR_RETURN,
		"ABSMAX[x,y[,..]]: as MAX, but compares the absolute values while preserving sign."
	},
	{ "even", ascanf_Even, 1, NOT_EOF_OR_RETURN, "even[x]: returns TRUE when x is even" },
	{ "ELEM", ascanf_ELEM, 5, NOT_EOF_OR_RETURN, "ELEM[x,low,high,<l_inclusive>,<h_inclusive>]" },
	{ "clip", ascanf_clip, 3, NOT_EOF_OR_RETURN, "clip[expr,min,max]" },
	{ "within", ascanf_within, AMAXARGS, NOT_EOF_OR_RETURN,
		"within[x,val[,val,..]]: check if x is a member of the collection formed by the 2nd and\n"
		" higher arguments. If one of these is a pointer to an array AND x is not a pointer itself,\n"
		" also check against the members of that array.\n"
	},
	{ "contained", ascanf_contained, 5, NOT_EOF_OR_RETURN,
		"contained[x,value-array-ptr[,margin]] or contained[x,low,high[,low-inclusive[,high-inclusive]]]:\n"
		" Similar to within[]. If x is a scalar, check if it is contained in the array (value-array-ptr)\n"
		" or in the second form, check if x is contained within the specified low,high boundaries.\n"
		" If x is a pointer to an array itself, perform these checks on all its elements until a hit is found.\n"
	},
	{ "\\#xce\\", ascanf_within, AMAXARGS, NOT_EOF_OR_RETURN,
		"\\#xce\\[x,val[,val,..]]; as within, math. symbol (\\\\\\#-x-c-e\\\\\\)"
	},
	{ "isNaNorInf", ascanf_isNaNorInf, AMAXARGS, NOT_EOF_OR_RETURN,
		"isNaNorInf[x[,...]]: is x or any of the other arguments a NaN or Inf?" },
	{ "isNaN", ascanf_isNaN, AMAXARGS, NOT_EOF_OR_RETURN,
		"isNaN[x[,...]]: is x or any of the other arguments a NaN?" },
	{ "isInf", ascanf_isInf, AMAXARGS, NOT_EOF_OR_RETURN,
		"isInf[x[,...]]: is x or any of the other arguments an Inf? Returns -1 if -Inf, 1 if Inf, 0 otherwise\n"
		" (= the sign of the first infinite value)."
	},

	{ "MEM", ascanf_mem, 2, NOT_EOF_OR_RETURN,
		"MEM[n[,expr]]: set MEM[n] to expr, or get MEM[n]: " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to set whole range"
	},
	{ "SETMXYZ", ascanf_setmxyz, 3, NOT_EOF_OR_RETURN, "SETMXYZ[I,J,K]: set new dimensions of MXYZ buffer" },
	{ "MXYZ+", ascanf_mxyz_cum, 4, NOT_EOF_OR_RETURN,
		"MXYZ+[i,j,k[,expr]]: increment MXYZ+[i,j,k] with expr, or get MXYZ[i,j,k]: i,j or k == -1 to increment range\n"
		" (returns value of last increment)\n"
	},
	{ "MXYZ", ascanf_mxyz, 4, NOT_EOF_OR_RETURN,
		"MXYZ[i,j,k[,expr]]: set MXYZ[i,j,k] to expr, or get MXYZ[i,j,k]: i,j or k == -1 to set range"
	},
	{ "SETMXY", ascanf_setmxy, 2, NOT_EOF_OR_RETURN, "SETMXY[I,J]: set new dimensions of MXY buffer" },
	{ "MXY+", ascanf_mxy_cum, 3, NOT_EOF_OR_RETURN,
		"MXY+[i,j[,expr]]: increment MXY+[i,j] with expr, or get MXY[i,j: i or j == -1 to increment range\n"
		" (returns value of last increment)\n"
	},
	{ "MXY", ascanf_mxy, 3, NOT_EOF_OR_RETURN,
		"MXY[i,j[,expr]]: set MXY[i,j] to expr, or get MXY[i,j]"
	},
	{ "SS_exact", ascanf_SS_exact, 2, NOT_EOF,
		"SS_exact[n[,expr]]: set $SS_exact for (and reset) individual statistics bin n\n"
		"\tSpecify n=-1 to set for every bin; missing expr resets bin(s)\n"
		" Returns expr or the current value.\n"
	},
	{ "SS_SampleSize", ascanf_SS_SampleSize, 2, NOT_EOF,
		"SS_SampleSize[n[,expr]]: set the sample size (and $SS_exact) for (and reset) individual statistics bin n\n"
		"\tSpecify n=-1 to set for every bin; missing expr resets bin(s); value<0 de-allocates.\n"
		" Returns expr or the current value.\n"
	},
	{ "SS_Add", ascanf_SS_set, 3, NOT_EOF_OR_RETURN,
		"SS_Add[n[,expr[,weight]]]: store expr in statistics bin n, with weigth (def.1): " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to store in every bin; missing expr resets bin(s)\n"
		" Returns expr or 0\n"
	},
	{ "M_SS_Add", ascanf_SS_set2, 3, NOT_EOF_OR_RETURN,
		"M_SS_Add[n[,expr[,weight]]]: store expr in statistics bin n, with weigth (def.1): " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to store in every bin; missing expr resets bin(s)\n"
		" Returns the current mean in the bin, or 0\n"
	},
	{ "SS_AddArray", ascanf_SS_setArray, 5, NOT_EOF_OR_RETURN,
		"SS_AddArray[n[,<src_p>[,<weight_p>[,start[,end]]]]]: store the array pointed to by <src_p> in statistics bin n, with <weigth_p> (def.1): " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to store in every bin; missing 2nd and 3rd arguments resets bin(s)\n"
		" <weight_p> may point to an array or to a variable, or be 0 to be ignored\n"
		" The optional <start> and <end> specify the starting and ending (exclusive!!) elements to be stored.\n"
		" Returns src_p[last_item] or 0\n"
	},
	{ "SS_Pop", ascanf_SS_pop, 4, NOT_EOF_OR_RETURN,
		"SS_Pop[n,count,pos[,update]]]: pop (remove) <count> items from bin <n>, starting at position <pos>.\n"
		" Makes sense only in 'exact' mode.\n"
		" Returns the number of items removed, or a value <0 in case of an  error\n"
	},
	{ "SS_Mean", ascanf_SS_Mean, 1, NOT_EOF_OR_RETURN, "SS_Mean[n]: get mean for stats bin n" },
	{ "SS_Median", ascanf_SS_Median, 1, NOT_EOF_OR_RETURN, "SS_Median[n]: get median for stats bin n ($SS_exact must be 1)" },
	{ "SS_Quantile", ascanf_SS_Quantile, 2, NOT_EOF_OR_RETURN,
		"SS_Quantile[n,prob]: get the quantile at <prob> ([0,1]) for stats bin n ($SS_exact must be 1)\n"
		" Tukey's quantiles (\"hinges\") are close to prob=1/4,3/4\n"
	},
	{ "SS_Stdev", ascanf_SS_St_Dev, 1, NOT_EOF_OR_RETURN, "SS_Stdev[n]: get standard deviation for stats bin n" },
	{ "SS_Sterr", ascanf_SS_St_Err, 1, NOT_EOF_OR_RETURN, "SS_Sterr[n]: get standard error for stats bin n" },
	{ "SS_CV", ascanf_SS_CV, 1, NOT_EOF_OR_RETURN, "SS_CV[n]: get coefficient of variance for stats bin n" },
	{ "SS_Adev", ascanf_SS_ADev, 1, NOT_EOF_OR_RETURN, "SS_Adev[n]: get average or mean absolute deviation for stats bin n ($SS_exact must be 1), using the median" },
	{ "SS_Skew", ascanf_SS_Skew, 1, NOT_EOF_OR_RETURN, "SS_Skew[n]: get the skewness for stats bin n" },
	{ "SS_Sum", ascanf_SS_Sum, 1, NOT_EOF_OR_RETURN, "SS_Sum[n]: get sum of elements (multiplied by the weights) in stats bin n" },
	{ "SS_SumSqr", ascanf_SS_SumSqr, 1, NOT_EOF_OR_RETURN,
		"SS_SumSqr[n]: get sum of the square of the elements (multiplied by the weights) in stats bin n"
	},
	{ "SS_SumCub", ascanf_SS_SumCub, 1, NOT_EOF_OR_RETURN,
		"SS_SumCub[n]: get sum of the 3rd power of the elements (multiplied by the weights) in stats bin n"
	},
	{ "SS_Count", ascanf_SS_Count, 1, NOT_EOF_OR_RETURN, "SS_Count[n]: get nr. of elements in stats bin n" },
	{ "SS_WeightSum", ascanf_SS_WeightSum, 1, NOT_EOF_OR_RETURN, "SS_WeightSum[n]: get sum of weights in stats bin n" },
	{ "SS_Min", ascanf_SS_min, 1, NOT_EOF_OR_RETURN, "SS_Min[n]: get minimum in stats bin n" },
	{ "SS_PosMin", ascanf_SS_pos_min, 1, NOT_EOF_OR_RETURN, "SS_PosMin[n]: get minimal positive value in stats bin n" },
	{ "SS_NegMax", ascanf_SS_neg_max, 1, NOT_EOF_OR_RETURN,
		"SS_NegMax[n]: get maximal negative value in stats bin n, that has to be in exact mode.\n"
		" This operation is done off-line (contr. to the PosMin determination), and at each\n"
		" invocation of SS_NegMax: it is thus (relatively expensive).\n"
	},
	{ "SS_Max", ascanf_SS_max, 1, NOT_EOF_OR_RETURN, "SS_Max[n]: get maximum in stats bin n" },
	{ "SS_Sample", ascanf_SS_Sample, 2, NOT_EOF_OR_RETURN,
		"SS_Sample[n,i]: return sample i in bin n that must be in exact mode.\n"
	},
	{ "SS_SampleWeight", ascanf_SS_SampleWeight, 2, NOT_EOF_OR_RETURN,
		"SS_SampleWeight[n,i]: return the weight for sample i in bin n that must be in exact mode.\n"
	},
	{ "SS_FTest", ascanf_SS_FTest, 2, NOT_EOF_OR_RETURN,
		"SS_FTest[n1,n2]: significance by F-Test on SS[n1] and SS[n2] ($SS_exact must be 1; returns 2 upon any error)\n"
		" n1 and n2 may be integers indicating the desired stat bins, both may be pointers to $SS statsbin variables,\n"
		" or n1 may be a pointer to a $SS statsbin *array*, and n2 a pointer to an array of at least 2 elements, indicating\n"
		" the elements in n1 to compare.\n"
	},
	{ "SS_TTest", ascanf_SS_TTest, 2, NOT_EOF_OR_RETURN,
		"SS_TTest[n1,n2]: significance by T-Test on SS[n1] and SS[n2] ($SS_exact must be 1; returns 2 upon any error)\n"
		" See under SS_FTest for the allowed values for n1 and n2.\n"
	},
	{ "SS_TTest_uneq", ascanf_SS_TTest_uneq, 2, NOT_EOF_OR_RETURN,
		"SS_TTest_uneq[n1,n2]: significance by T-Test on SS[n1] and SS[n2] n1 and n2 have unequal variances\n"
		" See under SS_FTest for the allowed values for n1 and n2.\n"
	},
	{ "SS_TTest_paired", ascanf_SS_TTest_paired, 2, NOT_EOF_OR_RETURN,
		"SS_TTest_paired[n1,n2]: significance by T-Test on paired samples from SS[n1] and SS[n2]\n"
		" See under SS_FTest for the allowed values for n1 and n2.\n"
	},
	{ "SS_TTest_correct", ascanf_SS_TTest_correct, 3, NOT_EOF_OR_RETURN,
		"SS_TTest_correct[n1,n2,f-prob_max]: significance by T-Test, equal or unequal variances depending on result of F-test and <f-prob_max>\n"
		" See under SS_FTest for the allowed values for n1 and n2.\n"
	},
	{ "$SS_StatsBin", ascanf_SS_StatsBin, AMAXARGS, NOT_EOF_OR_RETURN,
		"$SS_StatsBin[&varptr, exact[,value1[,value2]]]: convert a (scalar!) variable to an $SS statsbin that can be\n"
		" passed as a 1st argument to SS_Add, SS_Mean, etc. as a pointer.\n"
	},
	{ "SAS_exact", ascanf_SAS_exact, 2, NOT_EOF,
		"SAS_exact[n[,expr]]: set $SAS_exact for (and reset) individual angular statistics bin n\n"
		"\tSpecify n=-1 to set for every bin; missing expr resets bin(s)\n"
		" Returns expr or the current value.\n"
	},
	{ "SAS_SampleSize", ascanf_SAS_SampleSize, 2, NOT_EOF,
		"SAS_SampleSize[n[,expr]]: set the sample size (and $SAS_exact) for (and reset) individual statistics bin n\n"
		"\tSpecify n=-1 to set for every bin; missing expr resets bin(s); value<0 de-allocates.\n"
		" Returns expr or the current value.\n"
	},
	{ "SAS_GonioBase", ascanf_SAS_GonioBase, 2, NOT_EOF,
		"SAS_GonioBase[n[,expr]]: set the GonioBase (radix) value for individual statistics bin n\n"
		"\tSpecify n=-1 to set for every bin; missing expr resets bin(s).\n"
		" Returns expr or the current value\n"
	},
	{ "SAS_GonioOffset", ascanf_SAS_GonioOffset, 2, NOT_EOF,
		"SAS_GonioOffset[n[,expr]]: set the GonioOffset value for individual statistics bin n\n"
		"\tSpecify n=-1 to set for every bin; missing expr resets bin(s).\n"
		" Returns expr or the current value\n"
	},
	{ "SAS_Add", ascanf_SAS_set, 6, NOT_EOF_OR_RETURN,
		"SAS_Add[n[,expr[,weight[,radix[,offset[,convert]]]]]]: store angular expr in statistics bin n, with weigth (def.1): " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to store in every bin; missing expr resets bin(s)\n"
		" returns expr or 0\n"
	},
	{ "M_SAS_Add", ascanf_SAS_set2, 6, NOT_EOF_OR_RETURN,
		"M_SAS_Add[n[,expr[,weight[,radix[,radix_offset[,convert]]]]]]: store angular expr in statistics bin n, with weigth (def.1): " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to store in every bin; missing expr resets bin(s)\n"
		" Returns the current mean in the specified bin, or 0\n"
	},
	{ "SAS_AddArray", ascanf_SAS_setArray, 8, NOT_EOF_OR_RETURN,
		"SAS_AddArray[n[,<src_p>[,<weight_p>[,start[,end[,radix[,offset[,convert]]]]]]]]: store array of angles pointed to by <src_p>  in statistics bin n, with <weigth_p> (def.1): " STRING(AMAXARGS) " locations"\
		"\n\tSpecify n=-1 to store in every bin; missing 2nd and higher arguments resets bin(s)\n"
		" <weight_p> may point to an array or to a variable, or be 0 to be ignored\n"
		" Returns src_p[last_item] or 0\n"
	},
	{ "SAS_Mean", ascanf_SAS_Mean, 2, NOT_EOF_OR_RETURN, "SAS_Mean[n[,convert]]: get mean for angular stats bin n" },
	{ "SAS_Stdev", ascanf_SAS_St_Dev, 1, NOT_EOF_OR_RETURN, "SAS_Stdev[n]: get standard deviation for angular stats bin n" },
	{ "SAS_CV", ascanf_SAS_CV, 1, NOT_EOF_OR_RETURN, "SAS_CV[n]: get coefficient of variance for angular stats bin n" },
	{ "SAS_Count", ascanf_SAS_Count, 1, NOT_EOF_OR_RETURN, "SAS_Count[n]: get nr. of elements in angular stats bin n" },
	{ "SAS_Min", ascanf_SAS_min, 1, NOT_EOF_OR_RETURN, "SAS_Min[n]: get minimum in angular stats bin n" },
	{ "SAS_Max", ascanf_SAS_max, 1, NOT_EOF_OR_RETURN, "SAS_Max[n]: get maximum in angular stats bin n" },
	{ "SAS_Sample", ascanf_SAS_Sample, 2, NOT_EOF_OR_RETURN,
		"SAS_Sample[n,i]: return sample i in bin n that must be in exact mode.\n"
	},
	{ "SAS_SampleWeight", ascanf_SAS_SampleWeight, 2, NOT_EOF_OR_RETURN,
		"SAS_SampleWeight[n,i]: return the weight for sample i in bin n that must be in exact mode.\n"
	},
	{ "$SAS_StatsBin", ascanf_SAS_StatsBin, AMAXARGS, NOT_EOF_OR_RETURN,
		"$SAS_StatsBin[&varptr, exact[,radix,offset,convert,value1[,value2]]]: convert a (scalar!) variable to an $SAS statsbin\n"
		" that can be passed as a 1st argument to SAS_Add, SAS_Mean, etc. as a pointer.\n"
	},

	{ "F-Test", ascanf_FTest, 4, NOT_EOF_OR_RETURN, "F-Test[stdv1,n1,stdv2,n2]: significance by F-Test (returns 2 upon any error)" },
	{ "T-Test", ascanf_TTest, 6, NOT_EOF_OR_RETURN, "T-Test[mean1,stdv1,n1,mean2,stdv2,n2]: significance by T-Test (returns 2 upon any error)" },
	{ "T-Test_uneq", ascanf_TTest_uneq, 6, NOT_EOF_OR_RETURN, "T-Test_uneq[mean1,stdv1,n1,mean1,stdv2,n2]: significance by T-Test, unequal variances" },
	{ "T-Test_correct", ascanf_TTest_correct, 7, NOT_EOF_OR_RETURN,
		"T-Test_correct[mean2,stdv1,n1,mean1,stdv2,n2,f-prob_max] or\n"
		" T-Test_correct[&array1, &array2, f-prob_max]: significance by T-Test, equal or unequal variances,\n"
		" depending on result of F-test and <f-prob_max>\n"
		" Set f-prob_max[1] to always to an unequal-variance test, f-prob_max[0] for equal-variance tests.\n"
	},
	{ "Incomplete-Gamma", ascanf_IncomplGamma, 2, NOT_EOF,
		"Incomplete-Gamma[a,x]: returns the incomplete gamma function P(a,x)\n"
	},
	{ "ChiSquare-Prob", ascanf_ChiSquare, 5, NOT_EOF_OR_RETURN,
		"ChiSquare-Prob[&observed, &expected, constraints, &chsq-return[,&df-return]]: compare the observed number of events\n" 
		" in the bins &observed with the &expected numbers using a ChiSquare test, and return the probability that the\n"
		" observed distribution is consistent with the expected distribution. The bins are either exact-mode $SS_StatsBin\n"
		" variables, or arrays. <constraints> gives the number of constraints to impose; if the expected numbers (sum[expected]) were not\n"
		" normalised to match the observed number of events, than constraints==0, otherwise it must be 1. For each additional\n"
		" post-hoc-adjusted free parameter, constraints must be increased with 1. The calculated chi-square value is returned\n"
		" in &chsq-return (which must be a scalar). The determined degrees-of-freedom may be returned in the optional &df-return\n"
		" argument.\n"
	},
	{ "ChiSquare2-Prob", ascanf_ChiSquare2, 5, NOT_EOF_OR_RETURN,
		"ChiSquare2-Prob[&observed1, &observed2, constraints, &chsq-return[,&df-return]]: compare the observed number of events\n" 
		" in the bins &observed1 and &observed2 using a ChiSquare test, and return the probability that the\n"
		" observed distribution is consistent with the expected distribution. The bins are either exact-mode $SS_StatsBin\n"
		" variables, or arrays. <constraints> gives the number of constraints to impose; if the 2 samples were forced to equal\n"
		" length (by discarding superfluous samples; this refers to the sum over all bins; sum[Observed]!), than constraints==1.\n"
		" If they are equal by design, than constraints==0. The function imposes this condition based on the passed (and used)\n"
		" sample lengths, so it is possible to leave conditions==0.\n"
		" The calculated chi-square value is returned in &chsq-return (which must be a scalar). The determined degrees-of-freedom\n"
		" may be returned in the optional &df-return argument.\n"
	},

	{ "return", ascanf_return, 1, NOT_EOF_OR_RETURN,
		"return[x]: specify value to return from a list expression or procedure.\n"
		" This function behaves like its counterpart in Lisp: it does *not* halt the\n"
		" execution of the expression/procedure!\n"
	},
	{ "progn", ascanf_progn, AMAXARGS, NOT_EOF_OR_RETURN, "progn[expr1[,..,expN]]: value set by return[x]" },
	{ "for-to", ascanf_for_to, AMAXARGS, NOT_EOF_OR_RETURN, "for-to[init_expr,test_expr,expr1[,..,expN]]: value set by return[x]; $loop is set to <init_expr>" },
	{ "for-toMAX", ascanf_for_toMAX, AMAXARGS, NOT_EOF_OR_RETURN, "for-toMAX[init_expr,MaxCount,expr1[,..,expN]]: value set by return[x]; $loop is set to <init_expr> and increments by 1 until MaxCount-1" },
	{ "whiledo", ascanf_whiledo, AMAXARGS, NOT_EOF_OR_RETURN, "whiledo[test_expr,expr1[,..,expN]]: value set by return[x]" },
	{ "dowhile", ascanf_dowhile, AMAXARGS, NOT_EOF_OR_RETURN, "dowhile[expr1[,..,expN],test_expr]: value set by return[x]" },
	{ "break-loop", ascanf_break_loop, 1, NOT_EOF_OR_RETURN, 
		"break-loop or break-loop[flag]: break out of a looping construct by setting the <interrupt-processing> internal flag.\n"
		" This means that as soon the command is executed, all further processing of expressions inside the innermost loop\n"
		" is suspended. Processing resumes outside that innermost loop. The optional <flag> argument can be used to call this\n"
		" routine without effect (break-loop[0]). Always returns 1 when called inside a loop.\n"
	},
	{ "print", ascanf_print, AMAXARGS, NOT_EOF_OR_RETURN, "print[x[,..]]: returns first (and only) arg,\n\
\t\tvalue set by return[y]" },
	{ "Eprint", ascanf_Eprint, AMAXARGS, NOT_EOF_OR_RETURN, "Eprint[x[,..]]: prints on stderr" },
	{ "TBARprint", ascanf_TBARprint, AMAXARGS, NOT_EOF_OR_RETURN, "TBARprint[x[,..]]: prints in attached window's WM title bar" },
	{ "TBARprogress", ascanf_TBARprogress, 3, NOT_EOF_OR_RETURN, "TBARprogress[i,N[,step]]: shows progress (step i of N, with step) in window's WM titlebar" },
	{ "Dprint", ascanf_Dprint, AMAXARGS, NOT_EOF_OR_RETURN, "Dprint[[x,..]]: prints on $Dprint-file, separated by tabs, and without brackets &c." },
	{ "Doutput", ascanf_Doutput, AMAXARGS, NOT_EOF_OR_RETURN, "Doutput[[x,..]]: like Dprint[], but doesn't output a newline after each invocation." },
	{ "printf", ascanf_printf, AMAXARGS, NOT_EOF_OR_RETURN,
		"printf[target,format[,arglist]]: printf() like routine. <target> can be 1=stdout, 2=stderr, a stringpointer (`target)"
		" or an open file\n"
		" <arglist> can contain expressions evaluating to values or pointers-to-variables (which are printed as is),\n"
		" stringpointers (`stringvar; printed as string) and pointers-to-arrays. In that last case, all elements of\n"
		" the array are printed. It is the caller's responsability to provide enough printf() formatting fields in all cases.\n"
		" Values are printed as \"usual\" (format specified by *DPRINTF*), the printf() format-fields used should thus be those\n"
		" for string variables (%s)!!\n", 0, 1
	},
	{ "scanf", ascanf_scanf, AMAXARGS, NOT_EOF_OR_RETURN,
		"scanf[source,format,offset-or-targ1[,targs]]: scanf() like routine. <source> can be as the 1st arg. to printf[]\n"
		" <format> must be a stringpointer with the format argument, that may only contain %lf fields!!\n"
		" <offset-or-targ1> may be 1) a scalar number, 2) a stringpointer, or 3) a pointer to a scalar\n"
		" In the first 2 cases, it serves to specify a numerical offset to start parsing at, or a pattern to start\n"
		" parsing AFTER. Otherwise, it is taken as the 1st target variable. All <targ> arguments must be pointers to\n"
		" scalar variables. The function returns the number of variables read.\n"
	},
	{ "getenv", ascanf_getenv, 3, NOT_EOF_OR_RETURN,
		"getenv[name[,eval[,default]]]: searches the environment for a variable <name>,\n"
		" and returns the value it represents. The value is first tried as an\n"
		" ascanf expression. If that succeeds, the value returned is the result\n"
		" of the evaluation; upon failure, the variable's value string is returned.\n"
		" No evaluation is attempted when <eval> is False.\n"
		" <default> allows to set an optional default return value in case <name> isn't found.\n"
	},
	{ "setenv", ascanf_setenv, 2, NOT_EOF_OR_RETURN,
		"setenv[name[,value]]: sets or updates the environment variable <name>,\n"
		" and returns the new value it represents.\n"
	},
	{ "fopen", ascanf_fopen, 3, NOT_EOF_OR_RETURN,
		"fopen[`filename-variable,`\"mode-string\"]\n"
		" fopen[&fp,`filename-variable,`\"mode-string\"]: opens the file with the name associated with filename-variable,\n"
		" and assigns the resulting file-pointer to the same variable or <fp> in the 2nd form, and returns a pointer to it.\n"
		" The resulting file variable can afterwards be used as the first argument to printf[].\n"
		" \"mode-string\" determines the mode in which the file is opened: see the fopen(2)\n"
		" manpage. Upon failure, 0 is returned.\n"
	},
	{ "fflush", ascanf_fflush, 1, NOT_EOF_OR_RETURN,
		"fflush[`filename-variable]: flushes the buffer associated with the file\n"
		" fflush[n] with n=0,1,2 flushes stdin, stdout, stderr, respectively.\n"
		" fflush without arguments flushes stdout and stderr.\n"
		" Returns 0 upon success\n"
	},
	{ "fclose", ascanf_fclose, 1, NOT_EOF,
		"fclose[`filename-variable]: closes a file\n"
		" Returns 0 upon success\n"
	},
	{ "funlink", ascanf_unlink, 1, NOT_EOF,
		"funlink[path]: removes the named file (not a directory!).\n"
	},
	{ "File-Size", ascanf_fsize, 2, NOT_EOF_OR_RETURN,
		"File-Size[`filename-variable[,lines]]: determine the size of the specified file. When <lines> is True,\n"
		" returns the number of lines in the file instead.\n"
	},
#if defined(linux) || defined(__CYGWIN__)
	{ "File-Name", ascanf_fname, 1, NOT_EOF,
		"File-Name[`filename-variable]: retrieve the filename associated with an open file or file descriptor\n"
	},
#else
	{ "File-Name", ascanf_fname, 1, NOT_EOF,
		"File-Name[`filename-variable]: retrieve the filename associated with an open file or file descriptor\n"
		"NOT FUNCTIONAL on this platform!\n"
	},
#endif
	{ "ParseArguments", ascanf_ParseArguments, AMAXARGS, NOT_EOF_OR_RETURN,
		"ParseArguments[string[,string[,..]]]: parse string(s) for command line arguments, as in *ARGUMENTS*\n"
		" Arguments that modify set-specific settings apply to as-yet unused sets.\n"
	},
	{ "verbose", ascanf_verbose_fnc, AMAXARGS, NOT_EOF_OR_RETURN, "verbose[[x,..]]: turns ON verbosity for its scope" },
	{ "no-verbose", ascanf_noverbose_fnc, AMAXARGS, NOT_EOF_OR_RETURN, "no-verbose[[x,..]]: turns OFF verbosity for its scope" },
	{ "compile-noEval", ascanf_noEval, AMAXARGS, NOT_EOF_OR_RETURN, "compile-noEval[x,..]: compile and DON'T evaluate the argument(s)" },
	{ "compile", ascanf_compile, AMAXARGS, NOT_EOF_OR_RETURN, "compile[x,..]: compile and DO evaluate the argument(s)" },
	{ "call", ascanf_call, AMAXARGS, NOT_EOF_OR_RETURN,
		"call[<fptr>[,..]]: invoke the function or procedure pointed to by <fptr>, passing it the optional remaining\n"
		" arguments as its arguments. Returns the returned value.\n"
	},
	{ "eval", ascanf_eval, AMAXARGS, NOT_EOF_OR_RETURN,
		"eval[expression[,compiled[,args]]]: evaluate <expression>, which must be a stringpointer. <compiled> is a flag\n"
		" that indicates whether or not the expression should first be compiled to improve performace. It is possible to\n"
		" specify additional arguments; these are available in the $ array. The potential main interest of this function is to\n"
		" write and modify code \"on the fly\". But don't try to redefine or delete code while it is being exectuted!!\n"
	},
	{ "Apply2Array", ascanf_Apply2Array, AMAXARGS, NOT_EOF_OR_RETURN,
		"Apply2Array[<sourceArray>,<targetArray>,[pass,]<method>[,...]]: applies <method> to all elements of <sourceArray>\n"
		" in succesion, with $loop set to the current index. The first argument to <method> is sourceArray[$loop] unless <pass>\n"
		" is False. Any arguments specified after <method> are passed as additional arguments. The result of the calculation is\n"
		" stored in <targetArray>, or in <sourceArray> when <targetArray> is a negative value (and discarded when it is 0).\n"
		" Note that $loop as additional argument (i.e. in the Apply2Array scope, NOT in the <method> scope!!) is evaluated\n"
		" in the calling context, and will thus not vary with the <sourceArray> elements. To pass the current element as\n"
		" an argument to the called <method>, use &$loop instead (it is thus not possible to pass a pointer to $loop as an\n"
		" argument, which wouldn't be useful anyway). Apply2Array returns the last value calculated (set via return[] in case\n"
		" <method> is a user-defined procedure). <method> may be a pointer to a function, a procedure, or an array.\n"
		"Apply2Array[stride[,next_only]]: sets the stride used for (the) subsequent invocation(s). The default is 1, meaning all elements\n"
		" are 'visited' and in the order from 0 to N-1. For stride==-2, only every other element is visited, backwards from\n"
		" N-1. And so forth. Stride==0 is silently corrected to stride==1.\n"
	},
	{ "IDict", ascanf_IDict_fnc, AMAXARGS, NOT_EOF_OR_RETURN,
		"IDict[[x,..]]: stores in the internal dictionary the variables, arrays, etc. that are defined in its scope\n"
		" These internal variables are stored in the dictionary that also stores the automatic variables.\n"
		" They are not exported when an XGraph file is saved. Internal variables can be masked (shadowed) by \"external\" variables.\n"
		" Specific internal variables can be deleted with IDict[ Delete[<varname>] ]\n"
		" Variables created (automatically or not) in internal mode while compiling a procedure are local to that procedure!! (20061117)\n"
		" (If $AllowProcedureLocals is True.)\n"
		" Note that the verbose[] command does not set verbose mode while compiling in internal dictionary mode!\n"
	},
	{ "GLOBAL", ascanf_global_fnc, AMAXARGS, NOT_EOF_OR_RETURN,
		"GLOBAL[[x,...]]: switches off local variable declaration for its scope (unsets $AllowProcedureLocals temporarily)\n"
	},
	{ "matherr", ascanf_matherr_fnc, AMAXARGS, NOT_EOF_OR_RETURN, "matherr[[x,..]]: turns on matherr verbosity for its scope" },
	{ "comment", ascanf_comment_fnc, AMAXARGS, NOT_EOF_OR_RETURN, "comment[[x,..]]: include scope in Info box" },
	{ "popup", ascanf_popup_fnc, AMAXARGS, NOT_EOF_OR_RETURN, "popup[[x,..]]: popup scope's output in a dialog window" },
	{ "last_popup", ascanf_last_popup_fnc, AMAXARGS, NOT_EOF_OR_RETURN, "last_popup[[x,..]]: re-popup the previous' popup[] output" },
	{ "uniform", ascanf_uniform, 2, NOT_EOF_OR_RETURN, "uniform[av,stdv]: random number in a uniform distribution" },
	{ "abnormal", ascanf_abnormal, 2, NOT_EOF_OR_RETURN, "abnormal[av,stdv]: random number in an abnormal distribution" },
	{ "normal", ascanf_normal, 3, NOT_EOF_OR_RETURN, "normal[n,av,stdv]: random number in normal distribution number <n>" },

	{ "angsize1", ascanf_angsize1, 4, NOT_EOF_OR_RETURN, "angsize1[x,y[,base[,thres]]] angle subtended by object of radius x at distance y in 0..2PI" },
	{ "angsize2", ascanf_angsize2, 4, NOT_EOF_OR_RETURN, "angsize2[x,y[,base]] angle subtended by object of radius x at distance y in 0..PI (other algorithm)" },
	{ "(powX)", ascanf_powXFlag, 1, NOT_EOF_OR_RETURN, "(powX) value of powXFlag (-powx option)" },
	{ "(powY)", ascanf_powYFlag, 1, NOT_EOF_OR_RETURN, "(powY) value of powYFlag (-powy option)" },
	{ "barBase", ascanf_barBase, 2, NOT_EOF_OR_RETURN, "barBase[dum,[val]]: returns and/or sets the base for barplots"},
	{ "ActiveWin", ascanf_ActiveWin, 1, NOT_EOF,
		"ActiveWin: returns an id of the currently active window, or 0\n"
		" ActiveWin[new]: makes a new window active. new=-1 makes the root window or the calling terminal's window active.\n"
		" new=0 deactivates all. The return value is still the id of the window active at the time of the call.\n"
	},
	{ "ActiveWinPrinting", ascanf_ActiveWinPrinting, 1, NOT_EOF,
		"ActiveWinPrinting: returns an integer describing where the current window is sending its output:\n"
		" 0 (default): to the screen\n"
		" 1: to PostScript (truly printing...)\n"
		" 2: to an XGraph dump.\n"
	},
	{ "ActiveWinWidth", ascanf_ActiveWinTWidth, 1, NOT_EOF, "ActiveWinWidth: returns the currently active window's width, or 0"},
	{ "ActiveWinHeight", ascanf_ActiveWinTHeight, 1, NOT_EOF, "ActiveWinHeight: returns the currently active window's height, or 0"},
	{ "ActiveWinXMin", ascanf_ActiveWinXMin, 1, NOT_EOF_OR_RETURN,
		"ActiveWinXMin: returns the currently active window's current low X setting, or 0"
	},
	{ "ActiveWinXMax", ascanf_ActiveWinXMax, 1, NOT_EOF_OR_RETURN,
		"ActiveWinXMax: returns the currently active window's current high X setting, or 0"
	},
	{ "ActiveWinYMin", ascanf_ActiveWinYMin, 1, NOT_EOF_OR_RETURN,
		"ActiveWinYMin: returns the currently active window's current low Y setting, or 0"
	},
	{ "ActiveWinYMax", ascanf_ActiveWinYMax, 1, NOT_EOF_OR_RETURN,
		"ActiveWinYMax: returns the currently active window's current high Y setting, or 0"
	},
	{ "ActiveWinXLabel", ascanf_ActiveWinXLabel, 2, NOT_EOF_OR_RETURN,
		"ActiveWinXLabel[`ret[,`new]]: returns a pointer to the old (transformed) XLabel string (in ret)\n,"
		" optionally setting it to a new value. <ret> may be 0.\n"
		" <new> may contain opcodes referring to the raw XLabel (%X) and/or raw YLabel (%Y).\n"
	},
	{ "ActiveWinYLabel", ascanf_ActiveWinYLabel, 2, NOT_EOF_OR_RETURN,
		"ActiveWinYLabel[`ret[,`new]]: returns a pointer to the old (transformed) YLabel string (in ret)\n,"
		" optionally setting it to a new value. <ret> may be 0.\n"
		" <new> may contain opcodes referring to the raw XLabel (%X) and/or raw YLabel (%Y).\n"
	},
	{ "ActiveWinAxisValues", ascanf_ActiveWinAxisValues, 4, NOT_EOF_OR_RETURN,
		"ActiveWinAxisValues[which,raw,&values[,&labelled]]: retrieve the ticks/gridlines currently\n"
		" shown at the <which> axis (0= X, 1= Y, 2= Intensity), and store the result in the <values>\n"
		" array. This must be an array of doubles. If specified, a flag in the <labelled> array indicates\n"
		" whether or not a textual value label is shown at any given grid position. The arrays are resized\n"
		" as necessary. The raw argument is currently ignored. When an axis is not shown, or has no grids/\n"
		" ticks on it, the routine will return 0, but the values array will contain a single NaN. Otherwise,\n"
		" the routine returns the number of items in the values array.\n"
	},
	{ "ActiveWinDataWin", ascanf_DataWin, 9, NOT_EOF_OR_RETURN,
		"ActiveWinDataWin[apply[,flag,minX,flag,minY,flag,maxX,flag,maxY]]: define a rectangular clipping region outside of which\n"
		" no data should be shown, and impose that region on the current graph's axes. This is independent from\n"
		" any other settings related to the axes' bounds, although these are updated to show the current settings.\n"
		" Each of the four limits may be a NaN in order for it to be ignored while drawing; the individual <flag>\n"
		" arguments indicate whether or not to update the following limit. Only the <apply> argument is obligatory;\n"
		" each <flag,limit> pair is optional, but the order of interpretation is fixed (i.e. all must be present in order\n"
		" to specify <maxY>). The sign of the <apply> argument determines the \"user co-ordinates\" feature setting:\n"
		" when negative, padding will be applied to the specified clipping region, when positive, the region will be\n"
		" used 'as is'.\n"
	},
	{ "ActiveWinDataWinScroll", ascanf_DataWinScroll, 3, NOT_EOF_OR_RETURN,
		"ActiveWinDataWinScroll[X,Y[,apply]]: scrolls the previously defined DataWin over <X> and/or <Y>,\n"
		" potentially (de)activating it depending on the <apply> argument\n"
	},
	{ "WaitEvent", ascanf_WaitEvent, 2, NOT_EOF_OR_RETURN,
		"WaitEvent[[type[,`msg]]]: wait for an event to occur in the currently active window\n"
		" When type, a string pointer, is not specified, wait for any event, otherwise\n"
		" for an event of the specified type. Currently supported:\n"
		"   type=`\"key\": wait for a keypress.\n"
		" The optional 2nd argument can be a stringpointer to a message to be displayed in the waiting window's titlebar.\n"
	},

	{ "QueryPointer", ascanf_QueryPointer, 1, NOT_EOF,
		"QueryPointer: returns 1 and updates $PointerPos if the pointer\n"
		"is within the currently active window, or 0 otherwise (without changing $PointerPos)\n"
	},
	{ "Find_Point", ascanf_Find_Point, 6, NOT_EOF_OR_RETURN,
		"Find_Point[x,y,&set_return,&idx_return[,&X_return[,&Y_return]]]: finds the set with a point closest to (x,y)\n"
		" When (x,y) is the current pointer position, this finds the same point as interactive point finding would\n."
		" The set number is returned in <set_return>, and the index in <idx_return>. The other options will return\n"
		" the found point's X and Y co-ordinates. The function returns 0 when no point is found.\n"
		" When x (y) is NaN or Inf, the search is done only on the closest y (x) co-ordinate.\n"
		" When $Find_Point_exhaustive is false, uses the precision as defined by the current window/screen resolution.\n"
	},
	{ "FitOnce", ascanf_setFitOnce, 1, NOT_EOF_OR_RETURN,
		"FitOnce or FitOnce[val]: set the FitOnce (autoscaling in X & Y) field of a window\n"
		" Without arguments, returns true if the window has some form of (permanent) autoscaling\n"
	},
	{ "FitBounds", ascanf_FitBounds, 2, NOT_EOF_OR_RETURN,
		"FitBounds[x[,y]]: fit x and/or y (booleans) now. Only works when not already fitting.\n"
		" Should probably be used only in a *DRAW_AFTER* with $Really_DRAW_AFTER[1].\n"
	},
	{ "QuickMode", ascanf_setQuick, 1, NOT_EOF_OR_RETURN,
		"QuickMode[[val]]: set the QuickMode (use transformed values) field of a window, or return its setting"
	},
	{ "DPShadow", ascanf_DiscardedShadows, 1, NOT_EOF_OR_RETURN,
		"DPShadow[[val]]: set the DPShadow option (discarded points are shown as shadows, or return its setting\n"
		" If you need to use curve_len[] immediately after exiting QuickMode, this variable must be set.\n"
		" See the manpage entry to curve_len[] for more details.\n"
	},
	{ "ClearWindow", ascanf_ClearWindow, 4, NOT_EOF_OR_RETURN,
		"ClearWindow[[loX,loY,hiX,hiY]]: clears the currently active window, optionally inside the given rectangle"
	},
	{ "redraw", ascanf_setredraw, 1, NOT_EOF, "redraw or redraw[return[<exp>]]: set the redraw field of a window"},
	{ "RedrawNow", ascanf_RedrawNow, 2, NOT_EOF_OR_RETURN,
		"RedrawNow or RedrawNow[return[<exp>[,all]]]: redraw immediately.\n"
		" If <exp> evaluates to True or is missing, silenced mode is deactivated for this one redraw.\n"
		" <all> determines whether all currently open windows are redrawn.\n"
	},
	{ "DrawTime", ascanf_DrawTime, 1, NOT_EOF,
		"DrawTime: time elapsed since start of drawing (real time) or DrawTime[x] (system time)"
	},
	{ "Print", ascanf_Print, 2, NOT_EOF_OR_RETURN, "Print: print the current window"},
	{ "PrintOK", ascanf_PrintOK, 1, NOT_EOF, "PrintOK: like the -printOK argument: print the current window and close it."},
	{ "raw_display", ascanf_raw_display, 2, NOT_EOF,
		"raw_display[<newval>[,noredraw]]: (un)set raw_display switch; returns old value"},
	{ "Silenced?", ascanf_Silenced, 2, NOT_EOF,
		"Silenced?[[silence[,noredraw]]]: returns 1 when drawing is silenced (no output - e.g. when determining scale), -1 when no (active) window exists\n"
		" Can take an optional argument to (un)set silenced mode (silenced also ignores all non-user generated events)\n"
	},
	{ "Fitting?", ascanf_Fitting, 1, NOT_EOF,
		"Fitting?: returns 1 when determination of bounds is taking place (auto scaling),\n"
		" -1 when auto-fit is not set, or when no (active) window exists\n"
		" Useful only in DRAW_BEFORE or DRAW_AFTER statements, probably.\n"
	},
	{ "DumpProcessed?", ascanf_DumpProcessed, 1, NOT_EOF,
		"DumpProcessed?: returns 1 when dumping processed values, -1 when no (active) window exists or not dumping.\n"
	},
	{ "RecurseLevel?", ascanf_EventLevel, 1, NOT_EOF,
		"RecurseLevel?: the recurse level at which drawing currently takes place; -1 when no (active) window exists\n"
	},
	{ "SetCycleSet", ascanf_SetCycleSet, 2, NOT_EOF_OR_RETURN,
		"CycleSet[new[,flush]]: returns the set currently \"selected\" with the left/right cursor keys,\n"
		" possibly setting it to a new value.\n"
	},
	{ "CycleSet", ascanf_CycleSet, AMAXARGS, NOT_EOF_OR_RETURN, "CycleSet[#sets[,flush,return[<exp>]]]: cycle the displayed set #sets up or down"},
	{ "CycleDrawnSets", ascanf_CycleDrawnSets, AMAXARGS, NOT_EOF_OR_RETURN, "CycleDrawnSets[#sets[,flush,return[<exp>]]]: cycle the displayed collection of sets #sets up or down"},
	{ "CycleGroup", ascanf_CycleGroup, AMAXARGS, NOT_EOF_OR_RETURN, "CycleGroup[#grps[,flush,return[<exp>]]]: cycle the displayed group #grps up or down"},
	{ "CycleFile", ascanf_CycleFile, AMAXARGS, NOT_EOF_OR_RETURN, "CycleFile[#fls[,flush,return[<exp>]]]: cycle the displayed file #fls up or down"},
	{ "Reverse", ascanf_Reverse, AMAXARGS, NOT_EOF_OR_RETURN, "Reverse[[flush,return[<exp>]]]: swap the displayed sets (= ^S)"},
	{ "All", ascanf_All, AMAXARGS, NOT_EOF_OR_RETURN, "All[[flush,return[<exp>]]]: swap displaying all sets with the displayed sets (= ^A)"},
	{ "newGroup", ascanf_newGroup, 2, NOT_EOF_OR_RETURN,
		"newGroup[set[,new]]: return (and set) a set's new group field, which determines the grouping structure\n"
	},
	{ "set's-Group", ascanf_Group, AMAXARGS, NOT_EOF_OR_RETURN,
		"set's-Group[[n]]: return set <n>'s group number, or the total number of groups."
	},
	{ "set's-File", ascanf_File, AMAXARGS, NOT_EOF_OR_RETURN,
		"set's-File[[n[,newval]]]: return (and set) set <n>'s file number, or the total number of file."
	},
	{ "Group-Sets", ascanf_GroupSets, 1, NOT_EOF_OR_RETURN,
		"Group-Sets[[n]]: return the number of sets in the current set's group, or in set <n>'s group"
	},
	{ "File-Sets", ascanf_FileSets, 1, NOT_EOF_OR_RETURN,
		"File-Sets[[n]]: return the number of sets in the current set's file-set, or in set <n>'s file-set"
	},
	{ "FirstDrawn", ascanf_FirstDrawn, 1, NOT_EOF_OR_RETURN,
		"FirstDrawn[[set]]: When an argument is given, returns True when the specified set is the first to be drawn\n"
		" in the active window. When no argument is given, returns the index of the first drawn set.\n"
	},
	{ "LastDrawn", ascanf_LastDrawn, 1, NOT_EOF_OR_RETURN,
		"LastDrawn[[set]]: When an argument is given, returns True when the specified set is the last to be drawn\n"
		" in the active window. When no argument is given, returns the index of the last drawn set.\n"
	},
	{ "Marked", ascanf_Marked, AMAXARGS, NOT_EOF_OR_RETURN, "Marked[[flush,return[<exp>]]]: show only all marked sets (= M); returns the number of sets"},
	{ "DrawSet", ascanf_DrawSet, 2, NOT_EOF_OR_RETURN,
		"DrawSet[set[,val]]: whether or not a set is drawn: returns the (old) value.\n"
		" DrawSet[-1,1] draws all sets, DrawSet[-1,0] undraws all sets, DrawSet[-1,-1] swaps\n"
	},
	{ "MarkSet", ascanf_MarkSet, 2, NOT_EOF_OR_RETURN, "MarkSet[set[,val]]: whether or not a set is marked: returns the (old) value."},
	{ "HighlightSet", ascanf_HighlightSet, 2, NOT_EOF_OR_RETURN, "HighlightSet[set[,val]]: whether or not a set is highlighted: returns the (old) value."},
	{ "RawSet", ascanf_RawSet, 2, NOT_EOF_OR_RETURN, "RawSet[set[,val]]: whether or not a set is to be \"raw_displayed\": returns the (old) value."},
	{ "FloatSet", ascanf_FloatSet, 2, NOT_EOF_OR_RETURN,
		"FloatSet[set[,val]]: whether or not a set is to be ignored during auto-scaling: returns the (old) value."},
	{ "AdornInt", ascanf_AdornInt, 3, NOT_EOF_OR_RETURN,
		"AdornInt[set[,int[,noredraw]]]: the interval at which a set is marked: returns the (old) value.\n"
		" If <nodredraw> is True, then a redraw is not scheduled.\n"
	},
	{ "PlotInt", ascanf_PlotInt, 3, NOT_EOF_OR_RETURN,
		"PlotInt[set[,int[,noredraw]]]: the interval at which datapoints are plotted: returns the (old) value."},
	{ "Markers", ascanf_Markers, 4, NOT_EOF_OR_RETURN,
		"Markers[set[,has_marks[,type[,size]]]]: Returns 1 if the set has markers, possibly setting it to the new <has_marks>.\n"
		"<type>: 0=markers; 1=pixel; 2=blob\n"
		" The <size> argument sets the type=0 marker's size: set to NaN to use global size (-PSm)\n"
	},
	{ "ErrorType", ascanf_error_type, 3, NOT_EOF_OR_RETURN,
		"ErrorType[set[,type[,fit]]]: Returns the set's error type, possibly setting it to the new <type>.\n"
		" Where <type> is 0=no; 1=bars; 2=triangles; 3=regions; 4=vectors; 5=intensities; 6=msizes\n"
		" The <fit> argument can be used to prevent the rescaling (FitOnce) that is done in certain cases\n"
	},
	{ "RedrawSet", ascanf_RedrawSet, 2, NOT_EOF_OR_RETURN,
		" RedrawSet[set[,immediate]]: cause a possibly immediate redraw of <set> in all windows displaying it.\n"
		" Returns the number of windows redrawn (<immediate>== True) or set to be redrawn\n"
		" RedrawSet[-1,1] causes an immediate redraw of all windows wanting one\n"
	},
	{ "SkipSetOnce", ascanf_SkipSetOnce, 2, NOT_EOF_OR_RETURN,
		" SkipSetOnce[set[,val]]: ignore this set from the moment the command is issued (when val==True)\n"
		" Processing stops at that moment, and the set is not drawn at all.\n"
	},
	{ "ShiftSet", ascanf_ShiftSet, 4, NOT_EOF_OR_RETURN,
		"ShiftSet[set,direction[,extreme=0[,redraw=1]]]: shifts <set> in <direction> (-1: left; 1: right);\n"
		" to start or end when <extreme>. A global redraw is initiated when <redraw>.\n"
	},
	{ "LinkedSet?", ascanf_LinkedSet, 3, NOT_EOF_OR_RETURN,
		"LinkedSet?[set[,&linked-to|link-to,doit]]: whether a set is linked to another. The number of the source set is\n"
		" optionally returned in the &linked-to argument.\n"
		" The second call form, [set, link-to, doit], links a set to another, existing set, if <doit> is True.\n"
		" (unlinking is not possible, currently.)\n"
	},
	{ "CheckAssociations_AND", ascanf_CheckAssociations_AND, AMAXARGS, NOT_EOF_OR_RETURN, "CheckAssociations_AND[set[,val,...]]: check if <set> has all specified value(s) associated with it. Without values, returns the number of associations."},
	{ "CheckAssociations_OR", ascanf_CheckAssociations_OR, AMAXARGS, NOT_EOF_OR_RETURN, "CheckAssociations_OR[set[,val,...]]: check if <set> has one of the specified value(s) associated with it. Without values, returns the number of associations."},
	{ "CheckAssociation#_OR", ascanf_CheckAssociation_OR, AMAXARGS, NOT_EOF_OR_RETURN, "CheckAssociation#_OR[set,ass_nr,val1[,val,...]]: check if <set> association nr <ass_nr> equals one of the specified value(s)."},
	{ "getAssociation", ascanf_getAssociation, 2, NOT_EOF_OR_RETURN,
		"getAssociation[set[,n]]: return association <n> of set <set>.\n"
		" When <n> is missing, return the number of associations.\n"
		" When <n> points to an array, the current list of associations is returned in that array.\n"
	},
	{ "Associate", ascanf_Associate, AMAXARGS, NOT_EOF_OR_RETURN,
		"Associate[set,ass_nr[,val,...]]: add association <val,..> to set <set> from ass_nr onwards,\n"
		" With val,.. specified, pass <ass_nr>=-1 to append after last association\n"
		" Without values specified, set # of associations to <ass_nr> or remove all associations (<ass_nr>==0)\n"
	},
	{ "ValCat_X?", ascanf_ValCat_X, 3, NOT_EOF_OR_RETURN,
		"ValCatX?[val,exact?[,<return_p>]: checks whether an (exact) X category for <val> exists.\n"
		" If exact?=False, tries to find the closest match. If a return-value pointer is given,\n"
		" the value found (<val> if exact, else maybe another) or (if none found) <val> is returned in it\n"
		" If it is a stringpointer, the category string (or <val>, if no category is found) is stored in it also.\n"
	},
	{ "ValCat_Y?", ascanf_ValCat_Y, 3, NOT_EOF_OR_RETURN,
		"ValCatY?[val,exact?]: checks whether an (exact) Y category for <val> exists.\n"
		" If exact?=False, tries to find the closest match. If a return-value pointer is given,\n"
		" the value found (<val> if exact, else maybe another) or (if none found) <val> is returned in it\n"
		" If it is a stringpointer, the category string (or <val>, if no category is found) is stored in it also.\n"
	},
	{ "ValCat_I?", ascanf_ValCat_I, 3, NOT_EOF_OR_RETURN,
		"ValCatI?[val,exact?]: checks whether an (exact) I (intensity) category for <val> exists.\n"
		" If exact?=False, tries to find the closest match. If a return-value pointer is given,\n"
		" the value found (<val> if exact, else maybe another) or (if none found) <val> is returned in it\n"
		" If it is a stringpointer, the category string (or <val>, if no category is found) is stored in it also.\n"
	},
	{ "Arrays2ValCat", ascanf_Arrays2ValCat, 3, NOT_EOF_OR_RETURN,
		"Arrays2ValCat[which,&categories[,&values]]: redefine the given value categories.\n"
		" <axis>: 0=ValCat_X, 1=ValCat_Y, 2=ValCat_I. <categories> must be a pointer to an array containing\n"
		" the strings to be used as category labels. The optional <values> must be a pointer to an array with\n"
		" the values to be associated with the labels in <categories>; if it is not specified, the string addresses\n"
		" in <categories> will be used instead. If both arrays are specified, the smallest of the 2 determines the\n"
		" (maximum) number of categories to be defined. The function returns the number of successful associations.\n"
	},
	{ "XSync", ascanf_XSync, 2, NOT_EOF, "XSync[0/1] or XSync[0/1,return[<exp>]]: flush the X-window, (un)setting synchro mode."},
	{ "GetRGB", ascanf_GetRGB, 1, NOT_EOF,
		"GetRGB[pixel]: get the RGB values currently associated with <pixel>\n"
		" Values are stored in $IntensityRGBValues, as intensities (from 0 to 1), regardless\n"
		" of the $AllowGammaCorrection setting.\n"
	},
	{ "GetIntensityRGB", ascanf_GetIntensityRGB, 1, NOT_EOF,
		"GetIntensityRGB[I]: get the RGB values currently associated with intensity <I>\n"
		" Values are stored in $IntensityRGBValues, as intensities (0-1), regardless of\n"
		" the $AllowGammaCorrection setting.\n"
	},

	{ "radix", ascanf_radix, 2, NOT_EOF_OR_RETURN, "radix[[val,[set_redraw_and_flush]]]: returns (new) radix value"},
	{ "radix_offset", ascanf_radix_offset, 2, NOT_EOF_OR_RETURN,
		"radix_offset[[val,[set_redraw_and_flush]]]: returns (new) radix offset value\n"
		" 0 (default): 0 degrees rightwards; 90: 0 degrees is upwards (for radix==360)\n"
	},
	{ "polarBase", ascanf_radix, 2, NOT_EOF_OR_RETURN, "see radix[]"},
	{ "compress", ascanf_compress, 3, NOT_EOF_OR_RETURN, "compress[x,C[,F]]: x^F/(|x^f|+C^f) - default F=1" },
	{ "lowpass", ascanf_lowpass, 4, NOT_EOF_OR_RETURN, "lowpass[x,tau,mem_index[,dt]] - mem_index is index into MEM array" },
	{ "nlowpass", ascanf_nlowpass, 4, NOT_EOF_OR_RETURN, "nlowpass[x,tau,mem_index[,dt]]: normalised - mem_index is index into MEM array" },
	{ "nlowpass*", ascanf_nlowpassB, 4, NOT_EOF_OR_RETURN, "nlowpass*[x,tau,fac,mem_index[,dt]]: normalised \"special\" lowpass filter - mem_index is index into MEM array" },
	{ "nlowpass**", ascanf_nlowpassC, 4, NOT_EOF_OR_RETURN, "nlowpass**[x,tau,fac,mem_index[,dt]]: normalised \"special\" lowpass filter - mem_index is index into MEM array" },
	{ "shunt", ascanf_shunt, 5, NOT_EOF_OR_RETURN, "shunt[x,y,C,tau,mem_index[,dt]]: normalised shunting inhibition (x/(C+y)) - mem_index is index into MEM array" },

	/* spline functions were here */

	/* savgol functions were here */

	/* (r)fftw and ascanf_convolve functions were here */

	{ "Boing", ascanf_Boing, 1, NOT_EOF_OR_RETURN, "Boing or Boing[<volume-percentage>]: ring the bell..\n" },
	{ "CursorCross", ascanf_CursorCross, 1, NOT_EOF_OR_RETURN, "CursorCross or CursorCross[<val>]: as the -Cross commandline option\n" },
	{ "CheckEvent", ascanf_CheckEvent, 1, NOT_EOF, "CheckEvent or CheckEvent[interval]: check and handle event and/or change checking interval."},
	{ "debug", ascanf_setdebug, 2, NOT_EOF, "debug[0/1[,level]]: specify debugging flags."},
	{ "ClearReadBuffer", ascanf_ClearReadBuffer, 0, NOT_EOF,
		"ClearReadBuffer: clear the window(s)' read buffer, as a redraw or the Escape key would\n"
		" It returns the previous value of $ReadBufVal, which is also set to NaN.\n"
	},
	{ "PopReadBuffer", ascanf_PopReadBuffer, 0, NOT_EOF,
		"PopReadBuffer: pop the last character from the window(s)' read buffer.\n"
		" It returns the previous value of $ReadBufVal, which is set to new value,\n"
		" or NaN when the new buffer doesn't represent a numerical value.\n"
	},
	{ "MaxArguments", ascanf_MaxArguments, 1, NOT_EOF, "MaxArguments or MaxArguments[N]: query or set the maximal number of arguments allowed."},
	{ "SHelp", ascanf_SHelp, 1, NOT_EOF_OR_RETURN,
		"SHelp[name]: looks for <name> in the function, variable and procedure tabels,\n"
		" and prints the syntax if found. Do IDict[SHelp[name]] to search in the internal table!\n"
		" Called from Python, the syntax would be SHelp('name'[,isInternal]), i.e. the query must be a quoted string\n"
	},
	{ "2Dindex", ascanf_2Dindex, 4, NOT_EOF, "2Dindex[x,y,Nx,Ny]: index a 1D array as an Nx x Ny 2D array" },
	{ "nDindex", ascanf_nDindex, AMAXARGS, NOT_EOF_OR_RETURN,
		"nDindex[x,y[,z,..],Nx,Ny[,Nz,..]]: index a 1D array as an arbitrary dimension array\n"
		" nDindex[&array,x[,y,..][,Nx,Ny,..][,newval|default]]: read (or change to <newval>) an element of the array.\n"
		" nDindex[`string]: return a hash value for <string>.\n"
		" nDindex[return_default[,next_only]]: interpret the <newval> argument as <default> instead. Meaning, if an\n"
		"         invalid element is requested, return the <default> value instead of an error. Useful for procedure args.\n"
		" This function can also be invoked by its alias, @.\n"
	},
	{ "name", ascanf_name, 1, NOT_EOF,
		"name[<ptr>]: returns the name of the object that <ptr> points to. If not a pointer, it\n"
		" returns a string representation of the argument.\n"
	},
	{ "CheckPointers", ascanf_CheckPointers, AMAXARGS, NOT_EOF_OR_RETURN, "CheckPointers[p1[,p2,...]]: check the validity of the arguments as valid ascanf pointer to ascanf_variables. Returns the number of valid pointers."
	},
	{ "AccessHandler", ascanf_AccessHandler, 8, NOT_EOF_OR_RETURN,
			"AccessHandler[&variable,&target[,<par1>[,<par2>[use,value[,dump[,change]]]]]]: associates <target> with <variable>. When an assignment is made\n"
			" to <variable>, this is registered through <target>, depending on target's type, and the optional par1 and par2:\n"
			"  target type=variable: set it to <par1> or to 1 otherwise\n"
			"             =array: set element <par1> (default the last accessed) to <par2> (default 1)\n"
			"             =procedure: call the procedure (no values can be passed)\n"
			"             =function: call function[<par1>,<par2>] (default 1,0)\n"
			" The <use,value> argument pair selects a specific <value> when the handler should be called (when <use>==1).\n"
			"     use==2: handler called when <value> is larger than <variable>'s value\n"
			"     use==-2: handler called when <value> is smaller than <variable>'s value\n"
			" The <dump> argument causes some elementary trace-back to be printed before the accesshandler is called.\n"
			" The <change> argument causes the handler to be only invoked when the variable's value changes.\n"
			" A pointer to the previous accesshandler is returned.\n"
	},
	{ "$HandlerPars", ascanf_Variable, 8, _ascanf_array,
		"$HandlerPars[8]: The 2 arguments (val1,val2) passed to AccesHandler (NaN=not passed);\n"
		" a pointer to the variable that was just accessed; the 4rd and 5th element contain\n"
		" the new and old values of this variable; the 6th element contains the\n"
		" index of the accessed array's element, if relevant ($HandlerPars[5]>= 0)\n"
		" The 7th element contains a pointer to the handler itself, the 8th the level at which the access occurred.\n",
		0, 0, 0, 0, 0, 0, 0.0, &HandlerParameters[0], NULL, NULL, NULL, NULL, 0, 0, 8
	},
	{ "Load_Module", ascanf_LoadDyMod, 2, NOT_EOF_OR_RETURN,
		"Load_Module[\"modulename\"[,\"flags\"]]: equivalent to the *LOAD_MODULE* opcode.\n"
	},
	{ "Unload_Module", ascanf_UnloadDyMod, 2, NOT_EOF_OR_RETURN,
		"Unload_Module[\"modulename\"[,\"flags\"[,force]]]: equivalent to the *UNLOAD_MODULE* opcode.\n"
	},
	{ "Reload_Module", ascanf_ReloadDyMod, 2, NOT_EOF_OR_RETURN,
		"Reload_Module[\"modulename\"[,\"flags\"[,force]]]: equivalent to the *RELOAD_MODULE* opcode.\n"
	},

	{ "titleText", ascanf_titleText, 3, NOT_EOF_OR_RETURN,
		"titleText[[idx[,&return_string,[`new]]]]: return and/or set the global titleText.\n"
		" For idx==0 the default titleText (-t) is used, for idx==1, the alternate titleText (-T).\n"
		" If &return_string is omitted, a static internal variable is used.\n", 0, 1
	},
	{ "SetTitle", ascanf_SetTitle, 4, NOT_EOF_OR_RETURN,
		"SetTitle[[idx[,&return_string[,`new[,parse?]]]]]: return and/or set the title for the current or idx'th set\n"
		" If &return_string is omitted, a static internal variable is used.\n"
		" If parse? is given and True, the returned string is parsed for opcodes.\n"
		, 0, 1
	},
	{ "SetName", ascanf_SetName, 4, NOT_EOF_OR_RETURN,
		"SetName[[idx[,&return_string,[`new[,parse?]]]]]: return and/or set the legend-entry (name) for the current or idx'th set\n"
		" If &return_string is omitted or not a possible string variable, a static internal variable is used.\n"
		" If `new is given and a valid string variable, the setName is updated accordingly.\n"
		" If parse? is given and True, the returned string is parsed for opcodes.\n"
		, 0, 1
	},
	{ "SetInfo", ascanf_SetInfo, 3, NOT_EOF_OR_RETURN,
		"SetInfo[[idx[,&return_string[,`new_string]]]]: return the current set_info for the current or idx'th set\n"
		" If &return_string is omitted, a static internal variable is used.\n"
		" If `new_string is given, store the string it contains in the set's set_info.\n"
		" NB: return_string is *not* touched when there is no set info!\n"
		, 0, 1
	},
	{ "SetColumnLabels", ascanf_SetColumnLabels, 5, NOT_EOF_OR_RETURN,
		"SetColumnLabels[[idx[,column,&return_string,[`new[,replace?]]]]]: return and/or set the column labels (*LABELS*) for the current or idx'th set\n"
		" If the set doesn't have labels associated, and <new> is absent or 0, returns the global labels.\n"
		" If column < 0, return (or replace) all labels, else only for the specified column.\n"
		" If &return_string is omitted or not a possible string variable, a static internal variable is used.\n"
		" If `new is given and a valid string variable, the column labels are updated accordingly.\n"
		" If replace? is given and True, any existing labels are discarded (= *LABELS* new).\n"
		" NB: return_string is *not* touched when there are no column labels!\n"
		, 0, 1
	},
	{ "SetOverlap", ascanf_SetOverlap, 6, NOT_EOF_OR_RETURN,
		"SetOverlap[set1,set2,raw?,statbin1,statbin2[,&weight]]: calculate the overlaps between the 2 given sets\n"
		" as shown in the legend by the -overlap functionality. The function returns a simple representation of\n"
		" this overlap/closeness (1 for perfect overlap, 0 for none at all). If this value is positive, statsbin1 and statsbin2\n"
		" contain the data that was calculated pointwise; the average and standard deviation are the values shown in\n"
		" the legend. The statsbin variables must either point to existing $SS_StatsBin variables, or refer to a valid\n"
		" ascanf statsbin slot. The optional <weight> argument will return the average weight used in the calculations.\n"
		" The <raw?> argument is a Boolean that determines whether or not to use the set's raw data, or the results of\n"
		" the currently active transformations.\n"
		" You may have to ensure that sets were recently drawn in order to get the correct results!\n"
	},
/* linestyle, lineWidth, elinestyle, elineWidth, Colour[set,`af], markStyle, markSize	*/
	{ "SetLineStyle", ascanf_SetLineStyle, 2, NOT_EOF_OR_RETURN,
		" SetLineStyle[set[,style]]"
	},
	{ "SetLineWidth", ascanf_SetLineWidth, 2, NOT_EOF_OR_RETURN,
		" SetLineWidth[set[,width]]"
	},
	{ "SetELineStyle", ascanf_SetELineStyle, 2, NOT_EOF_OR_RETURN,
		" SetELineStyle[set[,error-line-style]]"
	},
	{ "SetELineWidth", ascanf_SetELineWidth, 2, NOT_EOF_OR_RETURN,
		" SetELineWidth[set[,error-line-width]]"
	},
	{ "SetBarWidth", ascanf_SetBarWidth, 2, NOT_EOF_OR_RETURN,
		" SetBarWidth[set[,bar-width]]"
	},
	{ "SetEBarWidth", ascanf_SetEBarWidth, 2, NOT_EOF_OR_RETURN,
		" SetEBarWidth[set[,error-bar-width]]: returns the true, actual width of this set's bars"
	},
	{ "SetCurrentBarWidth", ascanf_SetCBarWidth, 2, NOT_EOF_OR_RETURN,
		" SetCurrentBarWidth[set]: returns the true, actual width of this set's error bars"
	},
	{ "SetCurrentEBarWidth", ascanf_SetCEBarWidth, 2, NOT_EOF_OR_RETURN,
		" SetCurrentEBarWidth[set]"
	},
	{ "SetMarkStyle", ascanf_SetMarkStyle, 2, NOT_EOF_OR_RETURN,
		" SetMarkStyle[set[,style]]: the type of marker used when distinctive markers are selected."
	},
	{ "SetMarkSize", ascanf_SetMarkSize, 2, NOT_EOF_OR_RETURN,
		" SetMarkSize[set[,size]]: size=NaN: use global (-PSm) setting; size<0: axes-linked, scaling; else size (or scale in MSize mode)"
	},
	{ "SetColour", ascanf_SetColour, 3, NOT_EOF_OR_RETURN,
		" SetColour[set,`colour_ret[,newcolour]]: return a set's (previous) colour.\n"
		" When the set has a specific colour, a pointer to colour_ret is returned, which will contain the name.\n"
		" When it uses one of the predefined attribute colours, the colour number is returned.\n"
		" The <newcolour> argument can be either a stringpointer to a colourname, or a colour attribute number\n"
		" or r,g,b intensities (doubles) or 8bit values (integers)\n"
	},
	{ "SetHighlightColour", ascanf_SetHLColour, 3, NOT_EOF_OR_RETURN,
		" SetHighlightColour[set,`colour_ret[,newcolour]]: return a set's (previous) highlight colour.\n"
		" When the set has a specific colour, a pointer to colour_ret is returned, which will contain the name.\n"
		" Otherwise, colour_ret will contain the name of the global highlight colour.\n"
		" The <newcolour> argument can be either a stringpointer to a colourname, or a colour attribute number,\n"
		" or r,g,b intensities (doubles) or 8bit values (integers)\n"
		" newcolour<0 restores the use of the global highlight colour\n"
	},
	{ "SetProcess", ascanf_SetProcess, 3, NOT_EOF_OR_RETURN,
		"SetProcess[<set>[,`process[,`description]]: returns the current *SET_PROCESS* associated with <set>\n"
		" and optionally installs a new *SET_PROCESS* process, which must be a string. An\n"
		" empty string will remove the set's *SET_PROCESS*.\n"
	},
	{ "NumPens", ascanf_NumPens, 0, NOT_EOF,
		" NumPens: return the number of pens in the current window, or 0.\n"
	},
	{ "SelectPen", ascanf_SelectPen, 1, NOT_EOF,
		" SelectPen[pen_nr]: select pen <pen_nr>. If it does not exist, it is created.\n"
		" Selecting pen 10 when no pens are defined as yet creates 11 (0-10)...\n"
	},
	{ "PenNumPoints", ascanf_PenNumPoints, 0, NOT_EOF_OR_RETURN,
		" PenNumPoints[[numPoints]]: allocate numPoints points to the set, and return numPoints.\n"
		" The call will also set the increment with which future expansions of this and future sets will take place.\n"
		" Withouth argument, just returns the current number items in the pen.\n"
	},
	{ "PenIsDrawn", ascanf_PenDrawn, 1, NOT_EOF_OR_RETURN,
		"PenIsDrawn[[refresh]]: whether or not the current pen has been drawn or would be drawn (if <refresh> is True)\n"
	},
	{ "PensShown?", ascanf_PensShown, 1, NOT_EOF_OR_RETURN,
		"PensShown?: a function that returns False when the user requested that the pens are not to be shown.\n"
		" PensShown?[yesno]: change the setting, returning the old value.\n"
	},
	{ "PenDrawNow", ascanf_PenDrawNow, 1, NOT_EOF_OR_RETURN,
		" PenDrawNow[[Force]]: draw the pen now. With the <force> argument, ignore the NoPens\n"
		" and pen->skip settings (the old default behaviour).\n"
	},
	{ "PenOverwrite", ascanf_PenOverwrite, 1, NOT_EOF,
		" PenOverwrite[boolean]: whether to draw the current pen after/over the datasets."
	},
	{ "PenBeforeSet", ascanf_PenBeforeSet, 1, NOT_EOF,
		" PenBeforeSet[set]: draw the current pen just before drawing set <set>\n"
		" set=-1 restores the default behaviour.\n"
	},
	{ "PenAfterSet", ascanf_PenAfterSet, 1, NOT_EOF,
		" PenAfterSet[set]: draw the current pen just after drawing set <set>\n"
		" set=-1 restores the default behaviour.\n"
	},
	{ "PenFloating", ascanf_PenFloating, 1, NOT_EOF,
		" PenFloating[boolean]: whether or not to include the current pen in the autoscaling."
	},
	{ "PenClipping", ascanf_PenClipping, 1, NOT_EOF,
		" PenClipping[[boolean]]: whether or not to do some form of clipping while drawing pen lines (default on).\n"
		" This is a per-position setting: it affects all future drawing commands until a next PenClipping invocation.\n"
	},
	{ "PenSkip", ascanf_PenSkip2, 2, NOT_EOF_OR_RETURN,
		" PenSkip[[pen_nr[,boolean]]]: whether to skip drawing a pen.\n"
		" When no arguments are given, set the current pen to be skipped. When pen_nr<0,\n"
		" set all pens to be skipped, or as indicated by the 2nd argument. When pen_nr\n"
		" is a valid pen number, only modify that pen. Returns the previous setting when\n"
		" a single pen is modified, or NaN when multiple pens are involved; -1 on error.\n"
		" PenReset resets this flag.\n"
	},
	{ "PenSetLink", ascanf_PenSetLink, 1, NOT_EOF_OR_RETURN,
		"PenSetLink[[set_nr]]: link a pen to the set <set_nr> and/or return the set it is\n"
		" currently linked to. Specify a negative number or NaN to unlink the pen.\n"
		" Linked pens are drawn only when the set they're linked to is drawn\n"
		" (and they're not being skipped)\n"
	},
	{ "PenInfo", ascanf_PenInfo, 2, NOT_EOF_OR_RETURN,
		"PenInfo[&return_string[,`new_string]]]: return the current pen_info for the current pen\n"
		" If &return_string is omitted or 0, a static internal variable is used.\n"
		" If `new_string is given, store the string it contains in the pen's pen_info.\n"
		" NB: return_string is *not* touched when there is no pen info but in that case 0 is returned!\n"
		, 0, 1
	},
	{ "PenHighlightColour", ascanf_PenHighlightColour, 2, NOT_EOF_OR_RETURN,
		" PenHighlightColour[boolean[,colourname]]: whether to draw the current pen highlighted.\n"
		" When colourname is not a valid (string)pointer, the colour used\n"
		" is the current highlight colour. Otherwise, the specified colour is used.\n"
		" Omitting the colourname leaves the current (colour) setting intact.\n"
	},
	{ "PenHighlightText", ascanf_PenHighlightText, 1, NOT_EOF,
		" PenHighlightText[boolean]: whether the current pen's text is drawn highlighted.\n"
		" The highlight colour can be set with PenHighlightColour[].\n"
	},
	{ "PenReset", ascanf_PenReset, 2, NOT_EOF_OR_RETURN,
		" PenReset: reset the current pen. If you put this in a *DRAW_BEFORE*,\n"
		" take care to set $ReDo_DRAW_BEFORE=0, or to do verifications of the auto-fitting state.\n"
		" PenReset[pen_nr[,dealloc]]: reset pen <pen_nr>, where pen_nr must refer to an existing pen.\n"
		" If dealloc is given and True, de-allocate all stored commands.\n"
	},
	{ "PenCurrentPos", ascanf_PenCurrentPos, 0, NOT_EOF_OR_RETURN,
		"PenCurrentPos: returns a pointer to an array describing the current pen's\n"
		" current direction as {x,y,direction}.\n", 0, 1
	},
	{ "PenLift", ascanf_PenLift, 0, NOT_EOF_OR_RETURN,
		" PenLift: lift the current pen.\n"
		" This is equivalent to PenMoveTo[NaN,NaN] .\n"
	},
	{ "PenMoveTo", ascanf_PenMoveTo, 3, NOT_EOF_OR_RETURN,
		" PenMoveTo[x,y[,direction]]: set the current pen at (x,y)\n"
		" The optional <direction> sets the initial direction for PenLineTo-Ego\n"
	},
	{ "PenLineTo", ascanf_PenLineTo, 3, NOT_EOF_OR_RETURN,
		" PenLineTo[x,y] or PenLineTo[idx,xcol,ycol]: draw the current pen to (x,y).\n"
		" If there is no current position, move it to (x,y).\n"
		" x and y may be arrays; in that case, N points (x[i],y[i]) are drawn,\n"
		" where N is the size of the smallest of the 2 arrays.\n"
		" In the second form, draw columns (xcol,ycol) of the specified set.\n"
	},
	{ "PenMoveTo-Ego", ascanf_PenEgoMoveTo, 3, NOT_EOF_OR_RETURN,
		" PenMoveTo-Ego[len,direction[,radix]]]: set the current pen at a distance <len>\n"
		" in direction <direction> (which has radix <radix>). Turtle graphics (ego-centric).\n"
	},
	{ "PenLineTo-Ego", ascanf_PenEgoLineTo, 5, NOT_EOF_OR_RETURN,
		" PenLineTo-Ego[len,direction[,radix[,cumul_length[,world_angles]]]]: draw the current pen over a distance <len>\n"
		" in direction <direction> (which has radix <radix>). Turtle graphics (ego-centric).\n"
		" If there is no current position, move it to the specified position.\n"
		" When len is an array, it can taken as cumulative length, e.g. travelled distance since the first point.\n"
		" direction can be taken as (an array of) (an) absolute -i.e. world- angle(s).\n"
	},
	{ "PenRectangle", ascanf_PenRectangle, 6, NOT_EOF,
		" PenRectangle[x,y,w,h,centre?,fill?]: draws a rectangle with width w and height h\n"
		" relative to (x,y). If <centre?>, (x,y) is the centre of the rectangle, otherwise\n"
		" it is the lowerleft corner (if w and h both positive). When <fill?>, the rectangle\n"
		" is filled in PenFillColour, and outlined in PenColour (fill?> 0) or not outlined (fill?< 0).\n"
	},
	{ "PenEllipse", ascanf_PenEllipse, 4, NOT_EOF,
		" PenEllipse[x,y,rx[,ry]]: draws an ellipse with radii rx and ry\n"
		" relative to (x,y). In case there is e.g. a log transformation on one of the axes,\n"
		" the radii are re-determined from the transformed bounding rectangle, and an ellipse\n"
		" is drawn with these radii, and its centre shifted to be in the middle of the resulting\n"
		" radii. This is of course only an approximation of the shape that should be drawn.\n"
		" When <ry> is not specified, it a circle with radius <rx> is drawn.\n"
	},
	{ "PenPolygon", ascanf_PenPolygon, 3, NOT_EOF,
		" PenRectangle[&X,&Y,fill?]: draws a polygon as specified by the arrays X and Y;\n"
		" the smallest of the two arrays is taken for determining the number of points on the\n"
		" polygon. It is your responsability to close the shape.\n"
		" When <fill?>, the polygon is filled in PenFillColour, and outlined in PenColour (fill?> 0) or not outlined (fill?< 0).\n"
	},
	{ "PenLineStyleWidth", ascanf_PenLineStyleWidth, 2, NOT_EOF,
		" PenLineStyleWidth[style,width]: style and width for the current pen.\n"
		" Same conventions as for dataset line style and width.\n"
	},
	{ "PenMark", ascanf_PenMark, 2, NOT_EOF_OR_RETURN,
		" PenMark or PenMark[style,size]: mark from the current position onwards; without arguments, no marks are drawn"
		" Same conventions as for dataset marker style and size.\n"
	},
	{ "PenColour", ascanf_PenColour, 1, NOT_EOF_OR_RETURN,
		" PenColour[[]`colour]: colour can be a (string)pointer containing the name,\n"
		" or r,g,b intensities (doubles) or 8bit values (integers)\n"
		" or it can be a number referring to one of the predefined colours.\n"
		" The value returned by SetColour[] or SetHighlightColour[] can be used as argument.\n"
		" When called without arguments, PenColour returns the pen's current colour.\n"
	},
	{ "PenFillColour", ascanf_PenFillColour, 1, NOT_EOF,
		" PenFillColour[`colour]: fill colour for rectangles.\n"
		" The value returned by SetColour[] or SetHighlightColour[] can be used as argument.\n"
		" When called without arguments, PenColour returns the pen's current colour.\n"
	},
	{ "PenTextOutside", ascanf_PenTextOutside, 1, NOT_EOF_OR_RETURN,
		" PenTextOutside or PenTextOutside[flag]: indicate that text should (or should not if flag\n"
		" is False) be drawable outside the axes. If set, and text thus marked extends beyond the\n"
		" data plotting area spanned by the axes, this area is reduced such that the text fits inside\n"
		" the image plane. The effect is similar to the <legend_always_visible> option that applies to\n"
		" legend, axes labels and titles. NB: the pen used to draw the text must not be marked as floating!\n"
	},
	{ "PenText", ascanf_PenText, 8, NOT_EOF_OR_RETURN,
		" PenText[x,y,vx[,vy[,justX[,justY[,[fnt,]`dim_ret]]]]] or PenText[x,y,`string[,justX[,justY[,[fnt,]`dim_ret]]]]:\n"
		" Draw a text at the current position.\n"
		" NB: <fnt> is a true optional argument before `dim, but must come after justY.\n"
		"     fnt=0(default): axis font; 1= title; 2= label; 3= legend font\n"
		" justX/justY: -1=right/bottom, 0=centre, 1=left/top (normal) justification\n"
	},
	{ "PenTextBox", ascanf_PenTextBox, 9, NOT_EOF_OR_RETURN,
		" PenTextBox[x,y,vx[,vy[,justX[,justY[,[fnt,]`dim_ret]]]],fill?]\n"
		" PenTextBox[x,y,`string[,justX[,justY[,[fnt,]`dim_ret]]],fill?]:\n"
		" Draw a text at the current position, inside a fitting rectangle drawn just before.\n"
		" NB: <fnt> is a true optional argument before `dim, but must come after justY.\n"
		"     fnt=0(default): axis font; 1= title; 2= label; 3= legend font\n"
		" justX/justY: -1=right, 0=centre, 1=left (normal) justification\n"
		" The <fill?> argument should be given, conform the convention described for PenRectangle[].\n"
	},
	{ "CustomFont", ascanf_CustomFont, 6, NOT_EOF_OR_RETURN,
		" CustomFont[&store, `xfont, `psfont, psfont-size[, psreencode][,`alt-xfont]]: creates a CustomFont instance,\n"
		" consisting of an X11 (screen) font, a PostScript (printer font), specified as a name/size pair,\n"
		" and an optional alternative screen font. The result is stored as a special field in the variable <store>\n"
		" The optional switch <psreencode> determines whether or not the PostScript font is allowed to be\n"
		" reencoded (i.e. if it is a text font that has a Latin-1252/iso8859-1 encoding vector)\n"
		" Upon success, &store is returned, NaN or 0 otherwise.\n"
	},
	{ "BackgroundColour", ascanf_bgColour, 2, NOT_EOF_OR_RETURN,
		"BackgroundColour[`colour[,`newcolour]]: colour (to be) used for the window background.\n"
	},

}, *ascanf_FunctionList= vars_ascanf_Functions;

  /* The number of builtin entries (the size of the table divided by the size of the individual entries): */
int ascanf_Functions= sizeof(vars_ascanf_Functions)/sizeof(ascanf_Function);

/* Below we initialise the various internal representations of the externally visible variables.
 \ Note that this is done by direct referral to entries in the function table. This means that
 \ if you add an entry somewhere in the middle of the range in use, you will have to make sure
 \ that all references are updated. If not, you'll see some pretty weird behaviour, if not
 \ very earthly crashes!
 */
double *ascanf_setNumber= &vars_ascanf_Functions[1].value;
	ascanf_Function *af_setNumber= &vars_ascanf_Functions[1];
double *ascanf_TotalSets= &vars_ascanf_Functions[2].value;
double *ascanf_numPoints= &vars_ascanf_Functions[3].value;
	ascanf_Function *af_numPoints= &vars_ascanf_Functions[3];
double *ascanf_counter= &vars_ascanf_Functions[4].value;
double *ascanf_Counter= &vars_ascanf_Functions[5].value;
	ascanf_Function *af_Counter= &vars_ascanf_Functions[5];
ascanf_Function *af_loop_counter= &vars_ascanf_Functions[6];
double *ascanf_loop_counter= &vars_ascanf_Functions[6].value;
double *ascanf_data0= &vars_ascanf_Functions[7].value;
	ascanf_Function *af_data0= &vars_ascanf_Functions[7];
double *ascanf_data1= &vars_ascanf_Functions[8].value;
	ascanf_Function *af_data1= &vars_ascanf_Functions[8];
double *ascanf_data2= &vars_ascanf_Functions[9].value;
	ascanf_Function *af_data2= &vars_ascanf_Functions[9];
double *ascanf_data3= &vars_ascanf_Functions[10].value;
	ascanf_Function *af_data3= &vars_ascanf_Functions[10];
double *ascanf_index_value= &vars_ascanf_Functions[11].value;
double *ascanf_self_value= &vars_ascanf_Functions[12].value;
double *ascanf_current_value= &vars_ascanf_Functions[13].value;
double *curvelen_with_discarded= &vars_ascanf_Functions[14].value;
double *disable_SET_PROCESS= &vars_ascanf_Functions[15].value;
double *SET_PROCESS_last= &vars_ascanf_Functions[16].value;
double *SAS_converts_angle= &vars_ascanf_Functions[17].value;
double *SAS_converts_result= &vars_ascanf_Functions[18].value;
double *SAS_exact= &vars_ascanf_Functions[19].value;
double *SS_exact= &vars_ascanf_Functions[20].value;
double *SAS_Ignore_NaN= &vars_ascanf_Functions[21].value;
double *SS_Ignore_NaN= &vars_ascanf_Functions[22].value;
double *SS_Empty_Value= &vars_ascanf_Functions[23].value;
double *ascanf_verbose_value= &vars_ascanf_Functions[24].value;
	ascanf_Function *af_verbose= &vars_ascanf_Functions[24];
double *ascanf_popup_verbose= &vars_ascanf_Functions[25].value;
double *ascanf_compile_verbose= &vars_ascanf_Functions[26].value;
double *Allocate_Internal= &vars_ascanf_Functions[27].value;
double *ReDo_DRAW_BEFORE= &vars_ascanf_Functions[28].value;
double *Really_DRAW_AFTER= &vars_ascanf_Functions[29].value;
double *SS_TValue= &vars_ascanf_Functions[30].value;
double *SS_FValue= &vars_ascanf_Functions[31].value;
double *ascanf_IntensityColours= &vars_ascanf_Functions[32].value;
	/* #33 is $IntensityRGB */
double *ascanf_ActiveWinWidth= &vars_ascanf_Functions[34].value;
double *ascanf_ActiveWinHeight= &vars_ascanf_Functions[35].value;
double *ascanf_IgnoreQuick= &vars_ascanf_Functions[36].value;
double *ascanf_Variable_init= &vars_ascanf_Functions[37].value;
double *ascanf_ExitOnError= &vars_ascanf_Functions[38].value;
double *ascanf_UseConstantsLists= &vars_ascanf_Functions[39].value;
double *AllowSimpleArrayOps_value= &vars_ascanf_Functions[40].value;
	ascanf_Function *af_AllowSimpleArrayOps= &vars_ascanf_Functions[40];
double *AlwaysUpdateAutoArrays_value= &vars_ascanf_Functions[41].value;
	ascanf_Function *af_AlwaysUpdateAutoArrays= &vars_ascanf_Functions[41];
double *ascanf_ScreenWidth= &vars_ascanf_Functions[42].value;
double *ascanf_ScreenHeight= &vars_ascanf_Functions[43].value;
double *ascanf_ReadBufVal= &vars_ascanf_Functions[44].value;
  /* PointerPos[n] is #45	*/
double *ascanf_SetsReordered= &vars_ascanf_Functions[46].value;
double *ascanf_SyncedAnimation= &vars_ascanf_Functions[47].value;
double *ascanf_AllowSomeCompilingInitialisations= &vars_ascanf_Functions[48].value;
  /* ArgList[<n>] (af_ArgList!!!) is #49	*/
ascanf_Function *af_ArgList= &vars_ascanf_Functions[49];
double *ascanf_switch_case= &vars_ascanf_Functions[50].value;
double *AllowGammaCorrection= &vars_ascanf_Functions[51].value;
double *do_gsTextWidth_Batch= &vars_ascanf_Functions[52].value;
ascanf_Function *ascanf_VarLabel= &vars_ascanf_Functions[53];
ascanf_Function *ascanf_d3str_format= &vars_ascanf_Functions[54];
ascanf_Function *ascanf_ValuesPrinted= &vars_ascanf_Functions[55];
ascanf_Function *ascanf_d2str_NaNCode= &vars_ascanf_Functions[56];
  /* #57 is $LastActionDetails	*/
ascanf_Function *ascanf_XGOutput= &vars_ascanf_Functions[58];
  /* #59 is $elapsed	*/
ascanf_Function *ascanf_Dprint_fp= &vars_ascanf_Functions[60];
double *DataWin_before_Processing= &vars_ascanf_Functions[61].value;
double *AllowProcedureLocals_value= &vars_ascanf_Functions[62].value;
	ascanf_Function *af_AllowProcedureLocals= &vars_ascanf_Functions[62];
double *Find_Point_exhaustive_value= &vars_ascanf_Functions[63].value;
	ascanf_Function *af_Find_Point_exhaustive= &vars_ascanf_Functions[63];
double *AllowArrayExpansion_value= &vars_ascanf_Functions[64].value;
	ascanf_Function *af_AllowArrayExpansion= &vars_ascanf_Functions[64];

/* 20030413
 \ A certain number of functions was moved to dynamic modules. To ensure backward
 \ compatibility and that users don't need to modify existing code (add *LOAD_MODULE*
 \ statements, manage unloading/not loading of modules after DumpProcessed, ...),
 \ we provide an autoload feature.
 */
DyModAutoLoadTables _AutoLoadTable[]= {
	{ "$SplineInit-does-PWLInt-too", "splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineInit",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineX",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineY",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineE",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineL",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "Spline",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "GetSplineData",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineFromData",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "PWLIntX",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "PWLIntY",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "PWLIntE",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "PWLIntL",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "PWLInt",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SplineFinished",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "Spline-Resample",	"splines",	RTLD_LAZY|RTLD_GLOBAL },
	{ "PWLInt-Resample",	"splines",	RTLD_LAZY|RTLD_GLOBAL },

	{ "SavGolayCoeffs",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolayInit",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolayX",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolayY",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolayE",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolay",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolay2Array",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SavGolayFinished",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },

	{ "InitFFTW",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "CloseFFTW",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "$fftw-nthreads", "fourconv3", RTLD_LAZY|RTLD_GLOBAL },
	{ "$fftw-planner-level", "fourconv3", RTLD_LAZY|RTLD_GLOBAL },
	{ "rfftw",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "inv_rfftw",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "convolve_fft",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },
	{ "convolve",	"fourconv3",	RTLD_LAZY|RTLD_GLOBAL },

	{ "FitCircle2Triangle", "utils",	RTLD_LAZY|RTLD_GLOBAL },
	{ "MonotoneArray", "utils",	RTLD_LAZY|RTLD_GLOBAL },
	{ "RemoveTies", "utils",	RTLD_LAZY|RTLD_GLOBAL },
	{ "RemoveTrend", "utils",	RTLD_LAZY|RTLD_GLOBAL },
	{ "GetULabel", "utils",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SetULabel", "utils",	RTLD_LAZY|RTLD_GLOBAL },
	{ "SubsetArray", "utils",	RTLD_LAZY|RTLD_GLOBAL },

	{ "Make-Histogram", "stats",	RTLD_LAZY|RTLD_GLOBAL },
	{ "lmFit", "stats",	RTLD_LAZY|RTLD_GLOBAL },
	{ "rlmFit", "stats",	RTLD_LAZY|RTLD_GLOBAL },

	{ "basename",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "concat",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "encode",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "getstring",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strcasecmp",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strcasestr",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strcmp",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strdup",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strlen",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strstr",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strstr-count",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "strrstr",	"strings",	RTLD_LAZY|RTLD_GLOBAL },
	{ "timecode", "strings",	RTLD_LAZY|RTLD_GLOBAL },

	{ "sran-PM",	"simanneal", RTLD_LAZY|RTLD_GLOBAL },
	{ "ran-PM",	"simanneal", RTLD_LAZY|RTLD_GLOBAL },

	{ "EulerSum",	"integrators", RTLD_LAZY|RTLD_GLOBAL },
	{ "EulerSum-NaNResets",	"integrators", RTLD_LAZY|RTLD_GLOBAL },

	{ "Python-Shell",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-Call",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-Eval",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-EvalValue",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-EvalFile",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-Compile",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-DelCompiled",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-PrintCompiled",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-EvalCompiled",	"Python", RTLD_NOW|RTLD_GLOBAL },
	{ "Python-EvalValueCompiled",	"Python", RTLD_NOW|RTLD_GLOBAL },
};

int AutoLoads= sizeof(_AutoLoadTable)/sizeof(DyModAutoLoadTables);
