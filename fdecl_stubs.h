/* 20020404 RJVB:
 \ This is a file with stub definitions that allows fdecl.h to be included in files that do not want
 \ (or shouldn't, like ascanfc.c and ascanfc2.c) include xgraph.h .
 \ For a number of structure definitions, macros are declared that replace those types by voids; this is
 \ ok (syntactically and safe) as long as variables of these types are only handled as pointers in the
 \ file under consideration. LocalWin is replaced by struct LocalWin, however; this may lead to warnings
 \ of "definition only inside function declaration scope".
 \ Only the beginning (tested with ascanfc2.c and ascanfc.c); more may (have to) follow.
 \
 \ Of course, one may have to undefine some of the macros below after including fdecl.h !!
 \
 */

#ifndef _FDECL_STUBS_H
#define _FDECL_STUBS_H

  /* Define the xgraph.h header file's signature! */
#define _XGRAPH_H

#define LocalWin	struct LocalWin
#define LocalWindows	void
#define UserLabel	void
#define RGB	void
#define Transform	void
#define Process	void
#define XGStringList	void
#define LabelsList	void
#define ValCategory	void
#define AxisValues	void
#define LocalWinGeo	void
#define XGPenPosition	void
#define XGPen	void

#define AxisName	int

#endif
