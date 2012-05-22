#ifndef _FIG_SAVE_H

#include "fig_object.h"


A_FUN( Fig_write_arc, (FILE *fp, F_arc *o) );
A_FUN( Fig_write_compound, (FILE *fp, F_compound *o) );
A_FUN( Fig_write_ellipse, (FILE *fp, F_ellipse *o) );
A_FUN( Fig_write_line, (FILE *fp, F_line *o) );
A_FUN( Fig_write_spline, (FILE *fp, F_spline *o) );
A_FUN( Fig_write_text, (FILE *fp, F_text *o) );

#define _FIG_SAVE_H
#endif
