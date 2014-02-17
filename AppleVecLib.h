#ifndef _APPLE_VEC_LIB_H

// 20140217: On OS X > 10.6 including the Accelerate header leads to inclusion of /usr/include/MacTypes.h, which contains a
// typedef for Boolean. If XIntrinsic.h has already been included that leads to a compile error.

#ifndef _XtIntrinsic_h
#	include <Accelerate/Accelerate.h>
/* 20060907 RJVB
 \ STUPID Apple gcc 4.0 does a #define I _Complex_I in complex.h
 \ which seems really dumb to do
 \ So we undefine it here...
 \ 20111031: need to undefine it also everywhere after including vecLib.h !
 */
#	ifdef I
#		undef I
#	endif
#endif

#define _APPLE_VEC_LIB_H
#endif // _APPLE_VEC_LIB_H
