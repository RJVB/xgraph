#include <vecLib/vecLib.h>
/* 20060907 RJVB
 \ STUPID Apple gcc 4.0 does a #define I _Complex_I in complex.h
 \ which seems really dumb to do
 \ So we undefine it here...
 \ 20111031: need to undefine it also everywhere after including vecLib.h !
 */
#ifdef I
#	undef I
#endif
