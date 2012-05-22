#ifndef _XG_PRAGMAS_H

#if 1

#	if __GNUC__ >= 3
#		define pragma_pure	__attribute__ ((pure))
#		define pragma_const	__attribute__ ((const))
#		define pragma_noreturn	__attribute__ ((noreturn))
#		define pragma_malloc	__attribute__ ((malloc))
#		define pragma_must_check	__attribute__ ((warn_unused_result))
#		define pragma_deprecated	__attribute__ ((deprecated))
#		define pragma_used	__attribute__ ((used))
#		define pragma_unused	__attribute__ ((unused))
#		define pragma_packed	__attribute__ ((packed))
#		define pragma_likely(x)	__builtin_expect (!!(x), 1)
#		define pragma_unlikely(x)	__builtin_expect (!!(x), 0)
#	endif

#endif

#ifndef pragma_likely
#	define pragma_pure	/* */
#	define pragma_const	/* */
#	define pragma_noreturn	/* */
#	define pragma_malloc	/* */
#	define pragma_must_check	/* */
#	define pragma_deprecated	/* */
#	define pragma_used	/* */
#	define pragma_unused	/* */
#	define pragma_packed	/* */
#	define pragma_likely(x)	(x)
#	define pragma_unlikely(x)	(x)
#endif

#define _XG_PRAGMAS_H
#endif
