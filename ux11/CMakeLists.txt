set(UX11_SRCS
	ux11f_watter.c		ux11_open.c
	ux11f_gcvals.c		ux11g_value.c		ux11s_vismap.c
	ux11f_hints.c		ux11n_font.c
	ux11n_visual.c		ux11f_xa.c		ux11_err.c)

add_library(libux11 STATIC ${UX11_SRCS})
