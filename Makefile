#
# Makefile for the xgraph program
#
# David Harrison
# University of California,  Berkeley
# 1987
#
# gccopt and cmake are nothing but scripts that basically 
# expand to 'gcc -O' and 'make -f'

# a number of settings are determined via a series of 'machdepXXX' scripts, some that take arguments
# where possible, these should be called as
# SOMESETTING := $(shell machdepSOMESETTING SOME SETTING)
# rather than
# SOMESETTING = $(shell machdepSOMESETTING SOME SETTING)
# or directly in a build line, as the first version (using :=) retrieves the setting once only instead of executing
# the script each time one evaluates $(SOMESETTING) .

UNIBIN=-unibin64
CLEVEL=
CHECK=-c
STRIP=
XG_FLAGS=
DEBUG=
LASTOPTIONS=
_CLEVEL=$(UNIBIN) $(CLEVEL) #-Ac
_CFLAGS=$(_CLEVEL) $(DEBUG) -I. -Ixtb -Iux11 #-Q -DHPGL_DUMP #-DIDRAW_DUMP
CFLAGS=$(_CFLAGS) $(LASTOPTIONS)
DEBUGSUPPORT= #-DDEBUGSUPPORT
_XCFLAGS=$(DEBUG) -I. -Ixtb -Iux11 -I/usr/local/include #-Q
XCFLAGS=$(UNIBIN) $(_XCFLAGS) $(LASTOPTIONS)
COMP=$(shell ./machdepcomp)
CXXCOMP=$(shell ./machdepcxxcomp)
ASCANFCOMP:=$(shell ./machdepcomp ascanf)
DYMODCOMP:=$(shell ./machdepcomp dymod)
DYMODCOMPXX:=$(shell ./machdepcomp dymod c++)
ARCH=
CC=$(COMP) -DXGRAPH $(ARCH) #-safeOpt
CXX:=$(CXXCOMP) -DXGRAPH $(ARCH) #-safeOpt
ACC=$(ASCANFCOMP) -DXGRAPH $(ARCH) #-safeOpt
DCC=$(DYMODCOMP) -DXGRAPH $(ARCH) #-safeOpt
DCXX=$(DYMODCOMPXX) -DXGRAPH $(ARCH) #-safeOpt
# OTHEROBS = `./machdepobjects`
OTHEROBS:=$(shell ./machdepobjects)
OTHERLIBS:=$(shell ./machdeplibs $(MAKEFLAGS))

PREFSDIR:=$(shell ./machdepPrefsDir)

# 20010723: Options do be specified at compile time. See config.h for an uptodate list and settings.
# uncomment if you have FFTW installed and want to use it for runtime FFTs (in ascanfc2.c):
# # FFTW=-DHAVE_FFTW
# 20010725: fftw libraries inserted by machdeplibs when config.h contains #define HAVE_FFTW
# FFTW_LIBS=-lrfftw -lfftw
# # VSCANF = -DHAVE_VSCANF
# Define to compile in support for dynamic module (shared libraries) loading:
# # DYMOD = -DXG_DYMOD_SUPPORT

# # COPTS = $(FFTW) $(VSCANF) $(DYMOD)
# Compile time options above now defined in config.h!

ASCANFSRC = ascanfc-table.c ascanfc3.c ascanfcSS.c ascanfc.c ascanfc2.c ascanfcMap2.c vscanf/asscanf.c
SRC	= main.c xgX.c hard_devices.c dialog.c dialog_s.c new_ps.c matherr.c SS.c dymod.c xgPen.c ReadData.c xgInput.c xgsupport.c LegendsNLabels.c xgraph.c alloca.c $(ASCANFSRC) fascanf.c regex.c # hpgl.c idraw.c params.c 
LIBOBJ = xgX.o hard_devices.o dialog.o dialog_s.o new_ps.o matherr.o SS.o dymod.o xgPen.o alloca.o $(ASCANFSRC:.c=.o) arrayvops.o ascanfcMap.o fascanf.o ReadData.o xgInput.o LegendsNLabels.o regex.o # hpgl.o idraw.o params.o 
OBJS	  = $(LIBOBJ) xgsupport.o xgraph.o # hpgl.o idraw.o params.o
DSRC = constants.c utils.c stats.c strings.c

DYMOD_SOURCES=constants.c stats.c LineCircle.c utils.c strings.c CMaps.c Python/Python.c Python/DM_Python.h Python/PythonInterface.h Python/PyObjects.h Python/AscanfCall.c Python/DataSet.c Python/ULabel.c dm_example.c contrib/fourconv3.c contrib/fourconv3.c contrib/splines.c contrib/simanneal.c contrib/simanneal.h contrib/integrators.c contrib/ddeltaNEC.c contrib/fig_dist.c contrib/pearson_correl.o Import/GSRio.c Import/gsr.h Import/IEFio.c Import/ief.h  Import/CSVio.c

DYMOD_DEPHEADERS=dymod.h ascanf.h dymod_interface.h compiled_ascanf.h DataSet.h LocalWin.h 64typedefs.h

# the flags required to compile an object file that should be linked into a shared library (dynamic module):
SHOBJ:=$(shell ./machdepLDOPTS shobj)
# the flag to pass to the compiler to export a programme's symbols to a module loaded via dlopen(): (under linux..)
# (that is, when XG_DYMOD_IMPORT_MAIN is NOT defined and dymods have automatic access to all symbols from the main programme)
DYNAMIC:=$(shell ./machdepLDOPTS dynload)

PYTHONSRC = Python/Python.c Python/AscanfCall.c Python/DataSet.c Python/ULabel.c
PYTHONVERSION:= $(shell ./machdepLibVersions Python)
PYTHON_DM_NAME=Python.$(PYTHONVERSION).so
PYTHONDIR:=$(shell Python/machdep_header $(PYTHONVERSION))
PYTHONINC=$(PYTHONDIR)/include

UX11	= libux11.a
LUX11 = -lux11
# remove -lXinerama if your X11 server doesn't support it:
X11LIB= -L/usr/X11R6/lib
lX11 = $(X11LIB) -lXinerama -lXext -lX11
XTB	= libxtb.a
LXTB = -lxtb
SYSLIBS =
LIBS	= -L. -lxgraph -Lxtb $(LXTB) -Lux11 $(LUX11) $(lX11) $(FFTW_LIBS) $(SYSLIBS) -lm $(OTHERLIBS) -lm
XGLDOPTS:=$(shell ./machdepLDOPTS xgraph) 

TARGET	= xgraph

# Compress object files after linking. This reduces the space they take up, but isn't very necessary anymore,
# and can actually cause issues debugging (gdb on OS X 10.6 expects the debugging information in the individual
# object files used during the final link stage). Object files combined into a library are always compressed though
# COMPRESS = -Zbg

# build rule suitable for gccopt (that wants the target defined before the source):
.cc.o:
#	$(CXX) -fno-exceptions $(COPTS) $(CFLAGS) -o $@ $(CHECK) $<
	$(CXX) -fexceptions $(COPTS) $(CFLAGS) -o $@ $(CHECK) $<

.cpp.o:
	$(CXX) -fexceptions $(COPTS) $(CFLAGS) -o $@ $(CHECK) $<

.c.o:
	$(CC) $(COPTS) $(CFLAGS) -o $@ $(CHECK) $<

# all: $(TARGET) $(DSRC:.c=.so) CSVio.so $(PYTHON_DM_NAME)
all: $(TARGET) $(DSRC:.c=.so) CSVio.so Python.so

DM_Python: $(PYTHON_DM_NAME)

objs: $(OBJS)

All: all contrib import

ascanf: $(ASCANFSRC:.c=.o)

DEPLIBS= xtb_m ux11_m libxgraph.a $(OTHEROBS)
Libs: $(DEPLIBS)

#--------

$(TARGET):	config.h $(DEPLIBS) main.c xgsupport.o xgraph.o
# 	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LIBS)
	-no | mv $(TARGET) $(TARGET).prev
	-gzip -9vf $(TARGET).prev
# 	-rm -f $(TARGET).prev
# we always compile main.c with DEBUG to have a maximum of debugging info available, at little or no cost:
	echo "#define XGraphBuildPlatform \"`uname -nmrs` ; CC=\'$(CC)\' ACC=\'$(ACC)\'\"" > buildplatform.h
	$(CC) -g $(COPTS) $(CFLAGS) $(CHECK) main.c
	ranlib libxgraph.a
	$(CXX) -g $(XGLDOPTS) $(COMPRESS) $(CFLAGS) $(XG_FLAGS) $(DYNAMIC) -o $(TARGET) main.o xgsupport.o xgraph.o $(OTHEROBS) $(LIBS)
# make_debug will either make a stripped ${TARGET}, leaving a gzipped, nonstripped copy in ${TARGET}.bin.gz, or
# it will replace ${TARGET} by a wrapper script that will cause the debugger to be invoked (exe. in ${TARGET}.bin).
	./make_debug $(TARGET) CFLAGS= $(CFLAGS) XCFLAGS= $(XCFLAGS) XG_FLAGS= $(XG_FLAGS) DEBUGSUPPORT= $(DEBUGSUPPORT)
	rm buildplatform.h
	mkdirhier $(PREFSDIR)
	-scripts/ln2 $(PWD)/scripts/noprocs.xg $(PREFSDIR)
	-scripts/ln2 $(PWD)/scripts/xg_init.xg $(PREFSDIR)
	touch -r $(TARGET) .make_success

refresh: $(OBJS) $(TARGET)
	rm -i $(OBJS)
	rm $(TARGET)

config.h: cpu_cycles_per_second.h
	touch -r $< $@

lowlevel_timer.h: cpu_cycles_per_second.h
	touch -r $< $@

cpu_cycles_per_second.h: cpu_cycles_per_second
	./cpu_cycles_per_second > $@
	touch -r ./cpu_cycles_per_second $@

SYSHEADERS = /usr/include/X11/*.h /usr/local/include/*.h

#tags: $(SRC) $(DSRC) $(PYTHONSRC) Python/DM_Python.h Python/PyObjects.h $(SYSHEADERS) xgraph.h DataSet.h xgout.h new_ps.h ascanf.h Macros.h Sinc.h ux11/*.[ch] xtb/*.[ch] Elapsed.h lowlevel_timer.h xgALLOCA.h SS.h XXseg.h XGPen.h dymod.[ch] compiled_ascanf.h NaN.h fdecl.h config.h matherr.[ch] ReadData.h ascanfcMap.cpp arrayvops.[ch]
tags: $(SRC) $(DYMOD_SOURCES) $(SYSHEADERS) xgraph.h DataSet.h xgout.h new_ps.h ascanf.h Macros.h Sinc.h ux11/*.[ch] xtb/*.[ch] Elapsed.h lowlevel_timer.h xgALLOCA.h SS.h XXseg.h XGPen.h dymod.[ch] compiled_ascanf.h NaN.h fdecl.h config.h matherr.[ch] ReadData.h ascanfcMap.cpp arrayvops.[ch] $(DYMOD_DEPHEADERS)
	-xgctags $?
	-touch tags

libxgraph.a: $(LIBOBJ)
# 	scripts/update_lib $@ $?
# 	ranlib libxgraph.a
# 	zero $?
	scripts/update_lib $@ $(LIBOBJ)
	_obj_compress 1 "gzip -9" "" gz "" $(LIBOBJ) &

DataSet.h: SS.h compiled_ascanf.h
	touch -r `ls -1t $? | head -1` DataSet.h

xgraph.h: DataSet.h xgout.h Elapsed.h xtb/xtb.h LocalWin.h 64typedefs.h
# 	touch -r DataSet.h xgraph.h
	touch -r `ls -1t $? | head -1` xgraph.h

Elapsed.h: lowlevel_timer.h

xgALLOCA.h: xfree.h
	touch -r xfree.h xgALLOCA.h

# main.o: main.c copyright.h
# 	$(CC) $(COPTS) $(CFLAGS) -g $(CHECK) $<

dialog_s.o: dialog_s.c xgout.h xgraph.h hard_devices.h xtb/xtb.h ascanf.h
# 	$(CC) -simOpt $(CFLAGS) $(CHECK) $<
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

dialog.o: dialog.c hard_devices.h xgout.h xgraph.h hard_devices.h xtb/xtb.h new_ps.h
# 	$(CC) -simOpt $(COPTS) $(CFLAGS) $(CHECK) $<
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

hard_devices.o: hard_devices.c hard_devices.h xgraph.h xgout.h
#	$(CC) $(COPTS) $(CFLAGS) -fno-writable-strings $(CHECK) $<
	$(CC) $(COPTS) $(CFLAGS) -no-rwstrings $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

xgPen.o: xgPen.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h XGPen.h dymod.h
# pass XCFLAGS and XG_FLAGS because these routines used to be in xgraph.c and xgsupport.c until 20010718.
# 	$(ACC) -gOpt $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
	$(ACC) $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

xgInput.o: xgInput.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h XXseg.h dymod.h Python/PythonInterface.h
# 	$(CC) -simOpt $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
# 20040519: compile with -gOpt; there is an issue with SubstituteOpcodes that causes intermittent crashes.
	$(CC) -gOpt $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

ReadData.o: ReadData.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h XXseg.h dymod.h Python/PyObjects.h Python/PythonInterface.h
# 	$(CC) -simOpt $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
	$(CC) $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

xgsupport.o: xgsupport.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h XGPen.h dymod.h
# 	$(CC) -simOpt $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
	$(CC) $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<

LegendsNLabels.o: LegendsNLabels.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h XXseg.h XGPen.h
# 	$(CC) -simOpt $(COPTS) $(XCFLAGS) $(DEBUGSUPPORT) $(XG_FLAGS) $(CHECK) $<
	$(CC) $(COPTS) $(XCFLAGS) $(DEBUGSUPPORT) $(XG_FLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; 

xgraph.o: xgraph.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h XXseg.h XGPen.h Python/PythonInterface.h
# 	$(CC) -simOpt $(COPTS) $(XCFLAGS) $(DEBUGSUPPORT) $(XG_FLAGS) $(CHECK) $<
	$(CC) $(COPTS) $(XCFLAGS) $(DEBUGSUPPORT) $(XG_FLAGS) $(CHECK) $<

alloca.o: alloca.c
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

xgX.o: xgX.c xgout.h xgraph.h ascanf.h XGPen.h xtb/xtb.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

hpgl.o: hpgl.c xgout.h xgraph.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

new_ps.o: new_ps.c xgout.h xgraph.h new_ps.h xtb/xtb.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

matherr.o: matherr.c
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

SS.o: SS.c xgraph.h sse_mathfun.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

dymod.o: dymod.c dymod.h compiled_ascanf.h ascanf.h dymod_interface.h DataSet.h $(DYMOD_DEPHEADERS) Python/PythonInterface.h
	$(CC) $(COPTS) $(CFLAGS) $(DYNAMIC) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

ascanfcMap.o: ascanfcMap.cpp ascanf.h xfree.h

arrayvops.o: arrayvops.cpp arrayvops.h sse_mathfun.h

vscanf/asscanf.o: vscanf/asscanf.c vscanf/vfscanf.c

ascanfc-table.h: ascanfc.c ascanfc2.c ascanfc3.c ascanfcSS.c xgX.c xgPen.c ascanfc-table-template.h
	cat ascanfc-table-template.h > $@
	-cat $+ | grep '^int.*ASCB_ARGLIST' | sed -e 's/.*/extern & ;/' >> $@
	/bin/echo "#endif" >> $@

ascanfc-table.o: ascanfc-table.c ascanfc-table.h ascanf.h compiled_ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h Sinc.h DataSet.h #
	$(ACC) $(STRIP) -BSD $(COPTS) $(XG_FLAGS) $(CFLAGS) -O0 -fno-builtin $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

ascanfc.o: ascanfc.c ascanf.h compiled_ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h Sinc.h DataSet.h Python/PythonInterface.h sse_mathfun.h arrayvops.h  # xgraph.h ascanfc-table.o 
	$(ACC) -BSD $(COPTS) $(XG_FLAGS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

ascanfc2.o: ascanfc2.c ascanf.h compiled_ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h DataSet.h arrayvops.h # xgraph.h
	$(ACC) -BSD $(COPTS) $(XG_FLAGS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

ascanfc3.o: ascanfc3.c xgraph.h hard_devices.h ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h xgout.h xtb/xtb.h dymod.h
# pass XCFLAGS and XG_FLAGS because these routines used to be in xgraph.c and xgsupport.c until 20010718.
	$(ACC) $(COPTS) $(XCFLAGS) $(XG_FLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

ascanfcSS.o: ascanfcSS.c ascanf.h compiled_ascanf.h Elapsed.h lowlevel_timer.h xgALLOCA.h DataSet.h arrayvops.h # xgraph.h
	$(ACC) -BSD $(COPTS) $(XG_FLAGS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

fascanf.o: fascanf.c xgALLOCA.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

params.o: params.c params.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

idraw.o: idraw.c xgout.h xgraph.h
	$(CC) $(COPTS) $(CFLAGS) $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

regex.o: regex.c
#	$(CC) $(COPTS) $(CFLAGS) -fno-writable-strings $(CHECK) $<
	$(CC) $(COPTS) $(CFLAGS) -no-rwstrings $(CHECK) $<
# 	ar rv libxgraph.a $@; zero $@

xg11:		$(TOBJ) xtb_m ux11_m
		$(CC) $(COPTS) $(CFLAGS) -o xg11 $(TOBJ) $(LIBS)		

xtb_m:	xtb/xtb*.[ch]
#		sh -c "cd $(MAKEBDIR)/xtb ; cmake Makefile CLEVEL=$(_CLEVEL) $(XTB)"
#		( cd $(MAKEBDIR)/xtb ; make "CLEVEL=$(_CLEVEL)" $(XTB) )
		( cd xtb ; cmake xtb COMP=$(COMP) CXXCOMP=$(CXXCOMP) "ARCH=$(ARCH)" CHECK=$(CHECK) CLEVEL=$(CLEVEL) DEBUG=$(DEBUG) UNIBIN=$(UNIBIN) $(XTB) )
# 		( cd xtb ; cmake xtb COMP=$(COMP) CXXCOMP=$(CXXCOMP) "ARCH=$(ARCH)" CHECK=$(CHECK) CLEVEL=$(CLEVEL) DEBUG=-gOpt $(XTB) )

ux11_m:	ux11/ux11*.[ch]
#		sh -c "cd $(MAKEBDIR)/ux11 ; make CLEVEL=$(_CLEVEL) $(UX11)"
#		( cd $(MAKEBDIR)/ux11 ; make "CLEVEL=$(_CLEVEL)" $(UX11) )
		( cd ux11 ; cmake ux11 COMP=$(COMP) CXXCOMP=$(CXXCOMP) "ARCH=$(ARCH)" CHECK=$(CHECK) CLEVEL=$(CLEVEL) DEBUG=$(DEBUG) UNIBIN=$(UNIBIN) $(UX11) )
		touch ux11_m

xgtest:		xgtest.o
		$(CC) $(COPTS) $(CFLAGS) -o xgtest xgtest.o -lm

constants.so: constants.c $(DYMOD_DEPHEADERS)
	$(DCC) $(STRIP) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

stats.so: stats.c LineCircle.c $(DYMOD_DEPHEADERS) xgraph.h
	$(DCC) $(STRIP) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

utils.so: utils.c LineCircle.c $(DYMOD_DEPHEADERS) xgraph.h
	$(DCC) $(STRIP) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< $(lX11) #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

strings.so: strings.c $(DYMOD_DEPHEADERS) xgraph.h
	$(DCC) $(STRIP) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

CMaps.so: CMaps.c $(DYMOD_DEPHEADERS)
	$(DCC) $(STRIP) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)
	-cp -p scripts/CMaps.xg $(PREFSDIR)

Python.so: $(PYTHON_DM_NAME)
	-rm -f $@
	ln -s $+ $@ 
	mkdirhier $(PREFSDIR)
	-rm -f $(PREFSDIR)/$@ 
	-ln -s $+ $(PREFSDIR)/$@ 

$(PYTHON_DM_NAME): $(PYTHONSRC:.c=.$(PYTHONVERSION).o)
	$(DCC) $(STRIP) $(DEBUG) $(COPTS) $(CFLAGS) $(X11LIB) $(shell env ./machdepLDOPTS Python$(PYTHONVERSION)Module $@ $(PYTHONDIR)) -I./ -o $@ $+ #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

Python/Python_headers.h: Python/python$(PYTHONVERSION)_numpy.h

Python/python$(PYTHONVERSION)_numpy.h:
	Python/machdep_header $(PYTHONVERSION) numpy

Python/Python.$(PYTHONVERSION).o: Python/Python.c Python/DM_Python.h Python/PythonInterface.h Python/PyObjects.h $(DYMOD_DEPHEADERS) xgraph.h Python/Python_headers.h
	$(DCC) -DDEBUG $(DEBUG) $(COPTS) $(CFLAGS) $(SHOBJ) -I./ -I${PYTHONINC} -o $@ $(CHECK) $< -DPYTHON$(PYTHONVERSION)

Python/AscanfCall.$(PYTHONVERSION).o: Python/AscanfCall.c Python/DM_Python.h Python/PyObjects.h $(DYMOD_DEPHEADERS) xgraph.h Python/Python_headers.h
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(SHOBJ) -I./ -I${PYTHONINC} -o $@ $(CHECK) $< -DPYTHON$(PYTHONVERSION)

Python/DataSet.$(PYTHONVERSION).o: Python/DataSet.c Python/DM_Python.h Python/PyObjects.h $(DYMOD_DEPHEADERS) xgraph.h Python/Python_headers.h
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(SHOBJ) -I./ -I${PYTHONINC} -o $@ $(CHECK) $< -DPYTHON$(PYTHONVERSION)

Python/ULabel.$(PYTHONVERSION).o: Python/ULabel.c Python/DM_Python.h Python/PyObjects.h $(DYMOD_DEPHEADERS) xgraph.h Python/Python_headers.h
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(SHOBJ) -I./ -I${PYTHONINC} -o $@ $(CHECK) $< -DPYTHON$(PYTHONVERSION)

dm_example.so: dm_example.c dymod.h ascanf.h ascanfc.o dymod_interface.h
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

# 20010710: fig_dist.so depends on (the presence) of simanneal.so . It can however also include simanneal.o as an
# "internal" module!
# 20051214: opted for that latter approach!

contrib: simanneal.so fig_dist.so pearson_correlation.so integrators.so splines.so fourconv3.so fourconv3f.so CMaps.so

FDSRC = contrib/fig_dist.c # contrib/simanneal.c

fourconv.so: contrib/fourconv.c $(DYMOD_DEPHEADERS)
	-$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

# 'float' version: fftw accelerated calls might be faster but of course less precise.
fourconv3f.so: contrib/fourconv3.c $(DYMOD_DEPHEADERS) arrayvops.cpp
	$(DCC) -DFFTW_SINGLE $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ arrayvops.cpp $< -lstdc++ #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

## compile the 'double' variant with arrayvops.cpp to have access to the macstl accelerated stdext::valarray<double> type.
## (gcc will switch to C++ compilation for .cpp files, so no need to specify a C++ compiler - which means we don't have
##  to adapt all C and header code to support C linkage!)
fourconv3.so: contrib/fourconv3.c $(DYMOD_DEPHEADERS) arrayvops.cpp
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ arrayvops.cpp $< -lstdc++ #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

splines.so: contrib/splines.c $(DYMOD_DEPHEADERS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $< #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM $@ $(PREFSDIR)

simanneal.so: contrib/simanneal.c contrib/simanneal.h $(DYMOD_DEPHEADERS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ contrib/simanneal.c #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM simanneal.so $(PREFSDIR)

contrib/simanneal.o: contrib/simanneal.c contrib/simanneal.h $(DYMOD_DEPHEADERS)
	$(DCC) $(COPTS) $(CFLAGS) -I. $(SHOBJ) -o $@ $(CHECK) $<

integrators.so: contrib/integrators.c $(DYMOD_DEPHEADERS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ contrib/integrators.c #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM integrators.so $(PREFSDIR)

contrib/integrators.o: contrib/integrators.c $(DYMOD_DEPHEADERS)
	$(DCC) $(COPTS) $(CFLAGS) -I. $(SHOBJ) -o $@ $(CHECK) $<

ddeltaNEC.so: contrib/ddeltaNEC.c $(DYMOD_DEPHEADERS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ contrib/ddeltaNEC.c #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM ddeltaNEC.so $(PREFSDIR)

# 20010710: link with simanneal.so. This hardly alters fig_dist.so, but it mentions to the loader that this shared
# library depends on another. That way, one gets a clean runtime error (can't find simanneal.so) when the dependencies
# can't be resolved. When simanneal.so is NOT linked, the programme will abort in that case while attempting to load
# fig_dist.so!!
fig_dist.so: contrib/fig_dist.o # simanneal.so # contrib/simanneal.o
# 	$(DCC) $(DEBUG) $(COPTS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $(FDSRC:.c=.o) $(PREFSDIR)/simanneal.so #$(OTHERLIBS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ $(FDSRC:.c=.o) #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM fig_dist.so $(PREFSDIR)

contrib/fig_dist.o: contrib/fig_dist.c contrib/simanneal.h dymod.h xgraph.h ascanf.h dymod_interface.h
	$(DCC) $(COPTS) $(CFLAGS) -I. $(SHOBJ) -o $@ $(CHECK) $<

pearson_correlation.so: contrib/pearson_correl.o # simanneal.so
# 	$(DCC) $(DEBUG) $(COPTS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ contrib/pearson_correl.o $(PREFSDIR)/simanneal.so #$(OTHERLIBS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ contrib/pearson_correl.o #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM pearson_correlation.so $(PREFSDIR)

contrib/pearson_correl.o: contrib/pearson_correl.c contrib/simanneal.h dymod.h xgraph.h ascanf.h dymod_interface.h
	$(DCC) $(COPTS) $(CFLAGS) -I. $(SHOBJ) -o $@ $(CHECK) $<

import: GSRio.so IEFio.so CSVio.so

GSRio.so: Import/GSRio.c Import/gsr.h $(DYMOD_DEPHEADERS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ Import/GSRio.c #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM GSRio.so $(PREFSDIR)

IEFio.so: Import/IEFio.o
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ Import/IEFio.o #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM IEFio.so $(PREFSDIR)

Import/IEFio.o: Import/IEFio.c Import/ief.h $(DYMOD_DEPHEADERS) NaN.h
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) -I. $(SHOBJ) -o $@ $(CHECK) $<

CSVio.so: Import/CSVio.c $(DYMOD_DEPHEADERS)
	$(DCC) $(DEBUG) $(COPTS) $(CFLAGS) $(shell ./machdepLDOPTS shlib $@) -I./ -o $@ Import/CSVio.c #$(OTHERLIBS)
	mkdirhier $(PREFSDIR)
	-scripts/lnDM CSVio.so $(PREFSDIR)

clean:
		-rm -rf .finf .rsrc */.finf */.rsrc tmp all All *.dSYM
		-rm -f core tags *.out $(SRC:.c=.o) *.so $(TARGET) $(TARGET).new $(TARGET).prev* $(TARGET).bin* $(TARGET).dbin* xgtest.o xgtest xgtest.binout $(SRC:.c=.o.gz) *.o *.o.gz *.a *.i *.s *.b *~ *.cer* *.ps *.pspreview* *.ps.* *.cg* *.ss* kk* emsg/* emsg/..msg emsg/..err emsg/*/* emsg/*/..msg xtb_m ux11_m make_archive debug/* *.swp_* .*.swp */*.swp_* */.*.swp *.old *.prev *.prev.* *.err *.cer* NR*.pdf examples/*.xg.xg* examples/AntSearch*.xg examples/sgt3_*.xg examples/*.xg.ps examples/*.pdf *.xg *.xg.* gmon* contrib/*.o Import/*.o tim-asc-parm chkmap chkalign chkndn cpu_cycles_per_second.h xgraph.gcc[0-9]* $(PYTHONSRC:.c=.*.o)
		-(cd xtb; make clean)
		-(cd ux11; make clean)
		-(cd vscanf; make clean)
		-du -hc

remake:
		-rm -rf *.dSYM
		-rm -f core $(SRC:.c=.o) $(TARGET) $(TARGET).prev* $(TARGET).bin* xgtest.o xgtest $(SRC:.c=.o.gz) *.a *~ *.cer* emsg/* emsg/*/* xtb_m ux11_m *.swp_* .*.swp *.so *.dylib contrib/*.o Import/*.o ascanfcMap.o arrayvops.o $(PYTHONSRC:.c=.*.o)
		-(cd xtb; make clean)
		-(cd vscanf; make clean)
		-(cd ux11; make clean)
		-du -hc
