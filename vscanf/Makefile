CLEVEL=
CHECK=-c
DEBUG=
LASTOPTIONS=
_CLEVEL=$(CLEVEL)
_CFLAGS=$(_CLEVEL) $(DEBUG) -Q
CFLAGS=$(_CFLAGS) $(LASTOPTIONS)
DEBUGSUPPORT= #-DDEBUGSUPPORT
COMP=gccopt
CC=$(COMP)
COMPRESS = -Zbg

SRC = vfscanf.c vsscanf.c vscanf.c tstscanf.c ../snprintf/*snprintf-R162.c

# build rule suitable for gccopt (that wants the target defined before the source):
.c.o:
	$(CC) $(CFLAGS) -o $@ $(CHECK) $<

all: libvscanf.a ../snprintf/snprintf-R162.o ../snprintf/vsnprintf-R162.o tstscanf tags

# libvscanf.a: vfscanf.o vsscanf.o vscanf.o vsnprintf.o
libvscanf.a: vfscanf.o vsscanf.o vscanf.o ../snprintf/snprintf-R162.o ../snprintf/vsnprintf-R162.o
	update_lib $@ $?
	-ranlib $@
	-_obj_compress 1 gzip -9v gz "libvscanf.a" $?

../snprintf/snprintf-R162.o: ../snprintf/snprintf-R162.c
	( cd ../snprintf ; make CC=$(CC) CFLAGS= ) && touch snprintf

../snprintf/vsnprintf-R162.o: ../snprintf/vsnprintf-R162.c
	( cd ../snprintf ; make CC=$(CC) CFLAGS= ) && touch snprintf

tstscanf: tstscanf.c libvscanf.a
	$(CC) -g -o $@ $< -L./ -lvscanf

tags: *.[ch]
	ctags -wmtu $?

clean:
	-rm -rf .finf .rsrc */.finf */.rsrc
	-rm -f core a.out tstscanf $(SRC:.c=.o) $(SRC:.c=.o.gz) *.o *.o.gz *.a *.i *.b *~ *.cer* kk* emsg/* emsg/..msg emsg/..err emsg/*/* emsg/*/..msg make_archive *.swp_* .*.swp *.old *.prev *.prev.* *.err *.cer* gmon* snprintf
	-(cd ../snprintf ; make clean )
	-du
