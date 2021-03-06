#!/bin/sh

THISHOST="`uname`"
case $THISHOST in
	Linux|Darwin)
		;;
	*)
		THISHOST="`uname -m`"
		;;
esac

DOTP=""
case $THISHOST in 
	Darwin|"Power Macintosh")
		  # this is similar to tcsh's 'symlink chase' option: always use the physical path, not the path as given
		  # by any symlinks in it.
		set -P
		;;
	Linux)
		DOTP="-P"
		;;
esac

cd $DOTP $HOME/cworks

ARCHDIR="$HOME/work/Archive"
mkdir -p "$ARCHDIR"

if [ -x /usr/local/bin/echo ] ;then
	ECHO="/usr/local/bin/echo"
elif [ -x /bin/echo ] ;then
	ECHO="/bin/echo"
fi

CleanUp(){
	${ECHO} "Removing temp copy of xgraph directory"
	rm -rf ${ARCHDIR}/xgraph ${ARCHDIR}/xgraph.tar ${ARCHDIR}/XG_exmp.tar ${ARCHDIR}/XG_examples ${ARCHDIR}/XG.tar &
	exit 2
}

trap CleanUp 1
trap CleanUp 2
trap CleanUp 15

CP="cp"

OS="`uname`"
gcp --help 2>&1 | fgrep -- --no-dereference > /dev/null
if [ $? = 0 ] ;then
	 # hack... gcp must be gnu cp ...
	OS="LINUX"
	CP="gcp"
fi

cp --help 2>&1 | fgrep -- --no-dereference > /dev/null
if [ $? = 0 -o "$OS" = "Linux" -o "$OS" = "linux" -o "$OS" = "LINUX" -o "`uname -o 2>/dev/null`" = "Cygwin" ] ;then
	${ECHO} -n "Making temp copy of xgraph directory..."
	${CP} -prd xgraph ${ARCHDIR}
	${ECHO} " done."
	cd $DOTP ${ARCHDIR}
else
	${ECHO} -n "Making temp copy of xgraph directory (tar to preserve symb. links).."
# 	cp -pr xgraph ${ARCHDIR}
	gnutar -cf ${ARCHDIR}/XG.tar xgraph
	sleep 1
	${ECHO} "(untar).."
	cd $DOTP ${ARCHDIR}
	gnutar -xf XG.tar
fi

sleep 1
${ECHO} "Cleaning out the backup copy"
cd $DOTP xgraph
pwd
gunzip -vf *.gz
gunzip -v examples/*.gz
bunzip2 -vf *.bz2
bunzip2 -v examples/*.bz2
nice make clean
rm -rf old_examples snapshots wis.dat* wisdom.* XGraph-1.moved-aside build *.docset xgraph.i386 .git Python/python*_headers.h Python/python*_numpy.h sse_mathfun/.git
mv tim-asc-parm.c Tim-asc-parm.c
rm tim-asc-parm*
mv Tim-asc-parm.c tim-asc-parm.c
cd $DOTP ..
# mkdir XG_examples
# mv xgraph/*.xg* XG_examples

trap "" 1
trap "" 2
trap "" 15

sleep 1
${ECHO} "Archiving the cleaned backup directory"
gnutar -cf XG_exmp.tar xgraph/examples
rm -rf xgraph/examples
gnutar -cf xgraph.tar xgraph
# gzip -lv xgraph.tar.gz XG_exmp.tar.gz
ll -U xgraph.tar.bz2 XG_exmp.tar.bz2
bzip2 -vf xgraph.tar XG_exmp.tar

trap CleanUp 1
trap CleanUp 2
trap CleanUp 15
${ECHO} "Cleaning up"
rm -rf xgraph XG.tar XG_examples

trap 1
trap 2
trap 15

${ECHO} "Making remote backups - put your commands here"
