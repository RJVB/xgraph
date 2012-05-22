#!/bin/sh
# This script allows 
# 1) XGraph files to be "executed" by starting up xgraph when this can't be done directly
#    because a valid/accepted shell must be used. This is circumvented by making a header
#    #!/bin/sh /usr/local/bin/XGraph
# 2) Passing on arguments contained in the file as if they were typed directly on the commandline.
#    They actually are: XGraph scans for these, and invokes xgraph with the arguments found. Even
#    when the inputfile comes from stdin. Currently, only -MinBitsPPixel and -VisualType are processed in this way,
#    allowing a visual of appropriate depth to be used upon re-invocation.
# 3) Starting all files specified in a separate process by passing the -unique argument.
# 4) Print out the xgraph process id, and return it as the exit value: -pid.

OPTS=""
XGRAPH="`which xgraph`"
ARGS="$*"
OPTIONS=""
FILES=""
INPUT=""
UNIQUE=0
RETURN_PID=0
DUMP=0

# The grep command used to do the initial scan for *ARGUMENT* commands in the input.
# If you use compressed files, you will need a version of grep that can handle those
# (like zgrep that comes with gzip). I patched GNU grep v2.5. to handle compressed files.
GREP=grep

if [ -r "$HOME/Prefences/.xgraph/xg_init.xg" ] ;then
	XGRC="$HOME/Prefences/.xgraph/xg_init.xg"
else
	XGRC=""
fi

if [ "$1" = "-" ] ;then
	shift
	if [ "$TMPDIR" = "" ] ;then
		TMPDIR="/tmp"
	fi
	if [ -d "$TMPDIR" ] ;then
		INPUT="${TMPDIR}/XGraph.tmp$$"
	else
		INPUT="/tmp/XGraph.tmp$$"
	fi
	ARGS="$*"
	cat - > $INPUT
# # scan the newly created inputfile:
# 	/bin/echo -n "Scanning for X11 arguments..." 1>&2
# 	BPP="`grep 'ARGUMENTS.*MinBitsPPixel' $XGRC $INPUT`"
# 	/bin/echo -n "..." 1>&2
# 	VPP="`grep 'ARGUMENTS.*VisualType' $XGRC $INPUT`"
# 	/bin/echo 1>&2
	FILES="${INPUT}"
fi
# Just bluntly scan all the arguments for the supported xgraph options:
	while [ $# != 0 ] ;do
		case $1 in
			-pid)
				  # This is an option to XGraph!
				RETURN_PID=1
				;;
			-no-uniq*)
				  # This is an option to XGraph!
				UNIQUE=0
				;;
			-uniq*)
				  # This is an option to XGraph!
				UNIQUE=1
				;;
			-f|-script)
				OPTIONS="$OPTIONS $1 $2"
				shift
				;;
			-dump-debug)
				DUMP=1
				shift
				;;
			-*)
				OPTIONS="$OPTIONS $1"
				;;
			*)
				if [ -r "$1" ] ;then
					if [ "$FILES" = "" ] ;then
						FILES="$1"
					else
						FILES="$FILES $1"
					fi
				fi
				;;
		esac
		shift
	done
# fi

if [ "$FILES" = "" ] ;then
	echo "No files found on commandline or on disk..." 1>&2
	exit 1
fi

ScanOptions() {
	OPTS=""
	/bin/echo -n "Scanning for X11 arguments..." 1>&2
	OS="`uname`"
	if [ "$OS" = "Linux" -o "$OS" = "linux" -o "${OS}" = "Darwin" ] ;then
		ARGUMENTS="`$GREP -a 'ARGUMENTS' $XGRC $* | tr '\t' ' '`"
		DBE="`echo $ARGUMENTS | grep -a 'ARGUMENTS.*use_XDBE' $XGRC $*`"
		/bin/echo -n "..." 1>&2
		BPP="`echo $ARGUMENTS | grep -a 'ARGUMENTS.*MinBitsPPixel' $XGRC $*`"
		/bin/echo -n "..." 1>&2
		VPP="`echo $ARGUMENTS | grep -a 'ARGUMENTS.*VisualType' $XGRC $*`"
		/bin/echo 1>&2
	else
		ARGUMENTS="`$GREP 'ARGUMENTS' $XGRC $* | tr '\t' ' '`"
		DBE="`echo $ARGUMENTS | grep 'ARGUMENTS.*use_XDBE' $XGRC $*`"
		/bin/echo -n "..." 1>&2
		BPP="`echo $ARGUMENTS | grep 'ARGUMENTS.*MinBitsPPixel' $XGRC $*`"
		/bin/echo -n "..." 1>&2
		VPP="`echo $ARGUMENTS | grep 'ARGUMENTS.*VisualType' $XGRC $*`"
		/bin/echo 1>&2
	fi
	if [ "$BPP" != "" -o "$DBE" != "" ] ;then
# found something: parse it into a valid commandline argument.
		DOPTS=`echo "$DBE" | sed -e 's/.*\(-use_XDBE[^ ]*\) .*/\1/g'`
		BOPTS=`echo "$BPP" | sed -e 's/.*\(-MinBitsPPixel [^ ]*\).*/\1/g'`
		VTYPE=`echo "$BOPTS" | grep 'ARGUMENTS.*VisualType'`
		if [ "$VTYPE" = "" -a "$VPP" != "" ] ;then
			VOPTS=`echo "$VPP" | sed -e 's/.*\(-VisualType [^ ]*\).*/\1/g'`
		fi
		if [ "$DOPTS" != "" ] ;then
			echo "Double Buffer extension: ${DOPTS} ." 1>&2
			OPTS="$OPTS $DOPTS"
		fi
		if [ "$BOPTS" != "" ] ;then
			echo "Visual selection: ${BOPTS} ." 1>&2
			OPTS="$OPTS $BOPTS"
		fi
		if [ "$VOPTS" != "" ] ;then
			echo "Visual class selection: ${VOPTS} ." 1>&2
			OPTS="$OPTS $VOPTS"
		fi
	fi
}

PID=0

if [ "$INPUT" != "" ] ;then
# invoke xgraph with the correct options, and make it read the
# generated inputfile through stdin, as we did ourselves!
# In this case it is safe to start xgraph as a background process:
# the input-file won't easily be removed by our caller (and xgraph
# doesn't have a habit of returning error-codes).
	ScanOptions ${INPUT}
	if [ $RETURN_PID = 0 ] ;then
		echo "${XGRAPH} $XGRC ${OPTS} $ARGS - < $INPUT" 1>&2
		( ${XGRAPH} $XGRC ${OPTS} $ARGS - < $INPUT ; rm -f $INPUT ) &
# 		( ${XGRAPH} $XGRC ${OPTS} $ARGS - < $INPUT ) &
	else
		echo "${XGRAPH} $XGRC ${OPTS} ${OPTIONS} $INPUT -remove_inputfiles" 1>&2
		${XGRAPH} $XGRC ${OPTS} ${OPTIONS} $INPUT -remove_inputfiles &
# 		${XGRAPH} $XGRC ${OPTS} ${OPTIONS} $INPUT &
		PID=$!
		echo "xgraph process id is $PID" 1>&2
	fi
else
	if [ $UNIQUE = 0 ] ;then
		ScanOptions ${FILES}
		if [ $RETURN_PID = 0 ] ;then
			echo "${XGRAPH} ${XGRC} ${OPTS} $ARGS" 1>&2
			exec ${XGRAPH} $XGRC ${OPTS} $ARGS
		else
			echo "${XGRAPH} ${XGRC} ${OPTS} ${OPTIONS} ${FILES}" 1>&2
			${XGRAPH} $XGRC ${OPTS} ${OPTIONS} ${FILES} &
			PID=$!
			echo "xgraph process id is $PID" 1>&2
		fi
	else
	 # No RETURN_PID support in the case of spawning multiple files (even if only 1...)
		WPID=0
		for F in ${FILES} ;do
			ScanOptions ${F}
			if [ "${F}" = "${FILES}" ] ;then
			 # This is an quick'n'easy hack to make that for 1 file only, we don't
			 # put an xgraph in the background, but exec it.
				echo "${XGRAPH} $XGRC ${OPTS} ${OPTIONS} ${F}" 1>&2
				exec ${XGRAPH} $XGRC ${OPTS} ${OPTIONS} ${F}
			else
				echo "( ${XGRAPH} $XGRC ${OPTS} ${OPTIONS} ${F} ) &" 1>&2
				( ${XGRAPH} $XGRC ${OPTS} ${OPTIONS} ${F} ) &
				WPID=$!
			fi
		done
		if [ WPID != 0 ] ;then
			echo "Waiting for $F to be finished (warning: killing this process kills all childs too!)" 1>&2
			wait $WPID
		fi
	fi
fi

if [ $RETURN_PID = 1 ] ;then
	exit $PID
fi
