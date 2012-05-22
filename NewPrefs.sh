cd $HOME

PREFDIR=".Preferences"

if [ -d ${PREFDIR}/.xgraph ] ;then
	exit 0
fi

set -x

if [ ! -d ${PREFDIR} ] ;then
	mkdir ${PREFDIR}
fi

if [ ! -d ${PREFDIR}/.xgraph ] ;then
	if [ -d .xgraph ] ;then
		tar -cf - .xgraph | ( cd ${PREFDIR} ; tar -xf - )
		if [ $? = 0 ] ;then
			echo rm .xgraph
		fi
	else
		mkdir ${PREFDIR}/.xgraph
	fi
fi
