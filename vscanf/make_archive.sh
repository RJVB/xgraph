#!/bin/sh

# cd $HOME/cworks
cd "`dirname $0`"/..

if [ ! -d vscanf ] ;then
	echo "$0: can't find my own directory..."
	exit 1
fi

CleanUp(){
	echo "Removing temp copy of vscanf directory"
	rm -rf $HOME/work/Archive/vscanf $HOME/work/Archive/VSC.tar &
	exit 2
}

trap CleanUp 1
trap CleanUp 2
trap CleanUp 15

OS="`uname`"
if [ "$OS" = "Linux" -o "$OS" = "linux" -o "$OS" = "LINUX" ] ;then
	echo -n "Making temp copy of vscanf directory..."
	cp -prd vscanf $HOME/work/Archive
	echo " done."
	cd $HOME/work/Archive
else
	echo -n "Making temp copy of vscanf directory (tar to preserve symb. links).."
# 	cp -pr vscanf ../Archive
	gnutar -cf $HOME/work/Archive/VSC.tar vscanf
	sleep 1
	echo "(untar).."
	cd $HOME/work/Archive
	gnutar -xf VSC.tar
fi

sleep 1
echo "Cleaning out the backup copy"
cd vscanf
gunzip -v *.gz 
bunzip2 -v *.bz2
nice make clean
cd ..

trap "" 1
trap "" 2
trap "" 15

sleep 1
pwd
gnutar -cf vscanf.tar vscanf
bzip2 -vf vscanf.tar
ll vscanf.tar.bz2

trap CleanUp 1
trap CleanUp 2
trap CleanUp 15
echo "Cleaning up"
rm -rf vscanf VSC.tar

trap 1
trap 2
trap 15

echo "Making remote backups - put your commands here"
