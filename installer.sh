#!/bin/bash

source=$1
install_location=$2

if [ $# -lt 2 ]
then
	echo "Usage: installer.sh <source> <install location>"
	exit
fi

if [ ! -d $source ]
then
	echo "Source folder not found. Please check the location is correct and try again."
	exit
fi

if [ ! -d $install_location ]
then
	mkdir $install_location
fi

echo "Generating cmake file..."

cmkfile=`mktemp`

echo "module purge" >> $cmkfile
echo "module load cmake/3.12.2" >> $cmkfile
echo "module load gcc/6.2.0 cuda/10.0 openmpi/3.1.3 # hdf5/1.10.2 silo/4.10.2" >> $cmkfile
echo "" >> $cmkfile
echo "export LBPM_SOURCE=\"$source\"" >> $cmkfile
echo "export LBPM_DIR=\"$install_location\"" >> $cmkfile
echo "" >> $cmkfile
echo "cd \$LBPM_DIR" >> $cmkfile
echo "rm -rf Cmake*" >> $cmkfile
echo "cmake                                    \\" >> $cmkfile
echo "    -D CMAKE_BUILD_TYPE:STRING=Release    \\" >> $cmkfile
echo "    -D CMAKE_CXX_COMPILER:PATH=mpicxx        \\" >> $cmkfile
echo "    -D CMAKE_C_FLAGS=\"-fPIC\"            \\" >> $cmkfile
echo "    -D CMAKE_CXX_FLAGS=\"-fPIC\"          \\" >> $cmkfile
echo "    -D USE_MPI=1            \\" >> $cmkfile
echo "    \$LBPM_SOURCE" >> $cmkfile
echo "" >> $cmkfile
echo "make -j32 && make install && ctest" >> $cmkfile

echo "Executing cmake file..."

cmake $cmkfile


echo "Installing launcher dependencies..."
conda install --file $source/launcher/requirements.txt
pip install connected-components-3d
