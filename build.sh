#!/bin/bash
# compile facelib dll using mingw32

# generate "./bin/facelib.so.dll"
make -C ./face rebuild

# generate "./bin/facelib.so.lib"
pexports ./bin/facelib.dll > ./bin/facelib.def
dlltool -d ./bin/facelib.def -D ./bin/facelib.dll -l ./bin/facelib.lib
