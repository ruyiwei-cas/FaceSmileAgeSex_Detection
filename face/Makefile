########################################
#                                      #
#  Generic Makefile for C/C++ Program  #
#                                      #
#   Email:  yangzhou.du(at)gmail.com   #
#     rev:  George Foot, whyglinux     #
#                                      #
########################################   

# Customising
#
CC       := g++
TARGET   := ../bin/facelib.dll
SRCDIRS  := . ./feature ./faceutil ../3thparty/fann/src
SRCEXCL  := ../3thparty/fann/src/doublefann.c ../3thparty/fann/src/fixedfann.c ../3thparty/fann/src/floatfann.c
MACROS   := -D _WIN32 -D WIN32 -D NDEBUG -D _WINDOWS -D _USRDLL -D FACELIB_EXPORTS -D _MSC_EXTENSIONS
MTUNES   := -O3 -mmmx -msse -msse2 -msse3 -mtune=i686 -fopenmp
LDFLAGS  := -shared 
INCFLAGS := -I../3thparty/opencv200/include -I../3thparty/fann/include -I../3thparty/fftw/include -I../face -I../face/faceutil -I../face/feature -I../3thparty/pthread_win32/include
LIBFLAGS := -lstdc++ -lgomp ../3thparty/opencv200/lib/libcxcore200.dll.a ../3thparty/opencv200/lib/libcv200.dll.a ../3thparty/opencv200/lib/libcvaux200.dll.a ../3thparty/opencv200/lib/libhighgui200.dll.a ../3thparty/opencv200/lib/libml200.dll.a ../3thparty/fftw/lib/fftw3f.lib ../3thparty/pthread_win32/lib/pthreadVC2.lib

# You shouldn't need to change anything below this point.
# 
SRCEXTS  := .c .C .cc .cpp .CPP .c++ .cxx .cp
SOURCES  := $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,$(SRCEXTS))))
SOURCES  := $(filter-out $(SRCEXCL), $(SOURCES))
OBJS     := $(addsuffix .o, $(basename $(SOURCES)))
DEPS     := $(OBJS:%.o=%.d)

CPPFLAGS += -MMD -Wall 
CFLAGS   += $(MACROS) $(MTUNES) $(INCFLAGS)
CXXFLAGS += $(CFLAGS)
LDFLAGS  += 

.PHONY : build deps objs clean rebuild

build : $(TARGET)

deps : $(DEPS)

objs : $(OBJS)

clean :
	$(RM) $(OBJS) $(DEPS) $(TARGET)

rebuild : clean build 

sinclude $(DEPS)

$(TARGET) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBFLAGS)
