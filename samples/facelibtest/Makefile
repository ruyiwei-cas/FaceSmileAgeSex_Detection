########################################
#                                      #
#  Generic Makefile for C/C++ Program  #
#                                      #
######################################## 

# Customising
#
FACEDIR  := ../../face
UNIXDIR  := ../../3thparty/linux
CVDIR    := $(UNIXDIR)/opencv

# GCC compiler
ifeq ($(DEBUG), debug)
TARGET   := ../../bin/testfacelibd
MTUNES   := -g -O0
else
TARGET   := ../../bin/testfacelib
MTUNES   := -g -mmmx -msse -msse2 -mtune=i686
endif

LIBDIR   := ../../bin/
INCFLAGS := -I$(CVDIR)/include/ -I$(FACEDIR)/ -I$(FACEDIR)/faceutil
LIBFLAGS := -L$(CVDIR)/lib -ltiff -ljasper -lavcodec -lavformat -lavutil -lswscale -lcxcore -lcv -lhighgui -lml -lbz2 -L$(LIBDIR) -lface -lstdc++

#OpenMP supported based-on GCC
ifeq ($(OMP),gomp)
MTUNES   += -fopenmp
LIBFLAGS += -lgomp -lpthread -lfftw3f_threads
endif

#SOURCES  := testfacelib.cpp main.cpp
SRCDIRS  := ./

# You shouldn't need to change anything below this point.
# 
SRCEXTS  := .c .C .cc .cpp .CPP .c++ .cxx .cp
SOURCES  := $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,$(SRCEXTS))))
OBJS     := $(addsuffix .o, $(basename $(SOURCES)))
DEPS     := $(OBJS:%.o=%.d)

CPPFLAGS += -MMD -Wall
CFLAGS   += $(MACROS) $(MTUNES) $(INCFLAGS)
CXXFLAGS += $(CFLAGS)
LDFLAGS  += $(LIBFLAGS)

.PHONY : build deps objs clean rebuild

build : $(TARGET)

deps : $(DEPS)

objs : $(OBJS)

clean :
	$(RM) $(OBJS) $(DEPS) $(TARGET)

rebuild : clean build

sinclude $(DEPS)

$(TARGET) : $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LDFLAGS)
