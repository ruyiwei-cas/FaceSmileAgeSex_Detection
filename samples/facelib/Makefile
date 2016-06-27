########################################
#                                      #
#  Generic Makefile for C/C++ Program  #
#                                      #
######################################## 

# Customising
#
FACEDIR  := ../../face
FANNDIR  := ../../3thparty/fann
UNIXDIR  := ../../3thparty/linux
CVDIR    := $(UNIXDIR)/opencv

SRCDIRS  := $(FACEDIR) $(FACEDIR)/feature $(FACEDIR)/faceutil ./
SOURCES  := $(FANNDIR)/src/fann.c $(FANNDIR)/src/fann_cascade.c $(FANNDIR)/src/fann_error.c $(FANNDIR)/src/fann_io.c $(FANNDIR)/src/fann_train.c $(FANNDIR)/src/fann_train_data.c

INCFLAGS := -I$(UNIXDIR)/fftw/include -I$(FANNDIR)/include -I$(FACEDIR) -I$(FACEDIR)/feature -I$(FACEDIR)/faceutil
LIBFLAGS := -L$(UNIXDIR)/fftw/lib -lm -lfftw3f 

INCFLAGS += -I$(CVDIR)/include
LIBFLAGS += -L$(CVDIR)/lib -lbz2 -lbz2 -ltiff -ljasper -lavcodec -lavformat -lavutil -lswscale -lcxcore -lcv -lhighgui -lml -lpthread -lstdc++

MACROS   := -D CV_NO_BACKWARD_COMPATIBILITY

ifeq ($(DEBUG), debug)
TARGET   := ../../bin/libfaced.so
MTUNES   := -g -O0
else
TARGET   := ../../bin/libface.so
MTUNES   := -mmmx -msse -msse2 -mtune=i686
endif

#OpenMP supported based-on GCC
ifeq ($(OMP),gomp)
MTUNES   += -fopenmp
LIBFLAGS += -lgomp -lpthread -lfftw3f_threads
endif

# You shouldn't need to change anything below this point.
# 
SRCEXTS  := .c .C .cc .cpp .CPP .c++ .cxx .cp
SOURCES  += $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,$(SRCEXTS))))
OBJS     := $(addsuffix .o, $(basename $(SOURCES)))
DEPS     := $(OBJS:%.o=%.d)

CPPFLAGS += -MMD -Wall
CFLAGS   += $(MACROS) $(MTUNES) $(INCFLAGS)
CXXFLAGS += $(CFLAGS)
LDFLAGS  += $(LIBFLAGS) -shared

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
