# Default Lapacke: Openblas at CESGA
LDLIBS=-lopenblas

# Other systems (my Debian boxes, for example)
#LDLIBS=-llapacke

# Intel MKL at CESGA
# Module needed: imkl
# => module load openblas
# LDLIBS for intel compiler: icx (module needed: intel)
# Just invoke make like this: make CC=icx
# LDLIBS=-qmkl=sequential -lmkl_intel_lp64


# Compiler options
CC = icc
CFLAGS = 

# Targets for different optimization levels
dgesv_opt0: CFLAGS += -O0
dgesv_opt0: dgesv.o timer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

dgesv_opt1: CFLAGS += -O1
dgesv_opt1: dgesv.o timer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

dgesv_opt2: CFLAGS += -O2
dgesv_opt2: dgesv.o timer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

dgesv_opt3: CFLAGS += -O3 -qopenmp
dgesv_opt3: dgesv.o timer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Rules for the default target
dgesv: dgesv.o timer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Generic rules for object files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) dgesv *.o *~
