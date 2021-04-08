CC = cc.scorep.cray
CC = cc

CFLAGS=-DNOHPM

ifdef CRAY
    #CFLAGS += -O3 -hlist=a -h vector3 -h profile_generate
    CFLAGS += -O2 -hlist=a -h c99
    ifdef GPU
        CFLAGS += -hpragma=acc
    endif
endif


ifdef INTEL
CFLAGS += -vec-report2 -O3 -std=c99 
endif

#GPU VERSION (NVIDIA, dom)
ifdef PGI
    #CFLAGS += -O3 -Minfo -fast -Mipa=fast -Mfprelaxed -Mfpapprox
#NOTE: fastmath wasnt changing any performance. For unified memory, you could add
#       managed to ta flag. But you have to remove memory copy function in init.c
    CFLAGS += -O3 -Minfo
    CC = pgcc
    ifdef GPU
        CFLAGS += -ta=nvidia,cc35,fastmath -Minfo  -acc 
    endif
    ifdef UNIMEM
        CFLAGS += -ta=nvidia,managed -DUNIMEM
    endif
endif


COMMON_OBJ = init.o  main.o  na.o  utils.o prob.o

all: main

%.o: %.c %.h
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(COMMON_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)
	#pat_build -u $@

.PHONY: clean

clean:
	rm -f *.o *~ core main *.lst main+pat *.gpu *.ptx *.cub
