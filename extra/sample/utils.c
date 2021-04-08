#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

struct timeval tvBegin, tvEnd, tvDiff;

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}


#ifdef NOHPM 

void HPM_Start(char *label) {
    printf("\n Starting %s ", label);
    // do nothing
}

void HPM_Stop(char *label) {
    printf("\n Finishing %s ", label);
    // do nothing
}

#endif

