#ifndef _utils_
#define _utils_

#include <sys/time.h>
#include <time.h>
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

extern void HPM_Start(char *label);
extern void HPM_Stop(char *label);

#endif // _utils_
