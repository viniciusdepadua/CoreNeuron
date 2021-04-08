#ifndef _nah_
#define _nah_

#include "init.h"
#include "utils.h"

void na_cur(int mecindex, NrnThread *nt);
void na_state(int mecindex, NrnThread *nt);
void na_ecur(int mecindex, NrnThread *nt);
void na_estate(int mecindex, NrnThread *nt);

#endif // _nah_
