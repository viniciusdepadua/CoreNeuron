#include <math.h>
#include "init.h"
#include "na.h"

#if defined(_CRAYC)
#define _PRAGMA_FOR_VECTOR_LOOP_ _Pragma("_CRI ivdep")
#elif defined(__PGI)
#define _PRAGMA_FOR_VECTOR_LOOP_ _Pragma("vector")
#endif

#define _PRAGMA_FOR_STATE_ACC_LOOP_ _Pragma("acc parallel loop present(vec_v[0:cnt], nodeindex[0:cnt], nt_data[0:nt->_ndata], p[0:cnt*ml->szp], ppvar[0:cnt*ml->szdp])")
#define _PRAGMA_FOR_CUR_ACC_LOOP_ _Pragma("acc parallel loop present(vec_v[0:cnt], vec_d[0:cnt], vec_rhs[0:cnt], nodeindex[0:cnt], nt_data[0:nt->_ndata], p[0:cnt*ml->szp], ppvar[0:cnt*ml->szdp])")

void na_state(int mecindex, NrnThread *nt) {

    Mechanism * ml;
    double v;
    int i;
    double dt = nt->dt;

    ml = &(nt->ml[mecindex]);

    int *nodeindex = ml->nodeindices;
    int cnt = ml->n;
    double * __restrict__ p = ml->data;
    int * __restrict__ ppvar = ml->pdata;
    double * __restrict__ vec_v = nt->_actual_v;
    double * __restrict__ nt_data = nt->_data;

    _PRAGMA_FOR_VECTOR_LOOP_
    _PRAGMA_FOR_STATE_ACC_LOOP_
    for (i = 0; i < cnt; ++i) {
        double v = vec_v[nodeindex[i]];
        p[3*cnt+i] = nt_data[ppvar[0*cnt+i]];
        double lqt = 2.952882641412121 ;
        if ( v == - 32.0 ) {
            v = v + 0.0001 ;
        }
        double lmAlpha = ( 0.182 * ( v - - 32.0 ) ) / ( 1.0 - ( exp ( - ( v - - 32.0 ) / 6.0 ) ) );
        double lmBeta = ( 0.124 * ( - v - 32.0 ) ) / ( 1.0 - ( exp ( - ( - v - 32.0 ) / 6.0 ) ) );
        double lmInf = lmAlpha / ( lmAlpha + lmBeta ) ;
        double lmTau = ( 1.0 / ( lmAlpha + lmBeta ) ) / lqt ;
        p[1*cnt+i] = p[1*cnt+i] + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / lmTau)))*(- ( ( ( lmInf ) ) / lmTau ) / ( ( ( ( - 1.0) ) ) / lmTau ) - p[1*cnt+i]) ;
        if ( v == - 60.0 ) {
            v = v + 0.0001 ;
        }
        double lhAlpha = ( - 0.015 * ( v - - 60.0 ) ) / ( 1.0 - ( exp ( ( v - - 60.0 ) / 6.0 ) ) );
        double lhBeta = ( - 0.015 * ( - v - 60.0 ) ) / ( 1.0 - ( exp ( ( - v - 60.0 ) / 6.0 ) ) );
        double lhInf = lhAlpha / ( lhAlpha + lhBeta ) ;
        double lhTau = ( 1.0 / ( lhAlpha + lhBeta ) ) / lqt ;
        p[2*cnt+i] = p[2*cnt+i] + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / lhTau)))*(- ( ( ( lhInf ) ) / lhTau ) / ( ( ( ( - 1.0) ) ) / lhTau ) - p[2*cnt+i]) ;
    }

}


void na_cur(int mecindex, NrnThread *nt) {

    Mechanism * ml;
    double v;
    int i;
    double dt = nt->dt;

    ml = &(nt->ml[mecindex]);

    int *nodeindex = ml->nodeindices;
    int cnt = ml->n;
    double * __restrict__ p = ml->data;
    int * __restrict__ ppvar = ml->pdata;
    double * __restrict__ vec_v = nt->_actual_v;
    double * __restrict__ vec_d = nt->_actual_d;
    double * __restrict__ vec_rhs = nt->_actual_rhs;
    double * __restrict__ nt_data = nt->_data;

    _PRAGMA_FOR_CUR_ACC_LOOP_
    for (i = 0; i < cnt; ++i) {
        int idx = nodeindex[i];
        v = vec_v[idx];

        p[3*cnt+i] = nt_data[ppvar[0*cnt+i]];
        double lgNaTs2_t = p[0*cnt+i] * p[1*cnt+i] * p[1*cnt+i] * p[1*cnt+i] * p[2*cnt+i] ;
        double lina = lgNaTs2_t * ( v - p[3*cnt+i] ) ;
        double rhs = lina;
        double g = lgNaTs2_t;
        nt_data[ppvar[2*cnt+i]] += lgNaTs2_t;
        nt_data[ppvar[1*cnt+i]] += lina ;
        vec_rhs[idx] -= rhs;
        vec_d[idx] += g;
    }
}

