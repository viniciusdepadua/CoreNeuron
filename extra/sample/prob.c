#include <math.h>
#include "init.h"
#include "prob.h"

#if defined(_CRAYC)
#define _PRAGMA_FOR_VECTOR_LOOP_ _Pragma("_CRI ivdep")
#elif defined(__PGI)
#define _PRAGMA_FOR_VECTOR_LOOP_ _Pragma("vector")
#endif

#define _PRAGMA_FOR_STATE_ACC_LOOP_ _Pragma("acc parallel loop present(vec_v[0:cnt], nodeindex[0:cnt], nt_data[0:nt->_ndata], p[0:cnt*ml->szp], ppvar[0:cnt*ml->szdp])")
#define _PRAGMA_FOR_CUR_ACC_LOOP_ _Pragma("acc parallel loop present(vec_v[0:cnt], vec_d[0:cnt], vec_rhs[0:cnt], nodeindex[0:cnt], nt_data[0:nt->_ndata], p[0:cnt*ml->szp], ppvar[0:cnt*ml->szdp])")

void prob_state(int mecindex, NrnThread *nt) {

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


    double * __restrict__ p0 =  &p[20*cnt];
    double * __restrict__ p1 =  &p[21*cnt];
    double * __restrict__ p2 =  &p[22*cnt];
    double * __restrict__ p3 =  &p[23*cnt];

    _PRAGMA_FOR_VECTOR_LOOP_
    _PRAGMA_FOR_STATE_ACC_LOOP_
    for (i = 0; i < cnt; ++i) {
        p[20*cnt+i] = p[20*cnt+i] * p[13*cnt+i];
        p[21*cnt+i] = p[21*cnt+i] * p[14*cnt+i];
        p[22*cnt+i] = p[22*cnt+i] * p[15*cnt+i];
        p[23*cnt+i] = p[23*cnt+i] * p[16*cnt+i];
    }


}


void prob_cur(int mecindex, NrnThread *nt) {

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

    double gmax = 0.001;

    _PRAGMA_FOR_CUR_ACC_LOOP_
    for (i = 0; i < cnt; ++i) {
        int idx = nodeindex[i];
        v = vec_v[idx];

        double mfact =  1.e2/(nt_data[ppvar[0*cnt+i]]);

        double lmggate = 1.0 / ( 1.0 + exp ( 0.062 * - ( v ) ) * ( p[8*cnt+i] / 3.57 ) );
        double lg_AMPA = gmax * ( p[21*cnt+i] - p[20*cnt+i]);
        double lg_NMDA = gmax * ( p[23*cnt+i] - p[22*cnt+i]) * lmggate;
        double lg = lg_AMPA + lg_NMDA;
        double lvve = ( v - p[7*cnt+i] );
        double li_AMPA = lg_AMPA * lvve;
        double li_NMDA = lg_NMDA * lvve;
        double li = li_AMPA + li_NMDA;

        double rhs = li;
        lg *=  mfact;
        rhs *= mfact;

        #pragma acc atomic update
        vec_rhs[idx] -= rhs;

        #pragma acc atomic update
        vec_d[idx] += lg;

    }
}

