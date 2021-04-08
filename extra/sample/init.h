#ifndef _init_
#define _init_

typedef struct Membrane {
    int *pdata;
    double *data;
    int *nodeindices;
    int n;
    int szp;
    int szdp;
    int type;
    int is_art;
    long offset;
} Mechanism;

typedef struct NrnTh{
    int _ndata;
    int end;
    double *_data;
    double *_actual_rhs;
    double *_actual_d;
    double *_actual_a;
    double *_actual_b;
    double *_actual_v;
    double *_actual_area;

    int nmech;
    int ncompartment;
    Mechanism *ml;
    double dt;
} NrnThread;


void read_nt_from_file(char *filename, NrnThread *nt);
void print_iarray(int * data, int n);

#endif // _init_
