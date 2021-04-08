#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "init.h"

void print_iarray(int * data, int n) {
    int i = 0;
    printf ("\n N = %d \n", n);

    for(i = 0; i< n ; i++)
        printf("\t %d", data[i]);
    printf("\n");
}

void read_darray_from_file(FILE *hFile, double *data, int n) {
    int i;
    char buf[4096];

    for(i=0; i<n; i++) {
        fscanf(hFile, "%lf\n", &data[i]);
    }
    fscanf(hFile, "%s", buf);
    //printf("\n%s", buf);
}

void read_iarray_from_file(FILE *hFile, int *data, int n) {
    int i;
    char buf[4096];

    for(i=0; i<n; i++) {
        fscanf(hFile, "%d\n", &data[i]);
    }
    fscanf(hFile, "%s", buf);
    //printf("\n%s", buf);
}

#ifdef _OPENACC
#include <openacc.h>
#endif

void copy_nt_on_gpu(NrnThread *nt) {

#ifdef _OPENACC
#ifndef UNIMEM

    NrnThread *d_nt;

    /* -- copy NrnThread to device. this needs to be contigious vector because offset is used to find
     * corresponding NrnThread using Point_process in NET_RECEIVE block
     */
    d_nt = (NrnThread *) acc_copyin(nt, sizeof(NrnThread));

    printf("\n --- Copying to Device! --- ");

    double *d__data;                // nrn_threads->_data on device

    /*copy all double data for thread */
    d__data = (double *) acc_copyin(nt->_data, nt->_ndata*sizeof(double));

    /*update d_nt._data to point to device copy */
    acc_memcpy_to_device(&(d_nt->_data), &d__data, sizeof(double*));

    /* -- setup rhs, d, a, b, v, node_aread to point to device copy -- */
    double *dptr;

    dptr = d__data + 0*nt->end;
    acc_memcpy_to_device(&(d_nt->_actual_rhs), &(dptr), sizeof(double*));

    dptr = d__data + 1*nt->end;
    acc_memcpy_to_device(&(d_nt->_actual_d), &(dptr), sizeof(double*));

    dptr = d__data + 2*nt->end;
    acc_memcpy_to_device(&(d_nt->_actual_a), &(dptr), sizeof(double*));

    dptr = d__data + 3*nt->end;
    acc_memcpy_to_device(&(d_nt->_actual_b), &(dptr), sizeof(double*));

    dptr = d__data + 4*nt->end;
    acc_memcpy_to_device(&(d_nt->_actual_v), &(dptr), sizeof(double*));

    dptr = d__data + 5*nt->end;
    acc_memcpy_to_device(&(d_nt->_actual_area), &(dptr), sizeof(double*));

    Mechanism *d_mlist;
    d_mlist = acc_copyin(nt->ml, nt->nmech*sizeof(Mechanism));

    acc_memcpy_to_device(&(d_nt->ml), &(d_mlist), sizeof(Mechanism*));

    long int offset = 6*nt->end;
    int i;
    for (i=0; i<nt->nmech; i++) {

        Mechanism *ml = &nt->ml[i];
        Mechanism *d_ml = &d_mlist[i];
       
        dptr = d__data+offset; 
        acc_memcpy_to_device(&(d_ml->data), &(dptr), sizeof(double*));

        if (!ml->is_art) {
            int * d_nodeindices = (int *) acc_copyin(ml->nodeindices, sizeof(int)*ml->n);
            acc_memcpy_to_device(&(d_ml->nodeindices), &d_nodeindices, sizeof(int*));

        }

        if (ml->szdp) {
            int * d_pdata = (int *) acc_copyin(ml->pdata, sizeof(int)*ml->n*ml->szdp);
            acc_memcpy_to_device(&(d_ml->pdata), &d_pdata, sizeof(int*));
        }

        offset += ml->n * ml->szp;
    }

#endif    
#endif

}

void read_nt_from_file(char *filename, NrnThread *nt) {

    int i;
    long int offset;
    FILE *hFile;
    int ne;

    hFile = fopen(filename, "r");

    nt->dt = 0.025;

    fscanf(hFile, "%d\n", &nt->_ndata);

    //No posix_memalign wrapper captured by PGI for unified memory support
    #ifdef UNIMEM
        nt->_data = malloc(sizeof(double) * nt->_ndata);
    #else
        posix_memalign( (void **) &nt->_data, 64, sizeof(double) * nt->_ndata);
    #endif

    read_darray_from_file(hFile, nt->_data, nt->_ndata);
    
    fscanf(hFile, "%d\n", &nt->ncompartment);
    ne = nt->ncompartment;
    nt->end = ne;

    nt->_actual_rhs = nt->_data + 0*ne;
    nt->_actual_d = nt->_data + 1*ne;
    nt->_actual_a = nt->_data + 2*ne;
    nt->_actual_b = nt->_data + 3*ne;
    nt->_actual_v = nt->_data + 4*ne;
    nt->_actual_area = nt->_data + 5*ne;

    offset = 6*ne;
    fscanf(hFile, "%d\n", &nt->nmech);

    nt->ml = (Mechanism *) calloc(sizeof(Mechanism), nt->nmech);

    for (i=0; i<nt->nmech; i++) {

        Mechanism *ml = &nt->ml[i];
        fscanf(hFile, "%d %d %d %d %d %ld\n", &(ml->type), &(ml->is_art), &(ml->n), &(ml->szp), &(ml->szdp), &(ml->offset));
        ml->data = nt->_data + offset;
        offset += ml->n * ml->szp;

        printf("\n Mech Id %d with Index %d iteration count : %d " , ml->type, i, ml->n);

        if (!ml->is_art) {
            #ifdef UNIMEM
            ml->nodeindices = malloc(sizeof(int) * ml->n);
            #else
            posix_memalign((void **)&ml->nodeindices, 64, sizeof(int) * ml->n);
            #endif
            read_iarray_from_file(hFile, ml->nodeindices, ml->n);
        }

        if (ml->szdp) {
            #ifdef UNIMEM
            ml->pdata = malloc(sizeof(int) * ml->n*ml->szdp);
            #else
            posix_memalign((void **)&ml->pdata, 64, sizeof(int) * ml->n*ml->szdp);
            #endif
            read_iarray_from_file(hFile, ml->pdata, ml->n*ml->szdp);
        }
    }

    copy_nt_on_gpu(nt);

    fclose(hFile);
}

