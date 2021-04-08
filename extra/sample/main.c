#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "init.h"
#include "na.h"
#include "prob.h"
#include  <openacc.h>

#ifdef MPI
#include <mpi.h>
#endif

//#include <pat_api.h>

int main(int argc, char *argv[])
{
    char fname[4096];
    int i = 0;

#ifdef MPI
    MPI_Init(NULL, NULL);
#endif

    //PAT_record(PAT_STATE_OFF);

    if(argc < 2)
    {
        printf("\n Pass name of file \n");
        exit(1);
    }
    else
    {
        #pragma omp parallel
        {
            printf("\n Thread %d started initialization ", omp_get_thread_num());
            fflush(stdout);
            NrnThread nt;
            read_nt_from_file(argv[1], &nt);

            #pragma omp barrier 

            //PAT_record(PAT_STATE_ON);

          for(int j=0; j<5; j++) {
            #pragma omp master 
            HPM_Start("Na_State");
            na_state(17, &nt);
            #pragma omp master 
            HPM_Stop("Na_State");


            #pragma omp master 
            HPM_Start("Na_Cur");
            na_cur(17, &nt);
            #pragma omp master 
            HPM_Stop("Na_Cur");

            #pragma omp master 
            HPM_Start("Prob_State");
            prob_state(18, &nt);
            #pragma omp master 
            HPM_Stop("Prob_State");


            #pragma omp master 
            HPM_Start("Prob_Cur");
            prob_cur(18, &nt);
            #pragma omp master 
            HPM_Stop("Prob_Cur");
          }

        }
    }
   
   #ifdef _OPENACC
   acc_shutdown( acc_device_default );
   #endif
    
#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}
