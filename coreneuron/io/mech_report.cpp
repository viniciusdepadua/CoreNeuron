#include <string.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "coreneuron/nrnconf.h"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/mpi/nrnmpi.h"
#include "coreneuron/mpi/nrnmpi_impl.h"
#include "coreneuron/mpi/nrnmpidec.h"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/coreneuron.hpp"

namespace coreneuron {

#if NRNMPI

void print_mech_count(std::vector<unsigned long long>& mech_count) {
    auto n_memb_func = corenrn.get_memb_funcs().size();
    std::vector<unsigned long long> total_mech_count(n_memb_func);
    /// get global sum of all mechanism instances
    MPI_Allreduce(&mech_count[0], &total_mech_count[0], mech_count.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if(nrnmpi_myid == 0) {
        printf("\n================ MECHANISMS COUNT BY TYPE ==================\n");
        for(int i = 0; i < mech_count.size(); i++) {
            printf("Id, Name, Count :: %4d. %20s %10lld\n", i, nrn_get_mechname(i), total_mech_count[i]);
        }
        printf("=============================================================\n");
    }
}

/** Write mechanism count for every gid using MPI I/O */
void write_mech_report(std::string path, int ngroups, const int* gidgroups) {
	std::string fname(path);
    fname += "/mech.stats";

    /// remove if file already exist
    if(nrnmpi_myid == 0) {
        remove(fname.c_str());
    }
    nrnmpi_barrier();

    /// count no of gids/threads to allocate buffer
    int non_empty_thread_count = 0;
    for(int i=0; i<nrn_nthread; i++) {
        NrnThread& nt = nrn_threads[i];
        if(nt.ncell) {
            non_empty_thread_count++;
        }
    }

    auto n_memb_func = corenrn.get_memb_funcs().size();

    // each column is 64 chars chars (gid followed by all mechanism name)
    const int RECORD_LEN = 64;
    unsigned num_records = 1 + n_memb_func;
    int header = 0;

    /// header is written by first rank only
    if(nrnmpi_myid == 0) {
        header = 1;
    }

    /// allocate memoty
    unsigned num_bytes = (sizeof(char) * num_records * RECORD_LEN * (non_empty_thread_count+header));
    char *record_data = (char*) malloc(num_bytes);
    if(record_data == NULL) {
        throw std::runtime_error("Memory allocation error while writing mech stats");
    }

    strcpy(record_data, "");
    char record_entry[RECORD_LEN];

    /// prepare first row as header (first rank only)
    if(nrnmpi_myid == 0) {
        strcat(record_data, "gid,");
        for(int i=0; i<n_memb_func; i++) {
            const char*name = nrn_get_mechname(i);
            if(name == NULL)
                name = "";
            snprintf(record_entry, RECORD_LEN, "%s", name);
            strcat(record_data, record_entry);
            if((i+1) < n_memb_func) {
                strcat(record_data, ",");
            }
        }
        strcat(record_data, "\n");
    }

    /// mechanim count across all gids, local to rank
    std::vector<unsigned long long> total_mech_count(n_memb_func, 0);

    /// each gid record goes on separate row, only check non-empty threads
	for(int i=0; i<non_empty_thread_count; i++) {
		NrnThread& nt = nrn_threads[i];
		std::vector<int> mech_count(n_memb_func, 0);
		NrnThreadMembList* tml;
		for (tml = nt.tml; tml; tml = tml->next) {
			int type = tml->index;
			Memb_list* ml = tml->ml;
			mech_count[type] = ml->nodecount;
            total_mech_count[type] +=  ml->nodecount;
		}

        /// copy to buffer
		snprintf(record_entry, RECORD_LEN, "%d,", gidgroups[i]);
		strcat(record_data, record_entry);
		for(int i=0; i<n_memb_func; i++) {
			snprintf(record_entry, RECORD_LEN, "%d", mech_count[i]);
			strcat(record_data, record_entry);
            if((i+1) < n_memb_func) {
                strcat(record_data, ",");
            }
		}
		strcat(record_data, "\n");
	}

    /// print global stats to stdout
    print_mech_count(total_mech_count);

    // calculate offset into global file. note that we don't write
    // all num_bytes but only "populated" buffer
    unsigned long num_chars = strlen(record_data);
    unsigned long offset = 0;

    // global offset into file
    MPI_Exscan(&num_chars, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    // write to file using parallel mpi i/o
    MPI_File fh;
    MPI_Status status;

    // ibm mpi (bg-q) expects char* instead of const char* (even though it's standard)
    int op_status = MPI_File_open(MPI_COMM_WORLD, (char*) fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if(op_status != MPI_SUCCESS && nrnmpi_myid == 0) {
        throw std::runtime_error("Error while opening mech stat file " + fname);
    }

    op_status = MPI_File_write_at(fh, offset, record_data, num_chars, MPI_BYTE, &status);
    if(op_status != MPI_SUCCESS && nrnmpi_myid == 0) {
        throw std::runtime_error("Error while writing mech stats");
    }
    MPI_File_close(&fh);
}

#else

void output_spikes_parallel(const char* outpath) {
    std::cout << "Build and run with MPI to get mechanism stats \n";
}

#endif

}

