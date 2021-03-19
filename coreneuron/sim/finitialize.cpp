/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

#include "coreneuron/nrnconf.h"
#include "coreneuron/network/netpar.hpp"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/sim/fast_imem.hpp"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/utils/profile/profiler_interface.h"
#include "coreneuron/coreneuron.hpp"

namespace coreneuron {

void nrn_finitialize(int setv, double v) {
    Instrumentor::phase_begin("finitialize");
    t = 0.;
    dt2thread(-1.);
    nrn_thread_table_check();
    clear_event_queue();
    nrn_spike_exchange_init();
#if VECTORIZE
    nrn_play_init(); /* Vector.play */
                     /// Play events should be executed before initializing events
    for (int i = 0; i < nrn_nthread; ++i) {
        nrn_deliver_events(nrn_threads + i); /* The play events at t=0 */
    }
    if (setv) {
        for (auto _nt = nrn_threads; _nt < nrn_threads + nrn_nthread; ++_nt) {
            double* vec_v = &(VEC_V(0));
            // clang-format off

            #pragma acc parallel loop present(      \
                _nt[0:1], vec_v[0:_nt->end])        \
                if (_nt->compute_gpu)
            // clang-format on
            for (int i = 0; i < _nt->end; ++i) {
                vec_v[i] = v;
            }
        }
    }

    if (nrn_have_gaps) {
        nrnmpi_v_transfer();
        for (int i = 0; i < nrn_nthread; ++i) {
            nrnthread_v_transfer(nrn_threads + i);
        }
    }

    for (int i = 0; i < nrn_nthread; ++i) {
        nrn_ba(nrn_threads + i, BEFORE_INITIAL);
    }
    /* the INITIAL blocks are ordered so that mechanisms that write
       concentrations are after ions and before mechanisms that read
       concentrations.
    */
    /* the memblist list in NrnThread is already so ordered */
    for (int i = 0; i < nrn_nthread; ++i) {
        NrnThread* nt = nrn_threads + i;
        for (auto tml = nt->tml; tml; tml = tml->next) {
            mod_f_t s = corenrn.get_memb_func(tml->index).initialize;
            if (s) {
                (*s)(nt, tml->ml, tml->index);
            }
        }
    }
#endif

    init_net_events();
    for (int i = 0; i < nrn_nthread; ++i) {
        nrn_ba(nrn_threads + i, AFTER_INITIAL);
    }
    for (int i = 0; i < nrn_nthread; ++i) {
        nrn_deliver_events(nrn_threads + i); /* The INITIAL sent events at t=0 */
    }
    for (int i = 0; i < nrn_nthread; ++i) {
        setup_tree_matrix_minimal(nrn_threads + i);
        if (nrn_use_fast_imem) {
            nrn_calc_fast_imem(nrn_threads + i);
        }
    }
    for (int i = 0; i < nrn_nthread; ++i) {
        nrn_ba(nrn_threads + i, BEFORE_STEP);
    }
    for (int i = 0; i < nrn_nthread; ++i) {
        nrn_deliver_events(nrn_threads + i); /* The record events at t=0 */
    }
#if NRNMPI
    nrn_spike_exchange(nrn_threads);
#endif
    nrncore2nrn_send_init();
    for (int i = 0; i < nrn_nthread; ++i) {
        nrncore2nrn_send_values(nrn_threads + i);
    }
    Instrumentor::phase_end("finitialize");
}

/**
  All state from NEURON necessary to continue a run.

  In NEURON direct mode, we desire the exact behavior of
  ParallelContext.psolve(tstop). I.e. a sequence of such calls with and
  without intervening calls to h.finitialize(). Most state (structure
  and data of the substantive model) has been copied
  from NEURON during nrn_setup. Now we need to copy the event queue
  and set up any other invalid internal structures. I.e basically the
  nrn_finitialize above but without changing any simulation data. We follow
  some of the strategy of checkpoint_initialize.
**/
void direct_mode_initialize() {
  dt2thread(-1.);
  nrn_thread_table_check();
  nrn_spike_exchange_init();

  // in case some nrn_init allocate data we need to do that but do not
  // want to call initmodel.
  _nrn_skip_initmodel = true;
  for (int i = 0; i < nrn_nthread; ++i) {  // should be parallel
    NrnThread& nt = nrn_threads[i];
    for (NrnThreadMembList* tml = nt.tml; tml; tml = tml->next) {
      Memb_list* ml = tml->ml;
      mod_f_t s = corenrn.get_memb_func(tml->index).initialize;
      if (s) {
        (*s)(&nt, ml, tml->index);
      }
    }
  }
  _nrn_skip_initmodel = false;

  // the things done by checkpoint restore at the end of Phase2::read_file
    // vec_play_continuous n_vec_play_continuous of them
    // patstim_index
    // preSynConditionEventFlags nt.n_presyn of them
    // restore_events
    // restore_events
  // the things done for checkpoint at the end of Phase2::populate
    // checkpoint_restore_tqueue
  // Lastly, if PatternStim exists, needs initialization
    // checkpoint_restore_patternstim
}

}  // namespace coreneuron
