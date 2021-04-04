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
#include "coreneuron/utils/nrnoc_aux.hpp"

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

// helper functions defined below.
static void nrn2core_tqueue();

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
  // io/nrn_checkpoint.cpp: write_tqueue contains examples for each
    // DiscreteEvent type with regard to the information needed for each
    // subclass from the point of view of CoreNEURON.
    // E.g. for NetConType_, just netcon_index
    // The trick, then, is to figure out the CoreNEURON info from the
    // NEURON queue items and that should be available in passing from
    // the existing processing of nrncore_write.

  nrn2core_tqueue();
}

// For direct transfer of event queue information
// Must be the same as corresponding struct NrnCoreTransferEvents in NEURON
struct NrnCoreTransferEvents {
  std::vector<int> type; // DiscreteEvent type
  std::vector<double> td; // delivery time
  std::vector<int> intdata; // ints specific to the DiscreteEvent type
  std::vector<double> dbldata; // doubles specific to the type.
};


extern "C" {
/** Pointer to function in NEURON that iterates over its tqeueue **/
NrnCoreTransferEvents* (*nrn2core_transfer_tqueue_)(int tid);
}

/** Copy each thread's queue from NEURON **/
static void nrn2core_tqueue() {
  for (int tid=0; tid < nrn_nthread; ++tid) { // should be parallel
    NrnCoreTransferEvents* ncte = (*nrn2core_transfer_tqueue_)(tid);
    if (ncte) {
      size_t idat = 0;
      size_t idbldat = 0;
      NrnThread& nt = nrn_threads[tid];
      for (size_t i = 0; i < ncte->type.size(); ++i) {
        switch (ncte->type[i]) {
          case 2: { // NetCon
            int ncindex = ncte->intdata[idat++];
            printf("ncindex = %d\n", ncindex);
            NetCon* nc = nt.netcons + ncindex;
#define DEBUGQUEUE 0
#if DEBUGQUEUE
printf("nrn2core_tqueue tid=%d i=%zd type=%d tdeliver=%g NetCon %d\n",
tid, i, ncte->type[i], ncte->td[i], ncindex);
#endif
            nc->send(ncte->td[i], net_cvode_instance, &nt);
          } break;
          case 3: { // SelfEvent
            // target_type, target_instance, weight_index, flag movable

            // This is a nightmare and needs to be profoundly re-imagined.

            // Determine Point_process*
            int target_type = ncte->intdata[idat++];
            int target_instance = ncte->intdata[idat++];
            // From target_type and target_instance (mechanism data index)
            // compute the nt.pntprocs index.
            int offset = nt._pnt_offset[target_type];
            Point_process* pnt = nt.pntprocs + offset + target_instance;
            assert(pnt->_type == target_type);
            assert(pnt->_i_instance == target_instance);
            assert(pnt->_tid == tid);

            // Determine weight_index
            int netcon_index = ncte->intdata[idat++]; // via the NetCon
            int weight_index = nt.netcons[netcon_index].u.weight_index_;

            double flag = ncte->dbldata[idbldat++];
            int is_movable = ncte->intdata[idat++];
            // If the queue item is movable, then the pointer needs to be
            // stored in the mechanism instance movable slot by net_send.
            // And don't overwrite if not movable. Only one SelfEvent
            // for a given target instance is movable.
            int movable = 0;

#if DEBUGQUEUE
printf("nrn2core_tqueue tid=%d i=%zd type=%d tdeliver=%g SelfEvent\n",
tid, i, ncte->type[i], ncte->td[i]);
printf("  target_type=%d pnt data index=%d flag=%g is_movable=%d netcon index for weight=%d\n",
target_type, target_instance, flag, is_movable, netcon_index);
#endif
            net_send(nt._vdata + movable, weight_index, pnt, ncte->td[i], flag);
          } break;
          case 4: { // PreSyn
            int ps_index = ncte->intdata[idat++];
#if DEBUGQUEUE
printf("nrn2core_tqueue tid=%d i=%zd type=%d tdeliver=%g PreSyn %d\n",
tid, i, ncte->type[i], ncte->td[i], ps_index);
#endif
            PreSyn* ps = nt.presyns + ps_index;
            int gid = ps->output_index_;
            // Following assumes already sent to other machines.
            ps->output_index_ = -1;
            ps->send(ncte->td[i], net_cvode_instance, &nt);
            ps->output_index_ = gid;
          } break;
          case 7: { // NetParEvent
          } break;
          default: {
            static char s[20];
            sprintf(s, "%d", ncte->type[i]);
            hoc_execerror("Unimplemented transfer queue event type:", s);
          } break;
        }
      }
      delete ncte;
    }
  }
}

}  // namespace coreneuron
