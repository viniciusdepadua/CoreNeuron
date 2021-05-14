/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================
*/

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/io/nrn2core_direct.h"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/io/core2nrn_data_return.hpp"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/utils/vrecitem.h"

/** @brief, Information from NEURON to help with copying data to NEURON.
 *  Info for copying voltage, i_membrane_, and mechanism data.
 *  See implementaton in
 *  nrn/src/nrniv/nrnbbcore_write.cpp:nrnthreads_type_return.
 *  Return is size of either the returned data pointer or the number
 *  of pointers in mdata. tid is the thread index.
 */
size_t (*nrn2core_type_return_)(int type, int tid, double*& data, double**& mdata);

namespace coreneuron {

/** @brief permuted array copied to unpermuted array
 *  If permute is NULL then just a copy
 */
static void inverse_permute_copy(size_t n, double* permuted_src, double* dest, int* permute) {
    if (permute) {
        for (size_t i = 0; i < n; ++i) {
            dest[i] = permuted_src[permute[i]];
        }
    } else {
        std::copy(permuted_src, permuted_src + n, dest);
    }
}

/** @brief SoA permuted mechanism data copied to unpermuted AoS data.
 *  dest is an array of n pointers to the beginning of each sz length array.
 *  src is a contiguous array of sz segments of size stride. The stride
 *  may be slightly greater than n for purposes of alignment.
 *  Each of the sz segments of src are permuted.
 */
static void soa2aos_inverse_permute_copy(size_t n,
                                         int sz,
                                         int stride,
                                         double* src,
                                         double** dest,
                                         int* permute) {
    // src is soa and permuted. dest is n pointers to sz doubles (aos).
    for (size_t instance = 0; instance < n; ++instance) {
        double* d = dest[instance];
        double* s = src + permute[instance];
        for (int i = 0; i < sz; ++i) {
            d[i] = s[i * stride];
        }
    }
}

/** @brief SoA unpermuted mechanism data copied to unpermuted AoS data.
 *  dest is an array of n pointers to the beginning of each sz length array.
 *  src is a contiguous array of sz segments of size stride. The stride
 *  may be slightly greater than n for purposes of alignment.
 *  Each of the sz segments of src have the same order as the n pointers
 *  of dest.
 */
static void soa2aos_unpermuted_copy(size_t n, int sz, int stride, double* src, double** dest) {
    // src is soa and permuted. dest is n pointers to sz doubles (aos).
    for (size_t instance = 0; instance < n; ++instance) {
        double* d = dest[instance];
        double* s = src + instance;
        for (int i = 0; i < sz; ++i) {
            d[i] = s[i * stride];
        }
    }
}

/** @brief AoS mechanism data copied to AoS data.
 *  dest is an array of n pointers to the beginning of each sz length array.
 *  src is a contiguous array of n segments of size sz.
 */
static void aos2aos_copy(size_t n, int sz, double* src, double** dest) {
    for (size_t instance = 0; instance < n; ++instance) {
        double* d = dest[instance];
        double* s = src + (instance * sz);
        std::copy(s, s + sz, d);
    }
}

/** @brief Copy event queue and related state back to NEURON.
 */
static void core2nrn_tqueue(NrnThread&);

/** @brief copy data back to NEURON.
 *  Copies t, voltage, i_membrane_ if it used, and mechanism param data.
 *  Copies event queue and related state, e.g. WATCH, VecPlayContinuous.
 */
void core2nrn_data_return() {
    if (!nrn2core_type_return_) {
        return;
    }
    for (int tid = 0; tid < nrn_nthread; ++tid) {
        size_t n = 0;
        double* data = nullptr;
        double** mdata = nullptr;
        NrnThread& nt = nrn_threads[tid];

        n = (*nrn2core_type_return_)(0, tid, data, mdata);  // 0 means time
        if (n) {                                            // not the empty thread
            data[0] = nt._t;
        }

        if (nt.end) {  // transfer voltage and possibly i_membrane_
            n = (*nrn2core_type_return_)(voltage, tid, data, mdata);
            assert(n == size_t(nt.end) && data);
            inverse_permute_copy(n, nt._actual_v, data, nt._permute);

            if (nt.nrn_fast_imem) {
                n = (*nrn2core_type_return_)(i_membrane_, tid, data, mdata);
                assert(n == size_t(nt.end) && data);
                inverse_permute_copy(n, nt.nrn_fast_imem->nrn_sav_rhs, data, nt._permute);
            }
        }

        for (NrnThreadMembList* tml = nt.tml; tml; tml = tml->next) {
            int mtype = tml->index;
            Memb_list* ml = tml->ml;
            n = (*nrn2core_type_return_)(mtype, tid, data, mdata);
            assert(n == size_t(ml->nodecount) && mdata);
            if (n == 0) {
                continue;
            }
            // NEURON is AoS, CoreNEURON may be SoA and may be permuted.
            // On the NEURON side, the data is actually contiguous because of
            // cache_efficient, but that may not be the case for ARTIFICIAL_CELL.
            // For initial implementation simplicity, use the mdata info which gives
            // a double* for each param_size mech instance.
            int* permute = ml->_permute;
            double* cndat = ml->data;
            int layout = corenrn.get_mech_data_layout()[mtype];
            int sz = corenrn.get_prop_param_size()[mtype];
            if (layout == Layout::SoA) {
                int stride = ml->_nodecount_padded;
                if (permute) {
                    soa2aos_inverse_permute_copy(n, sz, stride, cndat, mdata, permute);
                } else {
                    soa2aos_unpermuted_copy(n, sz, stride, cndat, mdata);
                }
            } else { /* AoS */
                aos2aos_copy(n, sz, cndat, mdata);
            }
        }

        // Copy the event queue and related state.
        core2nrn_tqueue(nt);
    }
}

/** @brief Callbacks into NEURON for queue event types.
 */
extern "C" {
void (*core2nrn_NetCon_event_)(int tid, double td, size_t nc_index);

// must calculate netcon index from the weight index on this side
void (*core2nrn_SelfEvent_event_)(int tid,
                                  double td,
                                  int tar_type,
                                  int tar_index,
                                  double flag,
                                  size_t nc_index,
                                  int is_movable);
// the no weight case
void (*core2nrn_SelfEvent_event_noweight_)(int tid,
                                           double td,
                                           int tar_type,
                                           int tar_index,
                                           double flag,
                                           int is_movable);
}

static void core2nrn_tqueue_item(TQItem* q, NrnThread& nt) {
    DiscreteEvent* d = (DiscreteEvent*) q->data_;
    double td = q->t_;

    // potentially several SelfEvent TQItem* associated with same weight index.
    std::map<int, std::vector<TQItem*>> self_event_weight_map;

    switch (d->type()) {
        case NetConType: {
            NetCon* nc = (NetCon*) d;
            assert(nc >= nt.netcons && (nc < (nt.netcons + nt.n_netcon)));
            size_t nc_index = nc - nt.netcons;
            (*core2nrn_NetCon_event_)(nt.id, td, nc_index);
            break;
        }
        case SelfEventType: {
            SelfEvent* se = (SelfEvent*) d;
            Point_process* pnt = se->target_;
            assert(pnt->_tid == nt.id);
            int tar_type = (int) pnt->_type;
            int tar_index = pnt->_i_instance;
            double flag = se->flag_;
            TQItem** movable = (TQItem**) (se->movable_);
            int is_movable = (movable && *movable == q) ? 1 : 0;
            int weight_index = se->weight_index_;
            // the weight_index is useless on the NEURON side so we need
            // to convert that to NetCon index  and let the NEURON side
            // figure out the weight_index. To figure out the netcon_index
            // construct a {weight_index : [TQItem]} here for any
            // weight_index >= 0, otherwise send it NEURON now.
            if (weight_index >= 0) {
                self_event_weight_map[weight_index].push_back(q);
            } else {
                (*core2nrn_SelfEvent_event_noweight_)(
                    nt.id, td, tar_type, tar_index, flag, is_movable);
            }
            break;
        }
        case PreSynType: {
            PreSyn* ps = (PreSyn*) d;
            printf("PreSynType %g\n", td);
            break;
        }
        case NetParEventType: {
            // nothing to transfer
            break;
        }
        case PlayRecordEventType: {
            PlayRecord* pr = ((PlayRecordEvent*) d)->plr_;
            printf("PlayRecordEventType %g\n", td);
            break;
        }
        default: {
            // In particular, InputPreSyn does not appear in tqueue as it
            // immediately fans out to NetCon.
            assert(0);
            break;
        }
    }

    // For self events with weight, find the NetCon index and send that
    // to NEURON.
    if (!self_event_weight_map.empty()) {
        for (int nc_index = 0; nc_index < nt.n_netcon; ++nc_index) {
            NetCon& nc = nt.netcons[nc_index];
            int weight_index = nc.u.weight_index_;
            auto search = self_event_weight_map.find(weight_index);
            if (search != self_event_weight_map.end()) {
                auto& tqitems = search->second;
                for (auto q: tqitems) {
                    DiscreteEvent* d = (DiscreteEvent*) (q->data_);
                    double td = q->t_;
                    assert(d->type() == SelfEventType);
                    SelfEvent* se = (SelfEvent*) d;
                    int tar_type = se->target_->_type;
                    int tar_index = se->target_ - nt.pntprocs;
                    double flag = se->flag_;
                    TQItem** movable = (TQItem**) (se->movable_);
                    int is_movable = (movable && *movable == q) ? 1 : 0;
                    (*core2nrn_SelfEvent_event_)(
                        nt.id, td, tar_type, tar_index, flag, nc_index, is_movable);
                }
            }
        }
    }
}

void core2nrn_tqueue(NrnThread& nt) {
    // VecPlayContinuous

    // PatternStim

    // nrn_checkpoint.cpp has:
    // Avoid extra spikes due to some presyn voltages above threshold

    // The items on the queue
    NetCvodeThreadData& ntd = net_cvode_instance->p[nt.id];
    TQueue<QTYPE>* tqe = ntd.tqe_;
    TQItem* q;
    // TQItems from atomic_dq
    while ((q = tqe->atomic_dq(1e20)) != nullptr) {
        core2nrn_tqueue_item(q, nt);
    }
    // TQitems from binq_
    for (q = tqe->binq_->first(); q; q = tqe->binq_->next(q)) {
        core2nrn_tqueue_item(q, nt);
    }
}

}  // namespace coreneuron
