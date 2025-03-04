/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

#include "coreneuron/utils/ivocvect.hpp"

namespace coreneuron {
IvocVect* vector_new(int n) {
    return new IvocVect(n);
}
int vector_capacity(IvocVect* v) {
    return v->size();
}
double* vector_vec(IvocVect* v) {
    return v->data();
}

/*
 * Retro-compatibility implementations
 */
void* vector_new1(int n) {
    return (void*) (new IvocVect(n));
}

#pragma acc routine seq
int vector_capacity(void* v) {
    return ((IvocVect*) v)->size();
}

#pragma acc routine seq
double* vector_vec(void* v) {
    return ((IvocVect*) v)->data();
}

}  // namespace coreneuron
