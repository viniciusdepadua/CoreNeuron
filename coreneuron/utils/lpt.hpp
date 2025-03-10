/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================
*/

#ifndef lpt_h
#define lpt_h

#include <vector>

std::vector<std::size_t> lpt(std::size_t nbag,
                             std::vector<std::size_t>& pieces,
                             double* bal = nullptr);

double load_balance(std::vector<size_t>&);
#endif
