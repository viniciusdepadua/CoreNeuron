#pragma once
// Eventually this can be replaced with std::span when C++20 is available everywhere.
#include <gsl-lite/gsl-lite.hpp>
namespace coreneuron {
// Note that this does not support static extents.
// See: https://github.com/gsl-lite/gsl-lite/issues/153.
template <typename T>
using span = gsl_lite::span<T>;
}  // namespace coreneuron
