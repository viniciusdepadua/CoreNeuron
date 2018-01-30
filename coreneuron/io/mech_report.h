#ifndef NRN_MECH_REPORT_UTILS
#define NRN_MECH_REPORT_UTILS

#include <string>

namespace coreneuron {
    /// write mechanism counts to stdout and also per gid
    void write_mech_report(std::string, int, int*);
}

#endif
