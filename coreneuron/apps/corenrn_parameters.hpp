/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/

#ifndef CN_PARAMETERS_H
#define CN_PARAMETERS_H

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <CLI/CLI.hpp>

/**
 * \class corenrn_parameters
 * \brief Parses and contains Command Line parameters for Core Neuron
 *
 * This structure contains all the parameters that CoreNeuron fetches
 * from the Command Line. It uses the CLI11 libraries to parse these parameters
 * and saves them in an internal public structure. Each parameter can be
 * accessed or written freely. By default the constructor instantiates a
 * CLI11 object and initializes it for CoreNeuron use.
 * This object is freely accessible from any point of the program.
 * An ostream method is also provided to print out all the parameters that
 * CLI11 parse.
 * Please keep in mind that, due to the nature of the subcommands in CLI11,
 * the command line parameters for subcategories NEED to be come before the relative
 * parameter. e.g. --mpi --gpu gpu --nwarp
 * Also single dash long options are not supported anymore (-mpi -> --mpi).
 */

namespace coreneuron {

struct corenrn_parameters {
    enum verbose_level : std::uint32_t { NONE = 0, ERROR = 1, INFO = 2, DEBUG = 3, DEFAULT = INFO };

    const int report_buff_size_default = 4;

    unsigned spikebuf = 100'000;           /// Internal buffer used on every rank for spikes
    int prcellgid = -1;                    /// Gid of cell for prcellstate
    unsigned ms_phases = 2;                /// Number of multisend phases, 1 or 2
    unsigned ms_subint = 2;                /// Number of multisend interval. 1 or 2
    unsigned spkcompress = 0;              /// Spike Compression
    unsigned cell_interleave_permute = 0;  /// Cell interleaving permutation
    unsigned nwarp = 0;     /// Number of warps to balance for cell_interleave_permute == 2
    unsigned num_gpus = 0;  /// Number of gpus to use per node
    unsigned report_buff_size = report_buff_size_default;  /// Size in MB of the report buffer.
    int seed = -1;  /// Initialization seed for random number generator (int)

    bool mpi_enable = false;         /// Enable MPI flag.
    bool skip_mpi_finalize = false;  /// Skip MPI finalization
    bool multisend = false;          /// Use Multisend spike exchange instead of Allgather.
    bool threading = false;          /// Enable pthread/openmp
    bool gpu = false;                /// Enable GPU computation.
    bool cuda_interface = false;     /// Enable CUDA interface (default is the OpenACC interface).
                                  /// Branch of the code is executed through CUDA kernels instead of
                                  /// OpenACC regions.
    bool binqueue = false;  /// Use bin queue.

    bool show_version = false;  /// Print version and exit.

    bool model_stats = false;  /// Print mechanism counts and model size after initialization

    verbose_level verbose{verbose_level::DEFAULT};  /// Verbosity-level

    double tstop = 100;        /// Stop time of simulation in msec
    double dt = -1000.0;       /// Timestep to use in msec
    double dt_io = 0.1;        /// I/O timestep to use in msec
    double dt_report;          /// I/O timestep to use in msec for reports
    double celsius = -1000.0;  /// Temperature in degC.
    double voltage = -65.0;    /// Initial voltage used for nrn_finitialize(1, v_init).
    double forwardskip = 0.;   /// Forward skip to TIME.
    double mindelay = 10.;     /// Maximum integration interval (likely reduced by minimum NetCon
                               /// delay).

    std::string patternstim;             /// Apply patternstim using the specified spike file.
    std::string datpath = ".";           /// Directory path where .dat files
    std::string outpath = ".";           /// Directory where spikes will be written
    std::string filesdat = "files.dat";  /// Name of file containing list of gids dat files read in
    std::string restorepath;             /// Restore simulation from provided checkpoint directory.
    std::string reportfilepath;          /// Reports configuration file.
    std::string checkpointpath;  /// Enable checkpoint and specify directory to store related files.
    std::string writeParametersFilepath;  /// Write parameters to this file
    std::string mpi_lib;                  /// Name of CoreNEURON MPI library to load dynamically.

    CLI::App app{"CoreNeuron - Optimised Simulator Engine for NEURON."};  /// CLI app that performs
                                                                          /// CLI parsing

    corenrn_parameters();  /// Constructor that initializes the CLI11 app.

    void parse(int argc, char* argv[]);  /// Runs the CLI11_PARSE macro.

    inline bool is_quiet() {
        return verbose == verbose_level::NONE;
    }
};

std::ostream& operator<<(std::ostream& os,
                         const corenrn_parameters& corenrn_param);  /// Printing method.

extern corenrn_parameters corenrn_param;  /// Declaring global corenrn_parameters object for this
                                          /// instance of CoreNeuron.
extern int nrn_nobanner_;                 /// Global no banner setting

}  // namespace coreneuron

#endif  // CN_PARAMETERS_H
