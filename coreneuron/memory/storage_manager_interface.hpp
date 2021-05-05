#pragma once
namespace coreneuron {
/**
 * @brief Interface to managed memory storage.
 * 
 * This exists to allow NrnThread to own storage for lifetime management
 * purposes without exposing the allocation details of that storage to consumers
 * of NrnThread.
 */
struct IStorageManager {
    virtual void resize( std::size_t num_segments ) = 0;
    virtual double* rhs() = 0;
    virtual double* rhs_device() = 0;
    virtual double* rhs_sync_to_device() = 0;
    virtual double* rhs_sync_to_host() = 0;
    virtual ~IStorageManager() {}
};

}
