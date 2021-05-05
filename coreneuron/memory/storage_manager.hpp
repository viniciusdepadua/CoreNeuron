#pragma once
#include "coreneuron/memory/dual_vector.hpp"
#include "coreneuron/memory/storage_manager_interface.hpp"
namespace coreneuron {
  struct StorageManager : IStorageManager {
    void resize( std::size_t num_segments ) override {
      m_rhs.resize( num_segments );
    }
    double* rhs() override { return m_rhs.host_data(); }
    double* rhs_device() override { return m_rhs.device_data(); }
    double* rhs_sync_to_device() override {
      return m_rhs.sync_to_device();
    }
    double* rhs_sync_to_host() override {
      return m_rhs.sync_to_host();
    }
  private:
    dual_vector<double> m_rhs;    
  };
}
