#pragma once
#include "coreneuron/gpu/nrn_acc_manager.hpp"

#include <umpire/ResourceManager.hpp>
#include <umpire/TypedAllocator.hpp>

#if defined(_OPENACC)
#include <openacc.h>
#endif

#include <vector>
#include <type_traits>

namespace coreneuron {

// Obviously temporary...but should work OK for now for host-only execution.
// template <typename T>
// using dual_vector = std::vector<T>;

enum struct MemorySpace { Host, Device };

template <typename T>
struct dual_vector {
  // Assume for now that copying bytes between host and device is enough.
  static_assert( std::is_trivially_copyable<T>::value );

  dual_vector() {
    std::cout << "dual_vector::dual_vector(): " << m_device_alloc << std::endl;
  }
  // These could all be supported, but delete them for now to be sure
  // dual_vector is only being used in expected ways.
  dual_vector(dual_vector&&) = delete;
  dual_vector(dual_vector const&) = delete;
  dual_vector& operator=(dual_vector&&) = delete;
  dual_vector& operator=(dual_vector const&) = delete;

  void resize( std::size_t size ) {
    if ( size == m_host.size() ) {
      std::cout << "resize( " << size << " ) doing nothing because the host vector was already that big" << std::endl;
      return;
    }
    if ( m_device_ptr ) {
      throw std::runtime_error( "Attempting to resize host-side data while device-side copy is active. This is not supported." );
    }
    m_host.resize( size );
  }

  T* sync_to_device() {
    if ( m_host.empty() ) {
      std::cout << "Not syncing empty vector to device" << std::endl;
      return nullptr;
    }
    std::cout << "m_device_alloc = " << m_device_alloc << ", m_host.size() = " << m_host.size() << ", m_device_size = " << m_device_size << std::endl;
    if ( m_device_size != m_host.size() ) {
      free_device();
      m_device_size = m_host.size();
      std::cout << "Allocating " << m_device_size << " entries on the device" << std::endl;
      m_device_ptr = allocator_type{m_device_alloc}.allocate( m_device_size );
      // Tell the OpenACC runtime about the mapping between the host and device
      // pointers. This should mean that OpenACC offloading continues to work as
      // before.
      std::cout << "Mapping hptr=" << m_host.data() << " to dptr=" << m_device_ptr << " (" << m_device_size * sizeof( T ) << " bytes)" << std::endl;
      acc_map_data( m_host.data(), m_device_ptr, m_device_size * sizeof( T ) );
    }
    // Note we are going via a singleton here even though we actually know the
    // host/device allocators here.
    std::cout << "Syncing hptr=" << m_host.data() << " to dptr=" << m_device_ptr << std::endl;
    umpire::ResourceManager::getInstance().copy( m_device_ptr, m_host.data(), m_device_size * sizeof(T) );
    return m_device_ptr;
  }

  T* sync_to_host() {
    if ( m_device_size != m_host.size() ) {
      std::cout << "(m_device_size = " << m_device_size << ") != (m_host.size() = " << m_host.size() << ')' << std::endl;
      std::exit( 1 );
    }
    std::cout << "Syncing dptr=" << m_device_ptr << " to hptr=" << m_host.data() << std::endl;
    umpire::ResourceManager::getInstance().copy( m_host.data(), m_device_ptr, m_device_size * sizeof( T ) );
    return m_host.data();
  }

  T* host_data() { return m_host.data(); }
  T* device_data() { return m_device_ptr; }

  ~dual_vector() {
    std::cout << "dual_vector::~dual_vector(): " << m_device_alloc << std::endl;
    free_device();
  }

private:
  void free_device() {
    if ( m_device_ptr ) {
      // Remove the mapping from the OpenACC runtime. FIXME there should be more
      // protections against the host-side vector being reallocated or resized
      // while a device-side copy is active.
      std::cout << "Unmapping hptr=" << m_host.data() << ", deallocating dptr=" << m_device_ptr << std::endl;
      acc_unmap_data( m_host.data() );
      allocator_type{m_device_alloc}.deallocate( std::exchange( m_device_ptr, nullptr ), m_device_size );
    }
  }

  using allocator_type = umpire::TypedAllocator<T>;
  std::vector<T, allocator_type> m_host{allocator_type{umpire::ResourceManager::getInstance().getAllocator("HOST")}};
  T *m_device_ptr{};
  std::size_t m_device_size{};
  // This isn't very elegant...NEURON already tells OpenACC which device to use
  // (based on the MPI rank and number of compute devices), here as a quick and
  // dirty solution we tell Umpire to use the current OpenACC device and trust
  // that NEURON already set it.
  umpire::Allocator m_device_alloc{umpire::ResourceManager::getInstance().getAllocator(std::string{"DEVICE::"} + std::to_string(acc_get_device_num(acc_device_nvidia)))};
};

}
