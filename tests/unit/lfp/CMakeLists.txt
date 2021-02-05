
# =============================================================================
# Copyright (C) 2016-2020 Blue Brain Project
#
# See top-level LICENSE file for details.
# =============================================================================

include_directories(${CMAKE_SOURCE_DIR}/coreneuron ${Boost_INCLUDE_DIRS})
file(GLOB lfp_test_src "*.cpp")

add_executable(lfp_test_bin ${lfp_test_src})
target_link_libraries(lfp_test_bin ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${MPI_CXX_LIBRARIES})

if(CORENRN_ENABLE_GPU)
  target_link_libraries(lfp_test_bin ${link_cudacoreneuron} ${CUDA_LIBRARIES})
endif()

add_test(NAME lfp_test COMMAND ${TEST_EXEC_PREFIX}
                                     ${CMAKE_CURRENT_BINARY_DIR}/lfp_test_bin)
