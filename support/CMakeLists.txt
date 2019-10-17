#find_package(CUDAh)

###############################################################################
## HOST SUPPORT

set(CUPROF_CUDA_FLAGS "-O2" CACHE STRING "clang-cuda flags used to compile cuprof runtime")
message(STATUS "Using CUPROF_CUDA_FLAGS:" ${CUPROF_CUDA_FLAGS})


add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libcuprofhost.a"
  COMMAND bin/clang++
    -std=c++11 --cuda-host-only -c -O2
    ${CUPROF_CUDA_FLAGS}
    -o lib/libcuprofhost.o
    "${CMAKE_CURRENT_SOURCE_DIR}/host-support.cu"
    &&
    bin/llvm-ar
    rc lib/libcuprofhost.a
    lib/libcuprofhost.o
    &&
    rm lib/libcuprofhost.o
    
  
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS host-support.cu ../lib/Common.h clang llvm-ar
  VERBATIM
  )
add_custom_target(cuprofhost
  DEPENDS "${LLVM_BINARY_DIR}/lib/libcuprofhost.a"
  )
install(FILES ${LLVM_BINARY_DIR}/lib/libcuprofhost.a
  DESTINATION lib)

###############################################################################
## DEVICE SUPPORT

# compile to bitcode and headerize
add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/include/cuprofdevice.h"
  COMMAND bin/clang++
    --cuda-device-only
    -c -emit-llvm
    --cuda-gpu-arch=sm_50
    ${CUPROF_CUDA_FLAGS}
    -o lib/cuprofdevice.bc
    "${CMAKE_CURRENT_SOURCE_DIR}/device-support.cu"
    &&
    ${CMAKE_CURRENT_SOURCE_DIR}/headerize.sh
    "lib/cuprofdevice.bc" "include/cuprofdevice.h"
    &&
    rm -f lib/cuprofdevice.bc
    
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS device-support.cu ../lib/Common.h clang
  VERBATIM
  )
add_custom_target(cuprofdevice DEPENDS
  "${LLVM_BINARY_DIR}/include/cuprofdevice.h"
  )
#install(FILES ${LLVM_BINARY_DIR}/lib/cuprofdevice.bc
#  DESTINATION lib)
