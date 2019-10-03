set(LLVM_NO_RTTI 1)

add_llvm_library(libcuprof MODULE
  MarkAllDeviceForInline.cpp
  LinkDeviceSupport.cpp
  InstrumentDevice.cpp
  LocateKCalls.cpp

  InstrumentHost.cpp

  RegisterStandardPasses.cpp
  )
add_dependencies(libcuprof
  cuprofdevice
  cuprofhost
  )

if (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(libcuprof PUBLIC LLVM)
else ()
  target_link_libraries(libcuprof PUBLIC
    LLVMSupport
    LLVMCore
    LLVMScalarOpts
    LLVMInstCombine
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMMC
    LLVMPasses
    LLVMLinker
    LLVMIRReader
    # The libraries below are required for darwin: http://PR26392
    LLVMBitReader
    LLVMMCParser
    LLVMObject
    LLVMProfileData
    LLVMTarget
    LLVMVectorize
    clangTooling
    )
endif()

install(FILES ${CUPROF_SOURCE_DIR}/lib/TraceIO.h
  DESTINATION include)
