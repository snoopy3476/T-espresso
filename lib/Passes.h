#pragma once

#include "llvm/Pass.h"

namespace llvm {
  
  typedef struct InstrumentPassArg {
    bool trace_thread, trace_mem;
    std::vector<std::string> kernel;
    std::vector<uint32_t> sm, warp;
    std::vector<std::array<uint32_t, 3>> cta;
  } InstrumentPassArg;

  static InstrumentPassArg args_default = {
    true, true, {}, {}, {}
  };

  
  Pass *createMarkAllDeviceForInlinePass();
  Pass *createLinkDeviceSupportPass();
  Pass *createInstrumentDevicePass(InstrumentPassArg);

  Pass *createLinkHostSupportPass();
  Pass *createInstrumentHostPass(InstrumentPassArg);
}
