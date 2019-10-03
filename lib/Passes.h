#pragma once

#include "llvm/Pass.h"

namespace llvm {
  
  typedef struct InstrumentPassArg {
    bool trace_thread, trace_mem;
    std::vector<std::string> kernel;
    std::vector<uint8_t> sm;
    std::vector<uint64_t> cta;
    std::vector<uint32_t> warp;
  } InstrumentPassArg;

  static InstrumentPassArg args_default = {
    true, true, {}, {}, {}
  };

  
  Pass* createMarkAllDeviceForInlinePass();
  Pass* createLinkDeviceSupportPass();
  Pass* createInstrumentDevicePass(InstrumentPassArg);

  Pass* createLinkHostSupportPass();
  Pass* createInstrumentHostPass(InstrumentPassArg);
}
