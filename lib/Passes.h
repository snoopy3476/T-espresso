#pragma once

#include "llvm/Pass.h"

namespace llvm {
  
  typedef struct InstrumentPassArg {
    bool trace_thread, trace_mem, verbose;
    std::vector<std::string> kernel;
    std::vector<uint64_t> grid;
    std::vector<uint64_t> cta;
    std::vector<uint32_t> warpv;
    std::vector<uint32_t> sm;
    std::vector<uint32_t> warpp;
  } InstrumentPassArg;

  static InstrumentPassArg args_default = {
    true, true, false, {}, {}, {}, {}, {}
  };

  
  Pass* createMarkAllDeviceForInlinePass();
  Pass* createLinkDeviceSupportPass();
  Pass* createInstrumentDevicePass(InstrumentPassArg);

  Pass* createLinkHostSupportPass();
  Pass* createInstrumentHostPass(InstrumentPassArg);
}
