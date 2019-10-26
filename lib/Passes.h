#ifndef __PASSES_H__
#define __PASSES_H__

#include "llvm/Pass.h"

namespace cuprof {
  
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

  
  llvm::Pass* createMarkAllDeviceForInlinePass();
  llvm::Pass* createLinkDeviceSupportPass();
  llvm::Pass* createInstrumentDevicePass(InstrumentPassArg);

  llvm::Pass* createLinkHostSupportPass();
  llvm::Pass* createInstrumentHostPass(InstrumentPassArg);
}


#endif
