#ifndef __LOCATE_KCALLS_H__
#define __LOCATE_KCALLS_H__

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

struct KCall {
  KCall(CallInst* inst, Instruction* launch, Function* kernel)
    : configure_call(inst), kernel_launch(launch), kernel_obj(kernel)
  {}
  CallInst* configure_call;
  Instruction* kernel_launch;
  Function* kernel_obj;
};

class LocateKCallsPass : public ModulePass {
public:
  static char ID;
  LocateKCallsPass();
  bool runOnModule(Module& module) override;
  void releaseMemory() override;
  SmallVector<KCall, 32> getLaunchList() const;
  SmallVector<Function*, 32> getKernelList() const;
private:
  SmallVector<KCall, 32> launch_list;
  SmallVector<Function*, 32> kernel_list;
};

}

#endif
