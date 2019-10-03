#ifndef __LOCATE_KCALLS_H__
#define __LOCATE_KCALLS_H__

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

struct KCall {
  KCall(CallInst* cc, Instruction* kl, StringRef kn)
    : configure_call(cc), kernel_launch(kl), kernel_name(kn)
  {}
  CallInst* configure_call;
  Instruction* kernel_launch;
  std::string kernel_name;
};

class LocateKCallsPass : public ModulePass {
public:
  static char ID;
  LocateKCallsPass();
  bool runOnModule(Module& module) override;
  void releaseMemory() override;
  SmallVector<KCall, 4> getLaunches() const;
private:
  SmallVector<KCall, 4> launches;
};

}

#endif
