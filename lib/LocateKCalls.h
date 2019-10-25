#ifndef __LOCATE_KCALLS_H__
#define __LOCATE_KCALLS_H__

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

namespace cuprof {

  struct KCall {
    KCall(llvm::CallInst* inst, llvm::Instruction* launch, llvm::Function* kernel)
      : configure_call(inst), kernel_launch(launch), kernel_obj(kernel)
      {}
    llvm::CallInst* configure_call;
    llvm::Instruction* kernel_launch;
    llvm::Function* kernel_obj;
  };

  class LocateKCallsPass : public llvm::ModulePass {
  public:
    static char ID;
    LocateKCallsPass();
    bool runOnModule(llvm::Module& module) override;
    void releaseMemory() override;
    llvm::SmallVector<KCall, 32> getLaunchList() const;
    llvm::SmallVector<llvm::Function*, 32> getKernelList() const;
  private:
    llvm::SmallVector<KCall, 32> launch_list;
    llvm::SmallVector<llvm::Function*, 32> kernel_list;
  };

}

#endif
