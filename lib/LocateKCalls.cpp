#include "LocateKCalls.h"

#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "cuprof-locate-kernel-launches"

using namespace llvm;

SmallVector<CallInst*, 4> findConfigureCalls(Module& module) {
  Function* func = module.getFunction("cudaConfigureCall");
  if (func == nullptr) {
    return {};
  }

  SmallVector<CallInst*, 4> ret_val;
  for (auto* user : func->users()) {
    auto* callinst_cur = dyn_cast<CallInst>(user);
    if (callinst_cur != nullptr) {
      ret_val.push_back(callinst_cur);
    }
  }
  return ret_val;
}

Instruction* findLaunchFor(CallInst* configure_call) {
  auto* terminator = configure_call->getParent()->getTerminator();
  auto* branch_inst = dyn_cast<BranchInst>(terminator);
  if (branch_inst == nullptr) {
    errs() << "configure_call not followed by kcall.configok\n";
    return nullptr;
  }
  // follow to "kcall.configok" block
  BasicBlock* candidate = nullptr;
  for (auto* successor : branch_inst->successors()) {
    if (successor->getName().startswith("kcall.configok")) {
      candidate = successor;
      break;
    }
  }
  if (candidate == nullptr) {
    errs() << "configure_call not followed by kcall.configok\n";
    return nullptr;
  }
  // find first block NOT followed by a "setup.next*" block
  while (true) {
    terminator = candidate->getTerminator();
    branch_inst = dyn_cast<BranchInst>(terminator);
    if (branch_inst == nullptr) break;
    BasicBlock* next = nullptr;
    for (auto* successor : branch_inst->successors()) {
      if (successor->getName().startswith("setup.next")) {
        next = successor;
        break;
      }
    }
    if (next == nullptr) break;
    candidate = next;
  }

  Instruction* launch = nullptr;
  for (auto inst_cur = candidate->rbegin(); inst_cur != candidate->rend(); ++inst_cur) {
    if (isa<CallInst>(*inst_cur) || isa<InvokeInst>(*inst_cur)) {
      launch = &(*inst_cur);
      break;
    }
  }
  if (launch == nullptr) {
    errs() << "no launch found for configure call\n";
  }

  return launch;
}

std::string getKernelNameOf(Instruction* launch) {
  Function* callee = nullptr;
  Value* op1 = nullptr;
  CallInst* callinst = dyn_cast<CallInst>(launch);
  if (callinst != nullptr) {
    callee = callinst->getCalledFunction();
    if (callinst->getNumArgOperands() > 0) {
      op1 = callinst->getArgOperand(0);
    }
  } else {
    InvokeInst* invokeinst = dyn_cast<InvokeInst>(launch);
    if (invokeinst != nullptr) {
      callee = invokeinst->getCalledFunction();
      if (invokeinst->getNumArgOperands() > 0) {
        op1 = invokeinst->getArgOperand(0);
      }
    } else {
      return "";
    }
  }
  if (callee->hasName() && callee->getName() != "cudaLaunch") {
    return callee->getName();
  } else {
    op1 = op1->stripPointerCasts();
    callee = dyn_cast<Function>(op1);
    if (callee != nullptr && callee->hasName()) {
      return callee->getName();
    }
  }
  return "";
}


namespace llvm {

  LocateKCallsPass::LocateKCallsPass() : ModulePass(ID) {}

  bool LocateKCallsPass::runOnModule(Module& module) {
    launches.clear();
    for (auto* configure : findConfigureCalls(module)) {
      Instruction* launch = findLaunchFor(configure);
      std::string name = getKernelNameOf(launch);
      launches.push_back(KCall(configure, launch, name));
    }
    return false;
  }

  void LocateKCallsPass::releaseMemory() {
    launches.clear();
  }

  SmallVector<KCall, 4> LocateKCallsPass::getLaunches() const {
    return launches;
  }

  char LocateKCallsPass::ID = 0;

  Pass* createLocateKCallsPass() {
    return new LocateKCallsPass();
  }

}

static RegisterPass<LocateKCallsPass>
  X("cuprof-locate-kcalls", "locate kernel launches", false, false);
