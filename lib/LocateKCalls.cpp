#include "LocateKCalls.h"

#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "cuprof-locate-kernel-launches"

#include "compat/LLVM-8.h" // for backward compatibility

#ifndef CUDA_LAUNCH_FUNC_NAME
#define CUDA_LAUNCH_FUNC_NAME "cudaLaunchKernel"
#endif

#ifndef CUDA_PUSHCONF_FUNC_NAME
#define CUDA_PUSHCONF_FUNC_NAME "__cudaPushCallConfiguration"
#endif

#ifndef CUDA_POPCONF_FUNC_NAME
#define CUDA_POPCONF_FUNC_NAME "__cudaPopCallConfiguration"
#endif


using namespace llvm;

SmallVector<Function*, 32> findKernels(Module& module) {
  SmallVector<Function*, 32> kernel_list;
  
  Function* func_callee[] = {
    module.getFunction(CUDA_POPCONF_FUNC_NAME),
    module.getFunction(CUDA_LAUNCH_FUNC_NAME)
  };

  const int FUNC_COUNT = sizeof(func_callee)/sizeof(*func_callee);
  
  std::vector<Function*> caller_list[FUNC_COUNT];

  // find all caller that func_callee exists
  for (int i = 0; i < FUNC_COUNT; i++) {
    if (func_callee[i] == nullptr) return {};
    
    for (auto* user : func_callee[i]->users()) {
      CallInst* callinst_cur = dyn_cast<CallInst>(user);
      Function* func_cur;
      if (callinst_cur && (func_cur = callinst_cur->getCaller())) {
        if (std::find(caller_list[i].begin(), caller_list[i].end(), func_cur)
            == caller_list[i].end())
          caller_list[i].push_back(func_cur);
      }
    }

    // sort for get common below
    std::sort(caller_list[i].begin(), caller_list[i].end());
  }


  
  // filter common elements
  for (int i = 1; i < FUNC_COUNT; i++) {
    std::vector<Function*>::iterator
      iter = caller_list[0].begin(),
      iter_new = caller_list[i].begin();
    
    while (iter != caller_list[0].end() &&
           iter_new != caller_list[i].end()) {
      if (*iter == *iter_new) {
        ++iter;
        ++iter_new;
      } else if (*iter > *iter_new) {
        ++iter_new;
      } else {
        iter = caller_list[0].erase(iter);
      }
    }
  }



  // insert all common to return value
  for (auto iter = caller_list[0].cbegin();
       iter != caller_list[0].cend();
       ++iter) {
    kernel_list.push_back(*iter);
  }

  
  return kernel_list;
}

SmallVector<CallInst*, 32> findConfigureCalls(Module& module) {
  Function* func = module.getFunction(CUDA_PUSHCONF_FUNC_NAME);
  if (func == nullptr) {
    return {};
  }

  SmallVector<CallInst*, 32> ret_val;
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

Function* getCalledKernel(Instruction* launch) {
  if (launch == nullptr) return nullptr;
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
      return nullptr;
    }
  }
  if (callee->hasName() && callee->getName().str().compare(CUDA_LAUNCH_FUNC_NAME) != 0) {
    return callee;
  } else {
    op1 = op1->stripPointerCasts();
    callee = dyn_cast<Function>(op1);
    if (callee != nullptr && callee->hasName()) {
      return callee;
    }
  }
  return nullptr;
}


namespace llvm {

  LocateKCallsPass::LocateKCallsPass() : ModulePass(ID) {}

  bool LocateKCallsPass::runOnModule(Module& module) {
    launch_list.clear();
    kernel_list.clear();
    

    for (Function* kernel : findKernels(module)) {
      kernel_list.push_back(kernel);
    }

    for (CallInst* configure : findConfigureCalls(module)) {
      Instruction* launch = findLaunchFor(configure);
      Function* kernel = getCalledKernel(launch);
      launch_list.push_back(KCall(configure, launch, kernel));
    }
    
    
    return false;
  }

  void LocateKCallsPass::releaseMemory() {
    launch_list.clear();
    kernel_list.clear();
  }

  SmallVector<KCall, 32> LocateKCallsPass::getLaunchList() const {
    return launch_list;
  }

  SmallVector<Function*, 32> LocateKCallsPass::getKernelList() const {
    return kernel_list;
  }

  char LocateKCallsPass::ID = 0;

  Pass* createLocateKCallsPass() {
    return new LocateKCallsPass();
  }

}

static RegisterPass<LocateKCallsPass>
  X("cuprof-locate-kcalls", "detect kernels and locate its launches", false, false);
