#include "LocateKCalls.h"

#include "llvm/IR/Instructions.h"

#include <iostream> //////////////////

#define DEBUG_TYPE "memtrace-locate-kernel-launches"

using namespace llvm;

SmallVector<CallInst*, 4> findConfigureCalls(Module &M) {
  Function* F = M.getFunction("cudaConfigureCall");
  if (F == nullptr) {
    return {};
  }

  SmallVector<CallInst*, 4> R;
  for (auto *user : F->users()) {
    auto *CI = dyn_cast<CallInst>(user);
    if (CI != nullptr) {
      R.push_back(CI);
    }
  }
  return R;
}

Instruction* findLaunchFor(CallInst* configureCall) {
  auto* Terminator = configureCall->getParent()->getTerminator();
  auto* Br = dyn_cast<BranchInst>(Terminator);
  if (Br == nullptr) {
    errs() << "configureCall not followed by kcall.configok\n";
    return nullptr;
  }
  // follow to "kcall.configok" block
  BasicBlock *candidate = nullptr;
  for (auto *successor : Br->successors()) {
    if (successor->getName().startswith("kcall.configok")) {
      candidate = successor;
      break;
    }
  }
  if (candidate == nullptr) {
    errs() << "configureCall not followed by kcall.configok\n";
    return nullptr;
  }
  // find first block NOT followed by a "setup.next*" block
  while (true) {
    Terminator = candidate->getTerminator();
    Br = dyn_cast<BranchInst>(Terminator);
    if (Br == nullptr) break;
    BasicBlock *next = nullptr;
    for (auto *successor : Br->successors()) {
      if (successor->getName().startswith("setup.next")) {
        next = successor;
        break;
      }
    }
    if (next == nullptr) break;
    candidate = next;
  }

  Instruction* launch = nullptr;
  for (auto it = candidate->rbegin(); it != candidate->rend(); ++it) {
    if (isa<CallInst>(*it) || isa<InvokeInst>(*it)) {
      launch = &(*it);
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
  Value *op1 = nullptr;
  CallInst *CI = dyn_cast<CallInst>(launch);
  if (CI != nullptr) {
    callee = CI->getCalledFunction();
    if (CI->getNumArgOperands() > 0) {
      op1 = CI->getArgOperand(0);
    }
  } else {
    InvokeInst *II = dyn_cast<InvokeInst>(launch);
    if (II != nullptr) {
      callee = II->getCalledFunction();
      if (II->getNumArgOperands() > 0) {
        op1 = II->getArgOperand(0);
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

  


// Constant Definitions
    //ConstantPointerNull* const_ptr_2 = ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()));
    //ConstantPointerNull* const_ptr_2 = ConstantPointerNull::get(Constant::getNullValue(ArrayType::get(i8, 300)));

// Global Variable Definitions
    //gvar_ptr_abc->setInitializer(Constant::getNullValue(arty));
    
    //gvar_ptr_abc->setInitializer(ConstantAggregateZero::get(arty));

    //gvar_ptr_abc->setInitializer(init);
    
    


    /*
      std::string* utf8string = new std::string("asdf");
      Type *i8 = Type::getInt8Ty(M.getContext());
      std::vector<llvm::Constant *> chars(utf8string->size()+1);
      printf("%d\n", utf8string->size());
      const char* curstr = utf8string->data();
      printf("%s\n", curstr);
      for(unsigned int i = 0; i < utf8string->size()+1; i++) {
      chars[i] = ConstantInt::get(i8, curstr[i]);
      //printf("%d\n", chars[i]));
      }
      printf("%d\n", chars.size());
      Constant* init = ConstantArray::get(ArrayType::get(i8, chars.size()),
      chars);

      Constant* test = Constant::getNullValue(i8);
    */
    //ConstantDataArray cdatmp = new ConstantDataArray(Type::getInt8Ty(), curstr);
    //printf("cdatmp: %s\n", cdatmp.getRawDataValues().data());
    //printf("%s\n", init);

    //GlobalVariable * v = getOrCreateGlobalVar(M, test->getType(), "ASDF");

    /*
    GlobalVariable *tmp = M.getNamedGlobal("ASDF");
    if (tmp == nullptr || tmp->isNullValue())
      printf("NULL\n");
    */
    
    /*GlobalVariable *Global = M.getGlobalVariable("ASDF");
    
    if (Global->getInitializer()->isNullValue())
      printf("NULL\n");
    */
    //Constant *zero = Constant::getNullValue(T)
    /*
    GlobalVariable *Global = new GlobalVariable(M, init->getType(), true, GlobalValue::LinkOnceAnyLinkage, init, "ASDF");
    Global->setAlignment(8);
    //Global->setInitializer(init);
    assert(Global != nullptr);
    */
    //v->setInitializer(test);
    /*GlobalVariable * v = 
      new GlobalVariable(M, init->getType(), true,
      GlobalVariable::ExternalLinkage, 0,
      "ASDF");*/


  
    //Constant* tmp = M.getOrInsertGlobal("ASDF", init->getType());
    //GlobalVariable* gVar = M.getNamedGlobal("ASDF");
    //gVar->setAlignment(1);
    //v->setInitializer(init);
    //M.getOrInsertGlobal(v->getName(), v->getType());

    //M.getOrInsertGloba
    //return ConstantExpr::getBitCast(v, i8->getPointerTo());


  
  

  bool LocateKCallsPass::runOnModule(Module &M) {
    //LocateKCallsPass::testGlobal(M);
    launches.clear();
    for (auto *configure : findConfigureCalls(M)) {
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

  void LocateKCallsPass::print(raw_ostream &O, const Module *M) const {
    for (const auto &launch : launches) {
      O << "\n";
      O << "name:   " << launch.kernelName << "\n";
      O << "config: " << *launch.configureCall << "\n";
      if (launch.kernelLaunch != nullptr) {
        O << "launch: " << *launch.kernelLaunch << "\n";
      } else {
        O << "launch: (nullptr)\n";
      }
    }
  }

  char LocateKCallsPass::ID = 0;

  Pass *createLocateKCallsPass() {
    return new LocateKCallsPass();
  }

}

static RegisterPass<LocateKCallsPass>
  X("memtrace-locate-kcalls", "locate kernel launches", false, false);
