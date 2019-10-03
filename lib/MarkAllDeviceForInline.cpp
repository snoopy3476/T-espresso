#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cuprof-mark-device-for-inline"

using namespace llvm;

struct MarkAllDeviceForInlinePass : public ModulePass {
  static char ID;
  MarkAllDeviceForInlinePass() : ModulePass(ID) {}

  bool runOnModule(Module& module) override {
    bool is_cuda = module.getTargetTriple().find("nvptx") != std::string::npos;
    if (!is_cuda) return false;

    for (Function& func : module) {
      if (func.isIntrinsic()) continue;
      func.removeFnAttr(Attribute::AttrKind::OptimizeNone);
      func.removeFnAttr(Attribute::AttrKind::NoInline);
      func.addFnAttr(Attribute::AttrKind::AlwaysInline);
    }

    return true;
  }
};
char MarkAllDeviceForInlinePass::ID = 0;

namespace llvm {
  Pass* createMarkAllDeviceForInlinePass() {
    return new MarkAllDeviceForInlinePass();
  }
}

static RegisterPass<MarkAllDeviceForInlinePass> X("cuprof-mark-inline", "marks all functions for inlining", false, false);
