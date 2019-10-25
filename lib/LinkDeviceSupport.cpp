#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Linker/Linker.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "cuprofdevice.h"

#define DEBUG_TYPE "cuprof-link-device-support"

using namespace llvm;

namespace cuprof {

  struct LinkDeviceSupportPass : public ModulePass {
    static char ID;
    LinkDeviceSupportPass() : ModulePass(ID) {}

    bool runOnModule(Module& module) override {
      bool is_cuda = module.getTargetTriple().find("nvptx") != std::string::npos;
      if (!is_cuda) return false;

      SMDiagnostic err;
      LLVMContext& ctx = module.getContext();

      StringRef source = StringRef((const char*)device_utils, sizeof(device_utils));
      auto buf = MemoryBuffer::getMemBuffer(source, "source", false);

      auto util_module = parseIR(buf->getMemBufferRef(), err, ctx);
      if (util_module.get() == nullptr) {
        errs() << "error: " << err.getMessage() << "\n";
        report_fatal_error("unable to parse");
      }
      Linker::linkModules(module, std::move(util_module));

      return true;
    }
  };
  
  char LinkDeviceSupportPass::ID = 0;



  
  Pass* createLinkDeviceSupportPass() {
    return new LinkDeviceSupportPass();
  }
  
  static RegisterPass<LinkDeviceSupportPass>
  X("cuprof-link-device-support",
    "links device support functions into module",
    false, false);

}
