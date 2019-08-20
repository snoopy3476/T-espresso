#include "Passes.h"

#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/AST.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace llvm;

namespace clang {
  static bool TraceThread = true, TraceMem = true;

  
  static void registerStandardPasses(const PassManagerBuilder &, legacy::PassManagerBase &PM) {
    PM.add(createMarkAllDeviceForInlinePass());
    PM.add(createAlwaysInlinerLegacyPass());
    PM.add(createLinkDeviceSupportPass());
    PM.add(createInstrumentDevicePass(TraceThread, TraceMem));

    PM.add(createInstrumentHostPass());
  }

      

  class PluginEntry : public PluginASTAction {
  protected:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef strref) override {

      static RegisterStandardPasses RegisterTracePass(
        PassManagerBuilder::EP_ModuleOptimizerEarly, registerStandardPasses);
      static RegisterStandardPasses RegisterTracePass0(
        PassManagerBuilder::EP_EnabledOnOptLevel0, registerStandardPasses);

      return make_unique<ASTConsumer>();
    
    }

    

    bool ParseArgs(const CompilerInstance &CI,
                   const std::vector<std::string>& args) override {
      std::stringstream argslist;
      std::copy(args.begin(), args.end(), std::ostream_iterator<std::string>(argslist, ","));

      
      std::string optstr;
      while (getline(argslist, optstr, ',')) {
        if (optstr == "no-threadtrace") {
          errs() << "cuda-prof: Tracing thread scheduling - Disabled\n";
          TraceThread = false;
        } else if (optstr == "no-memtrace") {
          errs() << "cuda-prof: Tracing memory access - Disabled\n";
          TraceMem = false;
        } else if (optstr == "no-trace") {
          errs() << "cuda-prof: Tracing - Disabled\n";
          return false;
        }
        
      }

      return true;
    }
  
    // Automatically run the plugin after the main AST action
    PluginASTAction::ActionType getActionType() override {
      return PluginASTAction::ActionType::AddBeforeMainAction;
    }

  };

}

static clang::FrontendPluginRegistry::Add<clang::PluginEntry>
X("cuda-prof", "cuda profiler");
