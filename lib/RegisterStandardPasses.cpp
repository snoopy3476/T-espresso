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

#include "llvm/Support/MemoryBuffer.h"

static bool CuProfTraceThread = false, CuProfTraceMem = false;

using namespace llvm;
namespace clang {
  
  static void registerStandardPasses(const PassManagerBuilder &, legacy::PassManagerBase &);
  

  class PluginEntry : public PluginASTAction {
    
  protected:


    // Register pass according to args
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef strref) override {

      static RegisterStandardPasses RegisterTracePass(
        PassManagerBuilder::EP_ModuleOptimizerEarly, registerStandardPasses);
      static RegisterStandardPasses RegisterTracePass0(
        PassManagerBuilder::EP_EnabledOnOptLevel0, registerStandardPasses);

      return make_unique<ASTConsumer>();
    
    }

    
    // Get args of the plugin
    bool ParseArgs(const CompilerInstance &CI,
                   const std::vector<std::string>& args) override {
      
      CuProfTraceThread = true;
      CuProfTraceMem = true;
      
      
      std::stringstream argslist;
      std::copy(args.begin(), args.end(), std::ostream_iterator<std::string>(argslist, ","));

      
      std::string optstr;
      while (getline(argslist, optstr, ',')) {
        size_t equal_pos = optstr.find('=');
        std::string optname = optstr.substr(0, equal_pos);
        std::string optarg;
        if (equal_pos == std::string::npos) {
          optarg = "";
        } else {
          optarg = optstr.substr(equal_pos);
        }
        
        //printf("%s, %s\n", optname.c_str(), optarg.c_str());
        
        if (optstr == "no-threadtrace") {
          errs() << "cuprof: Tracing thread scheduling - Disabled\n";
          CuProfTraceThread = false;
        } else if (optstr == "no-memtrace") {
          errs() << "cuprof: Tracing memory access - Disabled\n";
          CuProfTraceMem = false;
        } else if (optstr == "no-trace") {
          errs() << "cuprof: Tracing - Disabled\n";
          return false;
        }
        
      }

      return true;
    }

    
    // Run the plugin automatically
    ActionType getActionType() override {
      return ActionType::AddBeforeMainAction;
    }

  private:

  };




  
  static void registerStandardPasses(const PassManagerBuilder &, legacy::PassManagerBase &PM) {
    
    PM.add(createMarkAllDeviceForInlinePass());
    PM.add(createAlwaysInlinerLegacyPass());
    PM.add(createLinkDeviceSupportPass());
    PM.add(createInstrumentDevicePass(CuProfTraceThread, CuProfTraceMem));

    PM.add(createInstrumentHostPass());
  }
  
  static FrontendPluginRegistry::Add<clang::PluginEntry>
  X("cuprof", "cuda profiler");
}

