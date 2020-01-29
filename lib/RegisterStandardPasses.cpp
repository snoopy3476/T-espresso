#include "Passes.h"

#include <sstream>

#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/AST.h"




using namespace llvm;

namespace cuprof {

  static InstrumentPassArg pass_args = {true, true};
  
  class CuprofPluginEntry : public clang::PluginASTAction {
    void anchor() override { }
  protected:


    // Register pass according to args
    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance&, StringRef) override {
      return make_unique<clang::ASTConsumer>();
    }

    
    // Get args of the plugin
    bool ParseArgs(const clang::CompilerInstance&,
                   const std::vector<std::string>& args) override {
      
      pass_args.trace_thread = true;
      pass_args.trace_mem = true;
      
      const char ARG_TYPE_DELIM = ':';
      const char ARG_TYPE_DELIM_STR[] = {ARG_TYPE_DELIM, 0};
      const char ARG_VAL_ASSIGN_DELIM = '=';
      const char ARG_VAL_DELIM = ',';
      const char ARG_VAL_DIM_DELIM = '/';
      
      std::stringstream argslist;
      std::copy(
        args.begin(), args.end(),
        std::ostream_iterator<std::string>(argslist, ARG_TYPE_DELIM_STR)
        );

      
      std::string optstr;
      while (getline(argslist, optstr, ARG_TYPE_DELIM)) {
        size_t equal_pos = optstr.find(ARG_VAL_ASSIGN_DELIM);
        std::string optname = optstr.substr(0, equal_pos);
        std::stringstream optarglist;
        if (equal_pos != std::string::npos) {
          optarglist = std::stringstream(optstr.substr(equal_pos+1));
        }
        
        
        
        if (optname == "thread-only") {
          //errs() << "cuprof: Tracing thread scheduling - Disabled\n";
          pass_args.trace_thread = true;
          pass_args.trace_mem = false;

          
        } else if (optname == "mem-only") {
          //errs() << "cuprof: Tracing memory access - Disabled\n";
          pass_args.trace_thread = false;
          pass_args.trace_mem = true;

          
        } else if (optname == "kernel") {
          std::string optarg;
          while (getline(optarglist, optarg, ARG_VAL_DELIM)) {
            
            pass_args.kernel.push_back(optarg);
          }

          
        } else if (optname == "grid") {
          std::string optarg;
          while (getline(optarglist, optarg, ARG_VAL_DELIM)) {
            
            if (std::all_of(optarg.begin(), optarg.end(), ::isdigit)) {
              uint64_t grid = (uint64_t)(std::stoi(optarg));
              pass_args.grid.push_back(grid);
            }
          }

          
        } else if (optname == "cta") {
          std::string optarg;
          while (getline(optarglist, optarg, ARG_VAL_DELIM)) {
            
            uint32_t ctaid[3] = {0, 0, 0};
            std::stringstream ctaidlist(optarg);
            std::string ctaid_cur;
            
            for (int i = 0; i < 3 && getline(ctaidlist, ctaid_cur, ARG_VAL_DIM_DELIM); i++) {
              if (std::all_of(ctaid_cur.begin(), ctaid_cur.end(), ::isdigit)) {
                ctaid[i] = stoi(ctaid_cur);
              }
            }

            uint64_t ctaid_conv =
              ( (uint64_t)ctaid[0] << 32 ) +
              ( (uint64_t)(ctaid[1] & 0xFFFF) << 16 ) +
              ( (uint64_t)ctaid[2] & 0xFFFF );
            pass_args.cta.push_back(ctaid_conv);
          }
          

        } else if (optname == "warpv") {
          std::string optarg;
          while (getline(optarglist, optarg, ARG_VAL_DELIM)) {
            
            if (std::all_of(optarg.begin(), optarg.end(), ::isdigit)) {
              uint32_t warpid_v = std::stoi(optarg);
              pass_args.warpv.push_back(warpid_v);
            }
          }

          
        } else if (optname == "sm") {
          std::string optarg;
          while (getline(optarglist, optarg, ARG_VAL_DELIM)) {
            
            if (std::all_of(optarg.begin(), optarg.end(), ::isdigit)) {
              uint32_t sm = (uint32_t)(std::stoi(optarg));
              pass_args.sm.push_back(sm);
            }
          }
          

        } else if (optname == "warpp") {
          std::string optarg;
          while (getline(optarglist, optarg, ARG_VAL_DELIM)) {
            
            if (std::all_of(optarg.begin(), optarg.end(), ::isdigit)) {
              uint32_t warpid_p = std::stoi(optarg);
              pass_args.warpp.push_back(warpid_p);
            }
          }
          
          
        } else {
          fprintf(stderr, "cuprof: unused argument: %s\n", optstr.c_str());
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
  
  static clang::FrontendPluginRegistry::Add<CuprofPluginEntry>
  X("cuprof", "cuda profiler");



  

  // Register passes to be executed automatically

  static void registerStandardPasses(const PassManagerBuilder&,
                                     legacy::PassManagerBase& pm) {
    
    pm.add(createMarkAllDeviceForInlinePass());
    pm.add(createAlwaysInlinerLegacyPass());
    pm.add(createLinkDeviceSupportPass());

    pm.add(createInstrumentHostPass(pass_args));
  }

  static RegisterStandardPasses pass_std_reg(
    PassManagerBuilder::EP_ModuleOptimizerEarly,
    registerStandardPasses);
  static RegisterStandardPasses pass_std_opt0(
    PassManagerBuilder::EP_EnabledOnOptLevel0,
    registerStandardPasses);
  

  // Register device pass to the optimizer last
  // to prevent hidering optimization
  static void registerDevicePass(const PassManagerBuilder&,
                                 legacy::PassManagerBase& pm) {
    
    pm.add(createInstrumentDevicePass(pass_args));
  }

  static RegisterStandardPasses pass_dev_reg(
    PassManagerBuilder::EP_OptimizerLast,
    registerDevicePass);
  static RegisterStandardPasses pass_dev_opt0(
    PassManagerBuilder::EP_EnabledOnOptLevel0,
    registerDevicePass);
  

  
}
  

