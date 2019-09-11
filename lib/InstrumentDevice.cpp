#include "Passes.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Constants.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <set>
#include <iostream>
#include <fstream>

#define INCLUDE_LLVM_MEMTRACE_STUFF
#include "cutrace_io.h"

#define DEBUG_TYPE "memtrace-device"
#define TRACE_DEBUG_DATA "___cuprof_accdat_instmd"

#define ADDRESS_SPACE_GENERIC 0
#define ADDRESS_SPACE_GLOBAL 1
#define ADDRESS_SPACE_INTERNAL 2
#define ADDRESS_SPACE_SHARED 3
#define ADDRESS_SPACE_CONSTANT 4
#define ADDRESS_SPACE_LOCAL 5

#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;


/******************************************************************************
 * Various helper functions
 */

// Prototype
// __device__ void __mem_trace (uint8_t* records, uint8_t* allocs,
//  uint8_t* commits, uint64_t desc, uint64_t addr, uint32_t slot) {
Constant *getOrInsertTraceDecl(Module &M) {
  LLVMContext &ctx = M.getContext();

  Type *voidTy = Type::getVoidTy(ctx);
  Type *i8PtrTy = Type::getInt8PtrTy(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *i32Ty = Type::getInt32Ty(ctx);

  return M.getOrInsertFunction("___cuprof_trace", voidTy,
			       i8PtrTy, i8PtrTy, i8PtrTy,
                               i64Ty, i64Ty, i64Ty, i32Ty, i32Ty);
}

Constant *getOrInsertSetDebugDataDecl(Module &M) {
  LLVMContext &ctx = M.getContext();

  Type *voidTy = Type::getVoidTy(ctx);
  Type *i8PtrTy = Type::getInt8PtrTy(ctx);
  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);

  return M.getOrInsertFunction("___cuprof_set_accdat", voidTy,
			       i8PtrTy, i64Ty);
}

std::vector<Function*> getKernelFunctions(Module &M) {
  std::set<Function*> Kernels;
  NamedMDNode * kernel_md = M.getNamedMetadata("nvvm.annotations");
  if (kernel_md) {
    // MDNodes in NamedMDNode
    for (const MDNode *node : kernel_md->operands()) {
      // MDOperands in MDNode
      for (const MDOperand &op : node->operands()) {
        Metadata * md = op.get();
        ValueAsMetadata *v = dyn_cast_or_null<ValueAsMetadata>(md);
        if (!v) continue;
        Function *f = dyn_cast<Function>(v->getValue());
        if (!f) continue;
        Kernels.insert(f);
      }
    }
  }
  return std::vector<Function*>(Kernels.begin(), Kernels.end());
}

GlobalVariable* defineDeviceGlobal(Module &M, Type* T, const Twine &name) {
  Constant *zero = Constant::getNullValue(T);
  auto *globalVar = new GlobalVariable(M, T, false,
                                       GlobalValue::ExternalLinkage, zero, name, nullptr,
                                       GlobalVariable::NotThreadLocal, 1, true);
  globalVar->setAlignment(1);
  globalVar->setDSOLocal(true);
  return globalVar;
}

/******************************************************************************
 * A poor man's infer address spaces, but instead of address spaces, we try
 * to infer visibility and it is implemented as a value analysis.
 */

enum PointerKind {
  PK_OTHER = 0,
  PK_GLOBAL,
  PK_UNINITIALIZED,
};

PointerKind mergePointerKinds(PointerKind pk1, PointerKind pk2) {
  return pk1 < pk2 ? pk1 : pk2;
}

PointerKind getPointerKind(Value* val, bool isKernel) {
  SmallPtrSet<Value*, 16> seen;
  SmallVector<Value*, 8> stack;
  PointerKind kind = PK_UNINITIALIZED;

  stack.push_back(val);
  while (!stack.empty()) {
    Value* node = stack.pop_back_val();
    if (seen.count(node) > 0)
      continue;
    seen.insert(node);

    //skip casts
    while (auto *cast = dyn_cast<BitCastOperator>(node)) {
      node = cast->getOperand(0);
    }
    if (isa<AllocaInst>(node)) {
      kind = mergePointerKinds(kind, PK_OTHER);
    } else if (isa<GlobalValue>(node)) {
      kind = mergePointerKinds(kind, PK_GLOBAL);
    } else if (isa<Argument>(node)) {
      kind = mergePointerKinds(kind, isKernel ? PK_GLOBAL : PK_OTHER);
    } else if (auto *gep = dyn_cast<GEPOperator>(node)) {
      stack.push_back(gep->getPointerOperand());
    } else if (auto *gep = dyn_cast<GetElementPtrInst>(node)) {
      stack.push_back(gep->getPointerOperand());
    } else if (auto *atomic = dyn_cast<AtomicRMWInst>(node)) {
      stack.push_back(atomic->getPointerOperand());
    } else if (isa<CallInst>(node)) {
      report_fatal_error("Base Pointer is result of function. No.");
    } else if (auto *phi = dyn_cast<PHINode>(node)) {
      int numIncoming = phi->getNumIncomingValues();
      for (int i = 0; i < numIncoming; ++i) {
        stack.push_back(phi->getIncomingValue(i));
      }
    }
  }

  return kind;
}



/******************************************************************************
 * Device instrumentation pass.
 * It performs 3 fundamental steps for each kernel:
 *
 * 1. collect globally visible memory accesses in this kernel
 * 2. setup data structures used by tracing infrastructure
 * 3. instrument globally visible memory accesses with traces
 *
 * This pass does not analyze across function boundaries and therefore requires
 * any device functions to be inlined.
 */

// Needs to be a ModulePass because we modify the global variables.
struct InstrumentDevicePass : public ModulePass {
  static char ID;
  InstrumentPassArg args;

  InstrumentDevicePass(InstrumentPassArg passargs = args_default) : ModulePass(ID), args(passargs) { }

  struct TraceInfoValues {
    Value *Allocs;
    Value *Commits;
    Value *Records;
    Value *Desc;
    Value *Slot;
  };


  static std::map<std::string, std::string> KernelCallHeader;
  static uint32_t RecordID;

  

  std::vector<Instruction*> collectGlobalMemAccesses(Function* kernel) {
    std::vector<Instruction*> result;
    for (auto &BB : *kernel) {
      for (auto &inst : BB) {
        PointerKind kind = PK_OTHER;
        if (auto *load = dyn_cast<LoadInst>(&inst)) {
          kind = getPointerKind(load->getPointerOperand(), true);
        } else if (auto *store = dyn_cast<StoreInst>(&inst)) {
          kind = getPointerKind(store->getPointerOperand(), true);
        } else if (auto *atomic = dyn_cast<AtomicRMWInst>(&inst)) {
          // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
          kind = getPointerKind(atomic->getPointerOperand(), true);
        } else if (auto *atomic = dyn_cast<AtomicCmpXchgInst>(&inst)) {
          // ATOMIC CAS //
          kind = getPointerKind(atomic->getPointerOperand(), true);
        } else if (auto *call = dyn_cast<CallInst>(&inst)) {
          Function* callee = call->getCalledFunction();
          if (callee == nullptr) continue;
          StringRef calleeName = callee->getName();
          if (calleeName.startswith("llvm.nvvm.atomic")) {
            // ATOMIC Inc/Dec //
            kind = getPointerKind(call->getArgOperand(0), true);
          } else if ( calleeName == "___cuprof_trace") {
            report_fatal_error("already instrumented!");
          } else if ( !calleeName.startswith("llvm.") ) {
            std::string error = "call to non-intrinsic: ";
            error.append(calleeName);
            report_fatal_error(error.c_str());
          }
        } else {
          continue;
        }

        if (kind != PK_GLOBAL)
          continue;
        result.push_back(&inst);
      }
    }
    return result;
  }

  
  std::vector<Instruction*> collectReturnInst(Function* kernel) {
    std::vector<Instruction*> result;
    for (auto &BB : *kernel) {
      for (auto &inst : BB) {
        if (isa<ReturnInst>(&inst)) {
          result.push_back(&inst);
        }
      }
    }
    return result;
  }



  

  bool setupAndGetKernelDebugData(Function* kernel, std::vector<char>& debug_data, std::vector<Instruction*> inst_list) {
    LLVMContext &ctx = kernel->getParent()->getContext();
    bool debugInfoNotFound = false;
    //char serialbuf[8] = {0};

    //uint64_t data_size = 0;

    uint64_t kernel_header_size =
      sizeof(trace_header_kernel_t) +
      sizeof(trace_header_inst_t) * inst_list.size() +
      4;
    trace_header_kernel_t * kernel_header =
      (trace_header_kernel_t *) malloc(kernel_header_size);
    if (!kernel_header) {
      fprintf(stderr, "cuprof: Failed to build debug data!\n");
      abort();
    }
    memset(kernel_header, 0, kernel_header_size);


    // append kernel info
    std::string kernel_name = kernel->getName().str();
    uint8_t kernel_name_len = std::min(kernel_name.length(), (size_t)0xFF);
    memcpy(kernel_header->kernel_name, kernel_name.c_str(), kernel_name_len);
    kernel_header->kernel_name_len = kernel_name_len;

    // '-g' option needed for debug info!
    uint32_t inst_id = 0;
    for (Instruction* inst : inst_list) {
      
      inst_id++; // id starts from 1

      // set inst info
      trace_header_inst_t *inst_header = &kernel_header->insts[inst_id - 1];
      inst_header->inst_id = inst_id;

      // set inst debug info
      const DebugLoc &loc = inst->getDebugLoc();
      if (loc) {
        std::string inst_path = loc->getFilename().str();
        while (inst_path.find("./") == 0) // remove leading "./" in path if exists
          inst_path.erase(0, 2);
        int inst_path_len = std::min(inst_path.length(), (size_t)0xFF);

        
        inst_header->row = loc->getLine();
        inst_header->col = loc->getColumn();
        inst_header->inst_filename_len = inst_path_len;
        memcpy(inst_header->inst_filename, inst_path.c_str(), inst_path_len);

      } else {
        debugInfoNotFound = true;
      }

      MDNode* metadata = MDNode::get(ctx, MDString::get(ctx, std::to_string(inst_id)));
      inst->setMetadata(TRACE_DEBUG_DATA, metadata);

    }
    kernel_header->insts_count = inst_id;

    
    char * kernel_data = (char *) malloc(get_max_header_size_after_packed(kernel_header));
    size_t kernel_data_size = header_pack(kernel_data, kernel_header);
    debug_data.reserve(debug_data.size() + kernel_data_size);
    debug_data.insert(debug_data.end(),
                      kernel_data,
                      kernel_data + kernel_data_size);

    free(kernel_data);
    

    return !debugInfoNotFound;
  }


  

  IRBuilderBase::InsertPoint setupTraceInfo(Function* kernel, TraceInfoValues *info) {
    LLVMContext &ctx = kernel->getParent()->getContext();
    Type *traceInfoTy = getTraceInfoType(ctx);

    IRBuilder<> IRB(kernel->getEntryBlock().getFirstNonPHI());

    Module &M = *kernel->getParent();
    std::string symbolName = getSymbolNameForKernel(kernel->getName());
    //errs() << "creating device symbol " << symbolName << "\n";
    auto* globalVar = defineDeviceGlobal(M, traceInfoTy, symbolName);
    assert(globalVar != nullptr);
    

    Value *AllocsPtr = IRB.CreateStructGEP(nullptr, globalVar, 0);
    Value *Allocs = IRB.CreateLoad(AllocsPtr, "allocs");

    Value *CommitsPtr = IRB.CreateStructGEP(nullptr, globalVar, 1);
    Value *Commits = IRB.CreateLoad(CommitsPtr, "commits");

    Value *RecordsPtr = IRB.CreateStructGEP(nullptr, globalVar, 2);
    Value *Records = IRB.CreateLoad(RecordsPtr, "records");

    IntegerType* I32Type = IntegerType::get(M.getContext(), 32);

    FunctionType *I32FnTy = FunctionType::get(I32Type, false);

    InlineAsm *SMIdASM = InlineAsm::get(I32FnTy,
                                        "mov.u32 $0, %smid;", "=r", false,
                                        InlineAsm::AsmDialect::AD_ATT );
    Value *SMId = IRB.CreateCall(SMIdASM);

    auto SMId64 = IRB.CreateZExtOrBitCast(SMId, IRB.getInt64Ty(), "desc");
    auto Desc = IRB.CreateShl(SMId64, 32);

    auto Slot = IRB.CreateAnd(SMId, IRB.getInt32(SLOTS_NUM - 1));

    
    info->Allocs = Allocs;
    info->Commits = Commits;
    info->Records = Records;
    info->Desc = Desc;
    info->Slot = Slot;
    
    return IRB.saveIP();
  }


  

  void instrumentMemAccess(Function *F, ArrayRef<Instruction*> MemAccesses,
                        TraceInfoValues *info) {
    Module &M = *F->getParent();

    Constant* TraceCall = getOrInsertTraceDecl(M);
    if (!TraceCall) {
      report_fatal_error("No ___cuprof_trace declaration found");
    }


    const DataLayout &DL = F->getParent()->getDataLayout();
    
    auto Allocs = info->Allocs;
    auto Commits = info->Commits;
    auto Records = info->Records;
    auto Slot = info->Slot;
    auto Desc = info->Desc;

    IRBuilder<> IRB(F->front().getFirstNonPHI());

      
    IntegerType* I64Type = IntegerType::get(M.getContext(), 64);
    FunctionType *ClockASMFTy = FunctionType::get(I64Type, false);
    InlineAsm *ClockASM = InlineAsm::get(ClockASMFTy,
                                         "mov.u64 $0, %clock64;", "=l", true,
                                         InlineAsm::AsmDialect::AD_ATT );

    for (auto *inst : MemAccesses) {
      Value *PtrOperand = nullptr;
      Value *Data = nullptr;
      Value *LDesc = nullptr;
      IRB.SetInsertPoint(inst);
      
      if (auto li = dyn_cast<LoadInst>(inst)) {
        PtrOperand = li->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_LOAD << RECORD_TYPE_SHIFT));
                             
      } else if (auto si = dyn_cast<StoreInst>(inst)) {
        PtrOperand = si->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_STORE << RECORD_TYPE_SHIFT));
                             
      } else if (auto ai = dyn_cast<AtomicRMWInst>(inst)) {
        // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
        PtrOperand = ai->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_ATOMIC << RECORD_TYPE_SHIFT));
                             
      } else if (auto ai = dyn_cast<AtomicCmpXchgInst>(inst)) {
        // ATOMIC CAS //
        PtrOperand = ai->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_ATOMIC << RECORD_TYPE_SHIFT));
                             
      } else if (auto *FuncCall = dyn_cast<CallInst>(inst)) {
        // ATOMIC Inc/Dec //
        assert(FuncCall->getCalledFunction()->getName()
               .startswith("llvm.nvvm.atomic"));
        PtrOperand = FuncCall->getArgOperand(0);
        LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_ATOMIC << RECORD_TYPE_SHIFT));
                             
      } else {
        report_fatal_error("invalid access type encountered, this should not have happened");
      }


      Data = IRB.CreatePtrToInt(PtrOperand,  IRB.getInt64Ty());
      auto PtrTy = dyn_cast<PointerType>(PtrOperand->getType());
      LDesc = IRB.CreateOr(LDesc, (uint64_t) DL.getTypeStoreSize(PtrTy->getElementType()));

      uint32_t inst_id = 0;
      if (MDNode* N = inst->getMetadata(TRACE_DEBUG_DATA)) {
        inst_id = (uint32_t) std::stoi(cast<MDString>(N->getOperand(0))->getString().str());
      }
      Constant *InstID = Constant::getIntegerValue(IRB.getInt32Ty(), APInt(32, inst_id));
      
      Value *Clock = IRB.CreateCall(ClockASM);
      IRB.CreateCall(TraceCall,  {Records, Allocs, Commits, LDesc, Data, Clock, Slot, InstID});
    }
  }


  

  void instrumentScheduling(Function *F, IRBuilderBase::InsertPoint ipfront,
                            ArrayRef<Instruction*> FunctionRetInst,
                            TraceInfoValues *info) {
	
    Module &M = *F->getParent();

    
    Constant* TraceCall = getOrInsertTraceDecl(M);
    if (!TraceCall) {
      report_fatal_error("No ___cuprof_trace declaration found");
    }
    
    auto Allocs = info->Allocs;
    auto Commits = info->Commits;
    auto Records = info->Records;
    auto Slot = info->Slot;
    auto Desc = info->Desc;



    IntegerType* I32Type = IntegerType::get(M.getContext(), 32);
    FunctionType *I32FnTy = FunctionType::get(I32Type, false);
    IntegerType* I64Type = IntegerType::get(M.getContext(), 64);
    FunctionType *I64FnTy = FunctionType::get(I64Type, false);
    
    InlineAsm *ClockASM = InlineAsm::get(I64FnTy,
                                         "mov.u64 $0, %clock64;", "=l", true,
                                         InlineAsm::AsmDialect::AD_ATT );
    InlineAsm *LaneIDASM = InlineAsm::get(I32FnTy,
                                          "mov.u32 $0, %laneid;", "=r", false,
                                          InlineAsm::AsmDialect::AD_ATT );
    
    Constant *InstID = Constant::getIntegerValue(I32Type, APInt(32, 0));
    
    
    IRBuilder<> IRB(F->front().getFirstNonPHI());
    IRB.restoreIP(ipfront);


    // trace call
    Value *Clock = IRB.CreateCall(ClockASM);
    Value *LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_EXECUTE << RECORD_TYPE_SHIFT));
    Value *LaneID = IRB.CreateCall(LaneIDASM);
    Instruction::CastOps LaneIDCastOp = CastInst::getCastOpcode(LaneID, false, I64Type, false);
    LaneID = IRB.CreateCast(LaneIDCastOp, LaneID, I64Type);
    IRB.CreateCall(TraceCall, {Records, Allocs, Commits, LDesc, LaneID, Clock, Slot, InstID});


    // trace ret
    for (auto *inst : FunctionRetInst) {
      Value *Data = nullptr;
      Value *LDesc = nullptr;
      IRB.SetInsertPoint(inst);

      if (isa<ReturnInst>(inst)) {
        Data = LaneID;
        LDesc = IRB.CreateOr(Desc, ((uint64_t)RECORD_RETURN << RECORD_TYPE_SHIFT));
      } else {
        report_fatal_error("invalid access type encountered, this should not have happened");
      }

      Value *Clock = IRB.CreateCall(ClockASM);
      IRB.CreateCall(TraceCall,  {Records, Allocs, Commits, LDesc, Data, Clock, Slot, InstID});
    }
  }

  
  
  GlobalVariable* setDebugData(Module &M, std::vector<char> input, const llvm::Twine &kernelName) {

    LLVMContext &ctx = M.getContext();

    const std::string varnameStr = getSymbolNameForKernel(kernelName, SYMBOL_DATA_VAR);
    
    
    GlobalVariable* debugData = M.getNamedGlobal(varnameStr.c_str());
    if (debugData != nullptr) {
      debugData->eraseFromParent();
      debugData = nullptr;
    }
    
    unsigned int data_len = input.size();
    ArrayRef<char> dataArrRef = ArrayRef<char>(input.data(), data_len);
    Constant* varInit = ConstantDataArray::get(ctx, dataArrRef);
    debugData = new GlobalVariable(M, varInit->getType(), false,
                                   GlobalValue::ExternalLinkage,
                                   varInit, varnameStr.c_str(), nullptr,
                                   GlobalValue::ThreadLocalMode::NotThreadLocal,
                                   1, false);
    debugData->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    debugData->setAlignment(1);


    return debugData;
  }

  
  bool runOnModule(Module &M) override {
    
    bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (!isCUDA) return false;

    
    // if kernel args is set, kernel filtering is enabled
    bool kernel_filtering = (args.kernel.size() != 0);
    

    bool debug_without_problem = true; // All debug data is written without problem
    std::vector<char> debug_data;
    for (auto *kernel : getKernelFunctions(M)) {
      std::string kernel_name_sym = kernel->getName().str();

      
      // kernel filtering
      
      if (kernel_filtering) {
        DISubprogram * kernel_debuginfo = kernel->getSubprogram();
        std::string kernel_name_orig;
        if (kernel_debuginfo) {
          kernel_name_orig = kernel_debuginfo->getName().str();
        }

        // stop instrumenting if not listed on enabled kernel
        if (std::find(args.kernel.begin(), args.kernel.end(), kernel_name_sym) == args.kernel.end() &&
            std::find(args.kernel.begin(), args.kernel.end(), kernel_name_orig) == args.kernel.end()) {
          continue;
        }
        
        fprintf(stderr, "cuprof: Selective kernel tracing enabled (%s)\n", kernel_name_sym.c_str());
      }


      // kernel instrumentation
      
      auto accesses = collectGlobalMemAccesses(kernel);
      auto retinsts = collectReturnInst(kernel);
      
      TraceInfoValues info;
      IRBuilderBase::InsertPoint ipfront = setupTraceInfo(kernel, &info);
      
      
      if (args.trace_mem) {
        debug_without_problem &= setupAndGetKernelDebugData(kernel, debug_data, accesses);
        instrumentMemAccess(kernel, accesses, &info);
      }

      if (args.trace_thread) {
        instrumentScheduling(kernel, ipfront, retinsts, &info);
      }

      
      setDebugData(M, debug_data, kernel_name_sym);
      debug_data.clear();
    }
    
    if (!debug_without_problem) {
      std::cerr << "cuprof: No memory access data for \""
                << M.getModuleIdentifier()
                << "\" found! Check if \"-g\" option is set.\n";
    }

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
  }

};
char InstrumentDevicePass::ID = 0;

namespace llvm {
  Pass *createInstrumentDevicePass(InstrumentPassArg args) {
    return new InstrumentDevicePass(args);
  }
}

static RegisterPass<InstrumentDevicePass> X("memtrace-device", "includes static and dynamic load/store counting", false, false);





  
/*
  std::string getModuleID(std::string ModuleID) {

    // "#" to "_#"
    size_t pos = (size_t)-2;
    while ((pos = ModuleID.find('#', pos+2)) != std::string::npos)
      ModuleID = ModuleID.replace(pos, 1, "_#");

    // "//" to "/"
    pos = (size_t)0;
    while ((pos = ModuleID.find("//", pos)) != std::string::npos)
      ModuleID = ModuleID.replace(pos, 2, "/");

    // "/" to "##"
    pos = (size_t)-2;
    while ((pos = ModuleID.find('/', pos+2)) != std::string::npos)
      ModuleID = ModuleID.replace(pos, 1, "##");
    

    ModuleID = ModuleID.insert(0, "debuginfo-").append(".txt");

    return ModuleID;
  }

  void setFuncAsGlobalKernel(Module &M, Function *F) {
    
    LLVMContext &ctx = M.getContext();
    NamedMDNode *nvvmAnnotation = M.getNamedMetadata("nvvm.annotations");

    if (nvvmAnnotation) {

      ArrayRef<Metadata*> arrRef({
          ValueAsMetadata::get(F),
          MDString::get(ctx, "kernel"),
          ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 1))
        });
      
      MDNode *mdnnew = MDNode::get(ctx, arrRef);      
      nvvmAnnotation->addOperand(mdnnew);

    }
    
  }
*/



/*
namespace {
  class PluginExampleAction : public clang::PluginASTAction {
  protected:
    // this gets called by Clang when it invokes our Plugin
    clang::ASTConsumer *CreateASTConsumer(clang::CompilerInstance &CI, StringRef file) {
    return new clang::ASTConsumer(&CI);
    }
    // implement this function if you want to parse custom cmd-line args
    bool ParseArgs(const clang::CompilerInstance &CI, const std::vector<std::string> &args) {
      for (unsigned i = 0, e = args.size(); i != e; ++i) {
        if (args[i] == "-some-arg") {
          // Handle the command line argument.
        }
      }
      return true;
    }
  };
}

static clang::FrontendPluginRegistry::Add<PluginExampleAction> Y("my-plugin-name", "my plugin description");
*/





/*
GlobalVariable* appendDebugData(Module &M, const char* input) {
    
  std::string dataStr = "\t";
  GlobalVariable* debugData = M.getNamedGlobal(TRACE_DEBUG_DATA);
  

  if (debugData != nullptr) {
    if (auto initOld = dyn_cast<ConstantDataArray>(debugData->getInitializer())) {
      dataStr = initOld->getAsString();
    }
    
    debugData->eraseFromParent();
    debugData = nullptr;
  }
    

  if (!dataStr.empty() && dataStr.back() == 0)
    dataStr.back() = '\t';
  dataStr.append(input);
  dataStr.push_back(0);
  const int txtlen = dataStr.size();
  Type *i8 = Type::getInt8Ty(M.getContext());
  ArrayType* arty = ArrayType::get(i8, txtlen);

    
  ArrayRef<char> dataArrRef = ArrayRef<char>(dataStr.c_str(), txtlen);
  Constant* initNew = ConstantDataArray::get(M.getContext(), dataArrRef);
    
  debugData = new GlobalVariable(M, //Module
                                 initNew->getType(), //Type
                                 true, //isConstant
                                 GlobalValue::ExternalLinkage, //Linkage
                                 initNew, //Initializer
                                 TRACE_DEBUG_DATA,  //Name
                                 nullptr,
                                 llvm::GlobalValue::ThreadLocalMode::NotThreadLocal,
                                 1,
                                 false);

  debugData->setAlignment(1);

  //debugData->setDSOLocal(true);
  //debugData->setVisibility(llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
  //debugData->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  
  printf(TRACE_DEBUG_DATA " = %s\n", dataStr.c_str());

  return debugData;
}

void setDebugDataExtern(Module &M) {

  
#define LINK_TEST_DEBUG 1
  
#if LINK_TEST_DEBUG == 0
  // Do Nothing
#elif LINK_TEST_DEBUG == 1
  appendDebugData(M, "DEBUG_FOR_CUDATRACE_LINKAGE");
#else


  GlobalVariable* debugData = M.getNamedGlobal(TRACE_DEBUG_DATA);

  
  if (debugData == nullptr) {
    //Constant *zero = ConstantDataArray::get(M.getContext(), dataArrRef); // Constant::getNullValue(ArrayType::get(Type::getInt8Ty(M.getContext()));
    Constant *zero = ConstantDataArray::getNullValue(Type::getInt8PtrTy(M.getContext()));
    
    
    debugData = new GlobalVariable(M, zero->getType(), true, GlobalValue::LinkOnceAnyLinkage, zero, TRACE_DEBUG_DATA);

    debugData->setAlignment(1);

    //debugData->setDSOLocal(true);
    //debugData->setVisibility(llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    //debugData->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  }

#endif
  
}
*/


/*


#define APPEND_DINFO_FUNC_NAME "__append_debug_info"
#define GLOBAL_CTOR_ARR_NAME "llvm.global_ctors"

  GlobalVariable* appendStaticInitFunction(Module &M, const char* input) {

    LLVMContext &ctx = M.getContext();
    
    std::string dataStr = "\t";
    GlobalVariable* globalCtorVar = M.getNamedGlobal(GLOBAL_CTOR_ARR_NAME);
    
    SmallVector<Constant *, 8> Ctors;
    Constant *init = nullptr;

    //globalCtorVar->dump();

    if (globalCtorVar != nullptr) {
      if (auto initOld = dyn_cast<ConstantArray>(globalCtorVar->getInitializer())) {
        unsigned int i = 0;
        while (auto elem = initOld->getAggregateElement(i++)) {
          Ctors.push_back(elem);
        }
      }
    
      globalCtorVar->eraseFromParent();
      globalCtorVar = nullptr;
    }
    
    
    // Ctor function type is void()*.
    Type *VoidTy = Type::getVoidTy(ctx);
    Type *VoidPtrTy = Type::getInt32PtrTy(ctx);
    Type *Int32Ty = Type::getInt32Ty(ctx);
    Type *Int8PtrTy = Type::getInt8PtrTy(ctx);
    FunctionType* CtorFTy = FunctionType::get(VoidTy, false);
    Type *CtorPFTy = PointerType::getUnqual(CtorFTy);

    // Get the type of a ctor entry, { i32, void ()*, i8* }.
    StructType *CtorStructTy = StructType::get(
      (Type*) Int32Ty, (Type*) PointerType::getUnqual(CtorFTy), (Type*) Int8PtrTy);


    Function* F = Function::Create(CtorFTy, Function::ExternalLinkage, APPEND_DINFO_FUNC_NAME, M);
    
    Constant *S[] = {
      ConstantInt::get(Int32Ty, 65535, false),
      ConstantExpr::getBitCast(F, CtorPFTy),
      Constant::getNullValue(Int8PtrTy)
    };
    
    Ctors.push_back(ConstantStruct::get(CtorStructTy, S));
    
    
    ArrayRef<Constant *> CtorArrRef(Ctors.begin(), Ctors.end());
    ArrayType *AT = ArrayType::get(CtorStructTy, Ctors.size());
    
    globalCtorVar = new GlobalVariable(M, AT, false,
                                       GlobalValue::AppendingLinkage,
                                       ConstantArray::get(AT, Ctors),
                                       GLOBAL_CTOR_ARR_NAME);
    
    
    
    if (globalCtorVar != nullptr)
      globalCtorVar->dump();
    
    return globalCtorVar;
  }

  
*/




  
/*
  if (!dataStr.empty() && dataStr.back() == 0)
  dataStr.back() = '\t';
  dataStr.append(input);
  dataStr.push_back(0);
  const int txtlen = dataStr.size();
  Type *i8 = Type::getInt8Ty(M.getContext());
  ArrayType* arty = ArrayType::get(i8, txtlen);

    
  ArrayRef<char> dataArrRef = ArrayRef<char>(dataStr.c_str(), txtlen);
  Constant* initNew = ConstantDataArray::get(M.getContext(), dataArrRef);
    
  debugData = new GlobalVariable(M, //Module
  initNew->getType(), //Type
  true, //isConstant
  GlobalValue::AppendingLinkage, //Linkage
  initNew, //Initializer
  TRACE_DEBUG_DATA,  //Name
  nullptr,
  llvm::GlobalValue::ThreadLocalMode::NotThreadLocal,
  1,
  false);
  debugData->setSection(".ctor");

  debugData->setAlignment(1);

*/


  /*


    Constant * FC = M.getOrInsertFunction(funcname, Type::getVoidTy(ctx), NULL);
    Function * func = dyn_cast<Function>(FC);
    Type *debugDataTy = getDebugDataType(ctx);
    
    if (func) {
      func->setCallingConv(CallingConv::C);
      BasicBlock* block = BasicBlock::Create(ctx, "entry", func);

      IRBuilder<> IRB(block);


      
      GlobalVariable* debugDataGv = defineDeviceGlobal(M, debugDataTy, "___DEBUG_DATA_PTR");
      assert(debugDataGv != nullptr);
    

      Value *debugDataLenGvPtr = IRB.CreateStructGEP(nullptr, debugDataGv, 0);
      Value *debugDataLenGv = IRB.CreateLoad(debugDataLenGvPtr, "debugdatalen");
      
      Value *debugDataVarGvPtr = IRB.CreateStructGEP(nullptr, debugDataGv, 1);
      Value *debugDataVarGv = IRB.CreateLoad(debugDataVarGvPtr, "debugdata");

      
      Value *debugDataL = IRB.CreateLoad(debugData, "debugdata22");

      //GlobalVariable *debugDataTarget = M.getNamedGlobal("___DEBUG_DATA_PTR");

      //debugDataGv->dump();
      //debugData->dump();
      //ArrayRef<unsigned int> idxArr({0});
      //IRB.CreateInsertValue(debugDataGv, ConstantInt::get(IntegerType::get(ctx, 64), 0), idxArr);
      //IRB.CreateStore(debugDataVarGv, debugDataVarGvPtr);
      //printf("\n\n\ndebugDataVarGvPtr\n\n");
      //debugDataVarGvPtr->dump();
      //printf("\n\n\ndebugDataVarGv\n\n");
      //debugDataVarGv->dump();
      //printf("\n\n\ndebugDataL\n\n");
      //debugDataL->dump();
      //varInit->dump();
      //IRB.CreateLoad(debugDataGv);
      //Value *valstruct = IRB.CreateAlloca(debugDataTy, 1);
      //Value *valstructptr1 = IRB.CreateStructGEP(nullptr, debugDataGv, 0);
      //Value *debugDataLenGv = IRB.CreateLoad(debugDataLenGvPtr, "debugdatalen");
      
      //Value *valstructptr2 = IRB.CreateStructGEP(nullptr, debugDataGv, 1);
      //Value *debugDataVarGv = IRB.CreateLoad(debugDataVarGvPtr, "debugdata");
      //IRB.CreateStore(debugData, valstructptr1);

      
      //IRB.CreateStore(valcast, debugDataLenGvPtr);
      //IRB.CreateStore(debugData, debugDataGv);
      //GlobalVariable *debugDataTarget = M.getNamedGlobal("___DEBUG_DATA_PTR");


      //GlobalVariable* testGv = defineDeviceGlobal(M, ArrayType::get(Type::getInt8Ty(ctx), 1024), "___testestasetasdf");
      //Constant *testinit = testGv->getInitializer();
      //ArrayRef<unsigned int> idxArr({0});
      //Value *Res = UndefValue::get(ArrayType::get(Type::getInt8Ty(ctx), 1024));
      IRB.CreateStore(ConstantInt::get(Type::getInt32Ty(ctx), 1302), debugDataLenGvPtr);
      //IRB.CreateInsertValue(debugDataGv, ConstantInt::get(Type::getInt32Ty(ctx), 1302), 0);
      Value* result = IRB.CreateInsertValue(debugDataVarGv, ConstantInt::get(Type::getInt8Ty(ctx), '#'), 3);
      IRB.CreateStore(result, debugDataVarGvPtr);



      //intrinst->setDest(debugDataVarGvPtr);
      //intrinst->setLength(ConstantInt::get(Type::getInt32Ty(ctx), 1024));
      //intrinst->
      

      //debugData->dump();
      //testGv->dump();
      //Value *test = IRB.CreateBitCast(debugData, testGv->
      //IRB.CreateStore(debugData, testGv);

      
      
      //CastInst *debugDataCastInst = AddrSpaceCastInst(debugDataTarget, Int8PtrTy
      //Value *debugDataPtr = IRB.CreateBitCast(debugDataTarget, Int8PtrTy);
      //IRB.CreateStore(debugData, debugDataPtr);
      IRB.CreateRetVoid();
    }
  */


    
    /*
    // Construct the constructor and destructor arrays.
    for (const auto &I : Fns) {
      Constant *S[] = {
        ConstantInt::get(Int32Ty, I.Priority, false),
        ConstantExpr::getBitCast(I.Initializer, CtorPFTy),
        (I.AssociatedData
        ? ConstantExpr::getBitCast(I.AssociatedData, VoidPtrTy)
        : Constant::getNullValue(VoidPtrTy))};
        Ctors.push_back(ConstantStruct::get(CtorStructTy, S));
        }
    */
    

    //debugData->setDSOLocal(true);
    //debugData->setVisibility(llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    //debugData->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);


    
    /*
    // Ctor function type is void()*.
    FunctionType* CtorFTy = FunctionType::get(VoidTy, false);
    Type *CtorPFTy = PointerType::getUnqual(CtorFTy);

    // Get the type of a ctor entry, { i32, void ()*, i8* }.
    StructType *CtorStructTy = StructType::get(
      Int32Ty, PointerType::getUnqual(CtorFTy), VoidPtrTy, nullptr);

    // Construct the constructor and destructor arrays.
    SmallVector<Constant *, 8> Ctors;
    for (const auto &I : Fns) {
      Constant *S[] = {
        ConstantInt::get(Int32Ty, I.Priority, false),
        ConstantExpr::getBitCast(I.Initializer, CtorPFTy),
        (I.AssociatedData
         ? ConstantExpr::getBitCast(I.AssociatedData, VoidPtrTy)
         : Constant::getNullValue(VoidPtrTy))};
      Ctors.push_back(ConstantStruct::get(CtorStructTy, S));
    }

    if (!Ctors.empty()) {
      ArrayType *AT = ArrayType::get(CtorStructTy, Ctors.size());
      new GlobalVariable(TheModule, AT, false,
                         GlobalValue::AppendingLinkage,
                         ConstantArray::get(AT, Ctors),
                         GlobalName);
    }
    */


