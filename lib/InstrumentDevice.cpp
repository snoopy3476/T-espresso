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
      
      
      debug_without_problem &= setupAndGetKernelDebugData(kernel, debug_data, accesses);
      if (args.trace_mem) {
        instrumentMemAccess(kernel, accesses, &info);
      }
      setDebugData(M, debug_data, kernel_name_sym); //////////////////////

      if (args.trace_thread) {
        instrumentScheduling(kernel, ipfront, retinsts, &info);
      }

      
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

