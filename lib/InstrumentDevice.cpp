#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
/*
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
*/

#include <set>
#include <iostream>
#include <fstream>

#define INCLUDE_LLVM_MEMTRACE_STUFF
#include "Common.h"

#define DEBUG_TYPE "memtrace-device"
#define TRACE_DEBUG_DATA "___CUDATRACE_DEBUG_DATA"

#define ADDRESS_SPACE_GENERIC 0
#define ADDRESS_SPACE_GLOBAL 1
#define ADDRESS_SPACE_INTERNAL 2
#define ADDRESS_SPACE_SHARED 3
#define ADDRESS_SPACE_CONSTANT 4
#define ADDRESS_SPACE_LOCAL 5

/*
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"
*/

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

  return M.getOrInsertFunction("__mem_trace", voidTy,
			       i8PtrTy, i8PtrTy, i8PtrTy, i64Ty, i64Ty, i64Ty, i32Ty, i32Ty);
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
  const bool TraceThread, TraceMem;

  InstrumentDevicePass(bool TraceThreadInput = true, bool TraceMemInput = true) : ModulePass(ID), TraceThread(TraceThreadInput), TraceMem(TraceMemInput) { }

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
          } else if ( calleeName == "__mem_trace") {
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

  

  bool setupAndGetDebugInfo(Function* kernel, std::string& debug_info, std::vector<Instruction*> inst_list) {
    LLVMContext &ctx = kernel->getParent()->getContext();
    bool debugInfoNotFound = false;
    
    debug_info = debug_info.append("<" + kernel->getName().str() + ">\n");

    // '-g' option needed for debug info!
    int inst_id = 1;
    for (Instruction* inst : inst_list) {
      //printf("[%s] ", inst->getOpcodeName());
      debug_info = debug_info.append("\t[" + std::to_string(inst_id) + "] ");
      const DebugLoc &loc = inst->getDebugLoc();
      if (loc) {
        debug_info = debug_info.append(inst->getOpcodeName()).append(
          (" " + loc->getDirectory() + "/" + loc->getFilename() + " : (" +
           std::to_string(loc->getLine()) + "," +
           std::to_string(loc->getColumn()) + ")\n").str());
      } else {
        debugInfoNotFound = true;
        debug_info = debug_info.append("(no info)\n");
      }

      MDNode* metadata = MDNode::get(ctx, MDString::get(ctx, std::to_string(inst_id)));
      inst->setMetadata(TRACE_DEBUG_DATA, metadata);

      inst_id++;
    }
    debug_info = debug_info.append("\n");

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

    // Execution Clock
    IntegerType* I64Type = IntegerType::get(M.getContext(), 64);
    FunctionType *I64FnTy = FunctionType::get(I64Type, false);
    InlineAsm *ClockASM = InlineAsm::get(I64FnTy,
                                         "mov.u64 $0, %clock64;", "=l", true,
                                         InlineAsm::AsmDialect::AD_ATT );
    //Value *Clock = IRB.CreateCall(ClockASM);

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
      report_fatal_error("No __mem_trace declaration found");
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
        LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_LOAD << ACCESS_TYPE_SHIFT));
                             
      } else if (auto si = dyn_cast<StoreInst>(inst)) {
        PtrOperand = si->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_STORE << ACCESS_TYPE_SHIFT));
                             
      } else if (auto ai = dyn_cast<AtomicRMWInst>(inst)) {
        // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
        PtrOperand = ai->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_ATOMIC << ACCESS_TYPE_SHIFT));
                             
      } else if (auto ai = dyn_cast<AtomicCmpXchgInst>(inst)) {
        // ATOMIC CAS //
        PtrOperand = ai->getPointerOperand();
        LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_ATOMIC << ACCESS_TYPE_SHIFT));
                             
      } else if (auto *FuncCall = dyn_cast<CallInst>(inst)) {
        // ATOMIC Inc/Dec //
        assert(FuncCall->getCalledFunction()->getName()
               .startswith("llvm.nvvm.atomic"));
        PtrOperand = FuncCall->getArgOperand(0);
        LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_ATOMIC << ACCESS_TYPE_SHIFT));
                             
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
      report_fatal_error("No __mem_trace declaration found");
    }

    //const DataLayout &DL = F->getParent()->getDataLayout();
    
    auto Allocs = info->Allocs;
    auto Commits = info->Commits;
    auto Records = info->Records;
    auto Slot = info->Slot;
    auto Desc = info->Desc;

    //IRBuilder<> IRB(F->front().getFirstNonPHI());


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
    Value *LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_CALL << ACCESS_TYPE_SHIFT));
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
        LDesc = IRB.CreateOr(Desc, ((uint64_t)ACCESS_RETURN << ACCESS_TYPE_SHIFT));
      } else {
        report_fatal_error("invalid access type encountered, this should not have happened");
      }

      Value *Clock = IRB.CreateCall(ClockASM);
      IRB.CreateCall(TraceCall,  {Records, Allocs, Commits, LDesc, Data, Clock, Slot, InstID});
    }
  }

  
  
#define ESCAPE_DIR_CHAR '#'
#define ESCAPE_DIR_TO_STR "_#"
#define DIR_DUP_STR "//"
#define DIR_DUP_TO_STR "/"
#define DIR_CHAR '/'
#define DIR_TO_STR "##"
#define DEBUG_DATA "debuginfo.txt"
#define DEBUG_TMP_DATA_PREFIX "debuginfo-"

  std::string getModuleID(std::string ModuleID) {

    // "#" to "_#"
    size_t pos = (size_t)-2;
    while ((pos = ModuleID.find(ESCAPE_DIR_CHAR, pos+2)) != std::string::npos)
      ModuleID = ModuleID.replace(pos, 1, ESCAPE_DIR_TO_STR);

    // "//" to "/"
    pos = (size_t)0;
    while ((pos = ModuleID.find(DIR_DUP_STR, pos)) != std::string::npos)
      ModuleID = ModuleID.replace(pos, 2, DIR_DUP_TO_STR);

    // "/" to "##"
    pos = (size_t)-2;
    while ((pos = ModuleID.find(DIR_CHAR, pos+2)) != std::string::npos)
      ModuleID = ModuleID.replace(pos, 1, DIR_TO_STR);
    

    ModuleID = ModuleID.insert(0, DEBUG_TMP_DATA_PREFIX).append(".txt");

    return ModuleID;
  }
  

  
  bool runOnModule(Module &M) override {
    //M.getContext().getOption<typename ValT, typename Base, ValT (Base::*Mem)>();
    //printf("%d\n", TestInt++);
    //pass_registry->getPassInfo(const void *TI);

    LLVMContext &ctx = M.getContext();
    

  
    bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (!isCUDA) return false;



    const std::string ModuleIDRaw = std::string(M.getModuleIdentifier());
    const std::string ModuleID = getModuleID(std::string(ModuleIDRaw));
    std::string debug_info = "";
    bool debugWithoutProblem = true;
    //printf("Module ID: %s\n", ModuleID.c_str());

    
    for (auto *kernel : getKernelFunctions(M)) {
      
      auto accesses = collectGlobalMemAccesses(kernel);
      auto retinsts = collectReturnInst(kernel);
      
      TraceInfoValues info;
      IRBuilderBase::InsertPoint ipfront = setupTraceInfo(kernel, &info);
      
      
      if (TraceMem) {
        debugWithoutProblem &= setupAndGetDebugInfo(kernel, debug_info, accesses);
        instrumentMemAccess(kernel, accesses, &info);
      }

      if (TraceThread) {
        instrumentScheduling(kernel, ipfront, retinsts, &info);
      }
      
    }
    
    std::ofstream debug_info_fs(ModuleID);
    debug_info_fs << debug_info;
    debug_info_fs.close();
    
    if (!debugWithoutProblem) {
      std::cerr << "Debug info for \"" << ModuleIDRaw << "\" not found! Check if \"-g\" option is set.\n";
    }

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
  }

};
char InstrumentDevicePass::ID = 0;

namespace llvm {
  Pass *createInstrumentDevicePass(bool TraceThread, bool TraceMem) {
    return new InstrumentDevicePass(TraceThread, TraceMem);
  }
}

static RegisterPass<InstrumentDevicePass> X("memtrace-device", "includes static and dynamic load/store counting", false, false);







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
