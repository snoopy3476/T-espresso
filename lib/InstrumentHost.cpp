/*******************************************************************************
 * This host instrumentation pass inserts calls into the host support library.
 * The host support library is used to set up queues for traces that are sinked
 * into a thread that writes them into a file.
 * A kernel launch is split into two major parts:
 * 1. cudaConfigureCall()
 * 2. <wrapper>() -> cudaLaunch()
 * The function cudaConfigurCall sets up the execution grid and stream to
 * execute in and the wrapper function sets up kernel arguments and launches
 * the kernel.
 * Instrumentation requires the stream, set in cudaConfigureCall, as well as the
 * kernel name, implicitly "set" by the wrapper function.
 * This pass defines the location of a kernel launch as the call to
 * cudaConfigureCall, which the module is searched for.
 *
 * Finding the kernel name boils down to following the execution path assuming
 * no errors occur during config and argument setup until we find:
 * 1. a call cudaLaunch and return the name of the first operand, OR
 * 2. a call to something other than cudaSetupArguent and return its name
 *
 */

#include <set>
#include <cuda_runtime_api.h>


#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"


#include "LocateKCalls.h"

#define INCLUDE_LLVM_MEMTRACE_STUFF
#include "Common.h"

#define DEBUG_TYPE "memtrace-host"
#define TRACE_DEBUG_DATA "___CUDATRACE_DEBUG_DATA"

using namespace llvm;


/*
void createPrintf(IRBuilder<> &IRB, const Twine &fmt, ArrayRef<Value*> values) {
  Module &M = *IRB.GetInsertBlock()->getModule();
  Function* Printf = M.getFunction("printf");
  if (Printf) {
    auto *FormatGlobal = IRB.CreateGlobalString(fmt.getSingleStringRef());
    Type* charPtrTy = IRB.getInt8Ty()->getPointerTo();
    Value* Format = IRB.CreateBitCast(FormatGlobal, charPtrTy);
    SmallVector<Value*, 4> args;
    args.append({Format});
    args.append(values.begin(), values.end());
    IRB.CreateCall(Printf, args);
  }
}

Instruction* createMalloc(IRBuilder<> &IRB, Type *ty,
  Value *count = nullptr) {
  Type* int32_ty = IRB.getInt32Ty();
  Type* int64_ty = IRB.getInt64Ty();
  if (count == nullptr) {
    count = ConstantInt::get(int64_ty, 1);
  }
    
  Constant* ty_size = ConstantExpr::getSizeOf(ty);
  Constant* elem_size = ConstantExpr::getTruncOrBitCast(ty_size, int32_ty);
  Instruction* malloc_inst = CallInst::CreateMalloc(IRB.GetInsertBlock(),
                                                    int32_ty, ty, elem_size,
                                                    count, nullptr, "");
  return IRB.Insert(malloc_inst);
}
*/
  


struct InstrumentHost : public ModulePass {
  static char ID;
  InstrumentHost() : ModulePass(ID) {}

  Type* traceInfoTy = nullptr;
  Type *SizeTy = nullptr;
  Type *SizePtrTy = nullptr;
  Type *CudaMemcpyKindTy = nullptr;
  Type *CudaErrorTy = nullptr;

  Constant *AccdatCtor = nullptr;
  Constant *AccdatDtor = nullptr;
  Constant *AccdatAppend = nullptr;
  Constant *TraceFillInfo = nullptr;
  Constant *TraceCopyToSymbol = nullptr;
  Constant *TraceTouch = nullptr;
  Constant *TraceStart = nullptr;
  Constant *TraceStop = nullptr;
  Constant *CudaMemcpyFromSymbol = nullptr;
  Constant *CudaGetSymbolSize = nullptr;

  /** Sets up pointers to (and inserts prototypes of) the utility functions
   * from the host-support library.
   * We're pretending all pointer types are identical, linker does not
   * complain in tests.
   *
   * Reference:
   * void __trace_fill_info(const void *info, cudaStream_t stream);
   * void __trace_copy_to_symbol(cudaStream_t stream, const char* symbol, const void *info);
   * void __trace_touch(cudaStream_t stream);
   * void __trace_start(cudaStream_t stream, const char *kernel_name);
   * void __trace_stop(cudaStream_t stream);
   */

  

  GlobalVariable* getOrCreateGlobalVar(Module &M, Type* T, const Twine &name) {
    // first see if the variable already exists
  
    GlobalVariable *Global = M.getGlobalVariable(name.getSingleStringRef());
    if (Global) {
      return Global;
    }

    // Variable does not exist, so we create one and register it.
    // This happens if a kernel is called in a module it is not registered in.
    Constant *zero = Constant::getNullValue(T);
    Global = new GlobalVariable(M, T, false, GlobalValue::LinkOnceAnyLinkage, zero, name);
    Global->setAlignment(8);
    assert(Global != nullptr);
    return Global;
  }
  
  void findOrInsertRuntimeFunctions(Module &M) {
    LLVMContext &ctx = M.getContext();
    Type* i8PtrTy = Type::getInt8PtrTy(ctx);
    Type* voidPtrTy = Type::getInt8PtrTy(ctx);
    Type* stringTy = Type::getInt8PtrTy(ctx);
    Type* i16Ty = Type::getInt16Ty(ctx);
    Type* i32Ty = Type::getInt32Ty(ctx);
    Type* i64Ty = Type::getInt64Ty(ctx);
    Type* voidTy = Type::getVoidTy(ctx);

    
    AccdatCtor = M.getOrInsertFunction("___cuprof_accdat_ctor",
                                       voidTy);
    AccdatDtor = M.getOrInsertFunction("___cuprof_accdat_dtor",
                                       voidTy);
    AccdatAppend = M.getOrInsertFunction("___cuprof_accdat_append",
                                         voidTy, stringTy, i64Ty);

    
    TraceFillInfo = M.getOrInsertFunction("__trace_fill_info",
                                          voidTy, voidPtrTy, i8PtrTy);
    TraceCopyToSymbol = M.getOrInsertFunction("__trace_copy_to_symbol",
                                              voidTy, i8PtrTy, voidPtrTy, voidPtrTy);
    TraceTouch = M.getOrInsertFunction("__trace_touch",
                                       voidTy, i8PtrTy);
    TraceStart = M.getOrInsertFunction("__trace_start",
                                       voidTy, i8PtrTy, stringTy, i16Ty);
    TraceStop = M.getOrInsertFunction("__trace_stop",
                                      voidTy, i8PtrTy);
    
    CudaMemcpyFromSymbol = M.getOrInsertFunction("cudaMemcpyFromSymbol",
                                                 CudaErrorTy, i8PtrTy, i8PtrTy, SizeTy, SizeTy, CudaMemcpyKindTy);
    CudaGetSymbolSize = M.getOrInsertFunction("cudaGetSymbolSize",
                                              CudaErrorTy, i8PtrTy, i8PtrTy);
  }

  /** Updates kernel calls to set up tracing infrastructure on host and device
   * before starting the kernel and tearing everything down afterwards.
   */
  void patchKernelCall(CallInst *configureCall, Instruction* launch,
                       const StringRef kernelName) {
    assert(configureCall->getNumArgOperands() == 6);
    auto *stream = configureCall->getArgOperand(5);

    // insert preparational steps directly after cudaConfigureCall
    // 0. touch consumer to create new one if necessary
    // 1. start/prepare trace consumer for stream
    // 2. get trace consumer info
    // 3. copy trace consumer info to device

      

    IRBuilder<> IRB(configureCall->getNextNode());

    Type* i8Ty = IRB.getInt8Ty();
    Type* i16Ty = IRB.getInt16Ty();
    Type* i32Ty = IRB.getInt32Ty();
    Type* i64Ty = IRB.getInt64Ty();
    Type* i8PtrTy = IRB.getInt8PtrTy();

    Value* kernelNameVal = IRB.CreateGlobalStringPtr(kernelName);

    // try adding in global symbol + cuda registration
    Module &M = *configureCall->getParent()->getParent()->getParent();

    Type* GlobalVarType = traceInfoTy;
    std::string kernelSymbolName = getSymbolNameForKernel(kernelName);
    //printf("kernelName : %s\n", kernelName.str().c_str());

    GlobalVariable *globalVar = getOrCreateGlobalVar(M, GlobalVarType, kernelSymbolName);

    auto *globalVarPtr = IRB.CreateBitCast(globalVar, i8PtrTy);
    auto* streamPtr = IRB.CreateBitCast(stream, i8PtrTy);



    // Thread block size of the current kernel call
      
    Instruction::CastOps castOp_i64_i16 = CastInst::getCastOpcode(
      Constant::getNullValue(i64Ty), false, i16Ty, false);
    Instruction::CastOps castOp_i32_i16 = CastInst::getCastOpcode(
      Constant::getNullValue(i32Ty), false, i16Ty, false);
      
    Value *blockSize = configureCall->getArgOperand(2);   // <32 bit: y> <32 bit: x>
    Value *blockSize_z = configureCall->getArgOperand(3); // <32 bit: z>
    blockSize = IRB.CreateMul(
      IRB.CreateAnd(blockSize, 0xFFFFFFFF),
      IRB.CreateLShr(blockSize, 32)
      ); // x * y
    blockSize = IRB.CreateMul(
      IRB.CreateCast(castOp_i64_i16, blockSize, i16Ty),
      IRB.CreateCast(castOp_i32_i16, blockSize_z, i16Ty)
      ); // <uint16_t>(x*y) * <uint16_t>z
      
    IRB.CreateCall(TraceTouch, {streamPtr});
    IRB.CreateCall(TraceStart, {streamPtr, kernelNameVal, blockSize});

    const DataLayout &DL = configureCall->getParent()->getParent()->getParent()->getDataLayout();

    size_t bufSize = DL.getTypeStoreSize(GlobalVarType);

    Value* infoBuf = IRB.CreateAlloca(i8Ty, IRB.getInt32(bufSize));
    IRB.CreateCall(TraceFillInfo, {infoBuf, streamPtr});
    IRB.CreateCall(TraceCopyToSymbol, {streamPtr, globalVarPtr, infoBuf});

    // insert finishing steps after kernel launch was issued
    // 1. stop trace consumer
    IRB.SetInsertPoint(launch->getNextNode());
    IRB.CreateCall(TraceStop, {streamPtr});
  }


  
#define GLOBAL_CTOR_ARR_NAME "llvm.global_ctors"
#define GLOBAL_DTOR_ARR_NAME "llvm.global_dtors"
#define GLOBAL_CDTOR_DEFAULT_PRIORITY 2147483647
  
  enum GlobalCdtorType {
    GLOBAL_CTOR = 0,
    GLOBAL_DTOR = 1
  };

  bool appendGlobalCtorDtor(Module &M, Function *func, GlobalCdtorType type,
                            uint32_t priority = GLOBAL_CDTOR_DEFAULT_PRIORITY) {

    if (!func) return false;

    
    LLVMContext &ctx = M.getContext();
    
    Type *VoidTy = Type::getVoidTy(ctx);
    Type *Int32Ty = Type::getInt32Ty(ctx);
    Type *Int8PtrTy = Type::getInt8PtrTy(ctx);
    FunctionType* CtorFTy = FunctionType::get(VoidTy, false);
    Type *CtorPFTy = PointerType::getUnqual(CtorFTy);
    StructType *CtorStructTy = StructType::get(
      (Type*) Int32Ty, (Type*) PointerType::getUnqual(CtorFTy), (Type*) Int8PtrTy);

    
    const char * gv_name;
    switch (type) {
      
    case GLOBAL_DTOR: // dtor
      gv_name = GLOBAL_DTOR_ARR_NAME;
      break;

    default: // ctor
      gv_name = GLOBAL_CTOR_ARR_NAME;
      break;
    }

    
    GlobalVariable* gv = M.getNamedGlobal(gv_name);
    
    SmallVector<Constant *, 1024> init_elem;
    Constant *init = nullptr;

    if (gv != nullptr) {
      if (auto init_old = dyn_cast<ConstantArray>(gv->getInitializer())) {
        unsigned int i = 0;
        while (auto elem = init_old->getAggregateElement(i++)) {
          init_elem.push_back(elem);
        }
      }
    
      gv->eraseFromParent();
      gv = nullptr;
    }
    


    // append created function to global ctor
    
    Constant *gv_new[] = {
      ConstantInt::get(Int32Ty, priority, false),
      ConstantExpr::getBitCast(func, CtorPFTy),
      Constant::getNullValue(Int8PtrTy)
    };
    init_elem.push_back(ConstantStruct::get(CtorStructTy, gv_new));
    
    
    ArrayRef<Constant *> init_elem_arr(init_elem.begin(), init_elem.end());
    ArrayType *cdtor_arr_ty = ArrayType::get(CtorStructTy, init_elem.size());
    
    new GlobalVariable(M, cdtor_arr_ty, false,
                       GlobalValue::AppendingLinkage,
                       ConstantArray::get(cdtor_arr_ty, init_elem_arr),
                       gv_name);

    return true;
  }


  

  bool getKernelDebugData(Module &M, const std::string kernelName) {

    std::string varname = getSymbolNameForKernel(kernelName, SYMBOL_DATA_VAR);
    std::string funcname = getSymbolNameForKernel(kernelName, SYMBOL_DATA_FUNC);

    // add function
    LLVMContext &ctx = M.getContext();
    Function* func = dyn_cast<Function>(M.getOrInsertFunction(
                                          funcname.c_str(),
                                          Type::getVoidTy(ctx), NULL) );
    if (!func || !func->empty()) return false;
    func->setCallingConv(CallingConv::C);


    
    // types
    Type *Int8Ty = Type::getInt8Ty(ctx);
    Type *Int32Ty = Type::getInt32Ty(ctx);
    Type *Int8PtrTy = Type::getInt8PtrTy(ctx);
    


    // add entry block for function
    BasicBlock* block = BasicBlock::Create(ctx, "entry", func);
    IRBuilder<> IRB(block);

    GlobalVariable *var_link = M.getNamedGlobal(varname);
    if (var_link == nullptr) {
      var_link = new GlobalVariable(M, Int8PtrTy, false,
                                    GlobalValue::LinkOnceAnyLinkage,
                                    Constant::getNullValue(Int8PtrTy),
                                    varname.c_str());
    }
    Value *var_sym = IRB.CreatePointerCast(var_link, Int8PtrTy);

    
    Value *varlen_alloca = IRB.CreateAlloca(SizeTy);
    Value *varlen_voidptr = IRB.CreatePointerCast(varlen_alloca, Int8PtrTy);
    IRB.CreateStore(ConstantInt::get(SizeTy, 0), varlen_alloca);
    IRB.CreateCall(CudaGetSymbolSize, {varlen_voidptr, var_sym});
    Value* varlen_val = IRB.CreateLoad(SizeTy, varlen_alloca);


    
    Value *var_alloca = IRB.CreateAlloca(Int8Ty, varlen_val);
    IRB.CreateCall(CudaMemcpyFromSymbol, {var_alloca, var_sym,
      varlen_val,
      ConstantInt::get(SizeTy, 0),
      ConstantInt::get(CudaMemcpyKindTy,
      cudaMemcpyDeviceToHost)});
                  
    IRB.CreateCall(AccdatAppend, {var_alloca, varlen_val});
    IRB.CreateRetVoid();



    // append created function to global ctor
    appendGlobalCtorDtor(M, func, GLOBAL_CTOR, GLOBAL_CDTOR_DEFAULT_PRIORITY + 1);
    
    
    return true;
  }

  
  void RegisterVars(Function *CudaSetup, ArrayRef<GlobalVariable*> Variables) {
    Module &M = *CudaSetup->getParent();
    IRBuilder<> IRB(M.getContext());

    IRB.SetInsertPoint(&CudaSetup->back().back());

    /** Get declaration of __cudaRegisterVar.
     * Protype:
     *  extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle,
     *   char  *hostVar, char  *deviceAddress, const char  *deviceName,
     *   int ext, int size, int constant, int global);
     */
    // no void*/* in llvm, we use i8*/* instead
    Type* voidPtrPtrTy = IRB.getInt8Ty()->getPointerTo()->getPointerTo();
    Type* charPtrTy = IRB.getInt8Ty()->getPointerTo();
    Type* intTy = IRB.getInt32Ty();
    auto *FnTy = FunctionType::get(intTy, {voidPtrPtrTy,
                                           charPtrTy, charPtrTy, charPtrTy,
                                           intTy, intTy, intTy, intTy}, false);
    auto *Fn = M.getOrInsertFunction("__cudaRegisterVar", FnTy);
    assert(Fn != nullptr);

    for (auto *Global : Variables) {

      auto *GlobalNameLiteral = IRB.CreateGlobalString(Global->getName());
      auto *GlobalName = IRB.CreateBitCast(GlobalNameLiteral, charPtrTy);
      auto *GlobalAddress = IRB.CreateBitCast(Global, charPtrTy);
      uint64_t GlobalSize = M.getDataLayout().getTypeStoreSize(Global->getType());
      Value *CubinHandle = &*CudaSetup->arg_begin();

      //createPrintf(IRB, "registering... symbol name: %s, symbol address: %p, name address: %p\n",
      //    {GlobalName, GlobalAddress, GlobalName});
      //errs() << "registering device symbol " << Global->getName().str() << "\n";

      IRB.CreateCall(Fn, {CubinHandle, GlobalAddress, GlobalName, GlobalName,
                          IRB.getInt32(0), IRB.getInt32(GlobalSize), IRB.getInt32(0), IRB.getInt32(0)});
    }
  }



  static bool setStrComparator(const std::string& lhs,
                               const std::string& rhs) {
    return lhs.compare(rhs) != 0;
  }
  
  typedef std::set <std::string,
                    bool(*)(const std::string& lhs,
                            const std::string& rhs)> kernelListSet;

  void createAndRegisterTraceVars(Function* CudaSetup, Type* VarType,
                                  kernelListSet kernel_list) {
    Module &M = *CudaSetup->getParent();
    //SmallVector<Function*, 8> registeredKernels;
    SmallVector<GlobalVariable*, 1024> globalVars;


    //globalVars.push_back(testGlobal(M));

  
    // do works on kernels for access data info
    for (auto kernel_name = kernel_list.cbegin();
         kernel_name != kernel_list.cend();
         ++kernel_name) {
      getKernelDebugData(M, *kernel_name);

      std::string accdat_var_str = getSymbolNameForKernel(*kernel_name, SYMBOL_DATA_VAR);
      if (GlobalVariable *accdat_var_gv = M.getNamedGlobal(accdat_var_str))
        globalVars.push_back(accdat_var_gv);
    }


  
    for (Instruction &inst : instructions(CudaSetup)) {
      auto *call = dyn_cast<CallInst>(&inst);
      if (call == nullptr) {
        continue;
      }
      auto *callee = call->getCalledFunction();
      if (!callee || callee->getName() != "__cudaRegisterFunction") {
        continue;
      }

      // 0: ptx image, 1: wrapper, 2: name, 3: name again, 4+: ?
      auto *wrapperVal = call->getOperand(1)->stripPointerCasts();
      assert(wrapperVal != nullptr && "__cudaRegisterFunction called without wrapper");
      auto *wrapper = dyn_cast<Function>(wrapperVal);
      assert(wrapper != nullptr && "__cudaRegisterFunction called with something other than a wrapper");

      StringRef kernelName = wrapper->getName();
      std::string varName = getSymbolNameForKernel(kernelName);
      GlobalVariable *globalVar = getOrCreateGlobalVar(M, VarType, varName);
      globalVars.push_back(globalVar);
    }

    RegisterVars(CudaSetup, globalVars);
  }

  


  bool runOnModule(Module &M) override {

    LLVMContext &ctx = M.getContext();
      
    bool isCUDA = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (isCUDA) return false;

      
    // type init
    traceInfoTy = getTraceInfoType(M.getContext());
    SizeTy = Type::getIntNTy(ctx, sizeof(size_t) * 8);
    SizePtrTy = Type::getIntNPtrTy(ctx, sizeof(size_t) * 8);
    CudaMemcpyKindTy = Type::getIntNTy(ctx, sizeof(cudaMemcpyKind) * 8);
    CudaErrorTy = Type::getIntNTy(ctx, sizeof(cudaError_t) * 8);


    //setFuncbaseAttr(M);
    findOrInsertRuntimeFunctions(M);

    

    kernelListSet kernel_list(&setStrComparator);

    // patch calls && collect kernels called
    for (auto &kcall : getAnalysis<LocateKCallsPass>().getLaunches()) {
      patchKernelCall(kcall.configureCall, kcall.kernelLaunch, kcall.kernelName);

      kernel_list.insert(StringRef(kcall.kernelName).str());
    }

    Function* CudaSetup = M.getFunction("__cuda_register_globals");
    
    
    // register global variables for trace info for all kernels registered
    // in this module
    appendGlobalCtorDtor(M, dyn_cast<Function>(AccdatCtor), GLOBAL_CTOR);
    appendGlobalCtorDtor(M, dyn_cast<Function>(AccdatDtor), GLOBAL_DTOR);
    
    if (CudaSetup != nullptr) {
      createAndRegisterTraceVars(CudaSetup, traceInfoTy, kernel_list);
    }

    
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.addRequired<LocateKCallsPass>();
  }

};

char InstrumentHost::ID = 0;

namespace llvm {
  Pass *createInstrumentHostPass() {
    return new InstrumentHost();
  }
}

static RegisterPass<InstrumentHost> X("memtrace-host", "inserts host-side instrumentation for mem-traces", false, false);
