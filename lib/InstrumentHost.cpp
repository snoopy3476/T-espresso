#include "Passes.h"

#include <set>
#include <cuda_runtime_api.h>


#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/DebugInfoMetadata.h"


#include "LocateKCalls.h"

#define INCLUDE_LLVM_CUPROF_TRACE_STUFF
#include "Common.h"

#define DEBUG_TYPE "cuprof-host"

using namespace llvm;


struct InstrumentHostPass : public ModulePass {
  static char ID;
  InstrumentPassArg args;
  
  InstrumentHostPass(InstrumentPassArg passargs = args_default)
    : ModulePass(ID), args(passargs) {}

  
  
/*****************
 * Pass Variable *
 *****************/

  
  Type* trace_info_ty = nullptr;
  Type* size_ty = nullptr;
  Type* sizep_ty = nullptr;
  Type* cuda_memcpy_kind_ty = nullptr;
  Type* cuda_err_ty = nullptr;

  Type* void_ty = nullptr;
  Type* i8p_ty = nullptr;
  Type* i32p_ty = nullptr;
  Type* i64p_ty = nullptr;
  Type* i8_ty = nullptr;
  Type* i16_ty = nullptr;
  Type* i32_ty = nullptr;
  Type* i64_ty = nullptr;

  FunctionCallee accdat_ctor = nullptr;
  FunctionCallee accdat_dtor = nullptr;
  FunctionCallee accdat_append = nullptr;
  FunctionCallee trc_fill_info = nullptr;
  FunctionCallee trc_copy_to_symbol = nullptr;
  FunctionCallee trc_touch = nullptr;
  FunctionCallee trc_start = nullptr;
  FunctionCallee trc_stop = nullptr;
  FunctionCallee cuda_memcpy_from_symbol = nullptr;
  FunctionCallee cuda_get_symbol_size = nullptr;


  
/***********************
 * Pass Initialization *
 ***********************/

  
  void initTypes(Module& module) {
    LLVMContext& ctx = module.getContext();
    
    trace_info_ty = getTraceInfoType(ctx);
    size_ty = Type::getIntNTy(ctx, sizeof(size_t) * 8);
    sizep_ty = Type::getIntNPtrTy(ctx, sizeof(size_t) * 8);
    cuda_memcpy_kind_ty = Type::getIntNTy(ctx, sizeof(cudaMemcpyKind) * 8);
    cuda_err_ty = Type::getIntNTy(ctx, sizeof(cudaError_t) * 8);

    void_ty = Type::getVoidTy(ctx);
    i8p_ty = Type::getInt8PtrTy(ctx);
    i32p_ty = Type::getInt32PtrTy(ctx);
    i64p_ty = Type::getInt64PtrTy(ctx);
    i8_ty = Type::getInt8Ty(ctx);
    i16_ty = Type::getInt16Ty(ctx);
    i32_ty = Type::getInt32Ty(ctx);
    i64_ty = Type::getInt64Ty(ctx);
  }

  
  void findOrInsertRuntimeFunctions(Module& module) {
    
    accdat_ctor = module.getOrInsertFunction("___cuprof_accdat_ctor",
                                             void_ty);
    accdat_dtor = module.getOrInsertFunction("___cuprof_accdat_dtor",
                                             void_ty);
    accdat_append = module.getOrInsertFunction("___cuprof_accdat_append",
                                               void_ty, i8p_ty, i64_ty);

    
    trc_fill_info = module.getOrInsertFunction("__trace_fill_info",
                                               void_ty, i8p_ty, i8p_ty);
    trc_copy_to_symbol = module.getOrInsertFunction("__trace_copy_to_symbol",
                                                    void_ty, i8p_ty, i8p_ty, i8p_ty);
    trc_touch = module.getOrInsertFunction("__trace_touch",
                                           void_ty, i8p_ty);
    trc_start = module.getOrInsertFunction("__trace_start",
                                           void_ty, i8p_ty, i8p_ty, i16_ty);
    trc_stop = module.getOrInsertFunction("__trace_stop",
                                          void_ty, i8p_ty);
    
    cuda_memcpy_from_symbol = module.getOrInsertFunction("cudaMemcpyFromSymbol",
                                                         cuda_err_ty, i8p_ty, i8p_ty,
                                                         size_ty, size_ty, cuda_memcpy_kind_ty);
    cuda_get_symbol_size = module.getOrInsertFunction("cudaGetSymbolSize",
                                                      cuda_err_ty, i8p_ty, i8p_ty);
  }



  
/*************************
 * Module Initialization *
 *************************/


  GlobalVariable* getOrInsertGlobalVar(Module& module, Type* ty, const Twine& name) {
    
    // first see if the variable already exists
  
    GlobalVariable* gv = module.getGlobalVariable(name.getSingleStringRef());
    if (gv) {
      return gv;
    }

    // If variable does not exist, create one and register it.
    // This happens if a kernel is called in a module it is not registered in.
    Constant* zero = Constant::getNullValue(ty);
    gv = new GlobalVariable(module, ty, false, GlobalValue::LinkOnceAnyLinkage, zero, name);
    gv->setAlignment(8);
    assert(gv != nullptr);
    return gv;
  }
  
  
#define GLOBAL_CTOR_ARR_NAME "llvm.global_ctors"
#define GLOBAL_DTOR_ARR_NAME "llvm.global_dtors"
#define GLOBAL_CDTOR_DEFAULT_PRIORITY ((uint32_t)2147483646)
  
  enum GlobalCdtorType {
    GLOBAL_CTOR = 0,
    GLOBAL_DTOR = 1
  };

  bool appendGlobalCtorDtor(Module& module, Function *func, GlobalCdtorType type,
                            uint32_t priority = GLOBAL_CDTOR_DEFAULT_PRIORITY) {
    if (!func) return false;

    FunctionType* ctor_fty = FunctionType::get(void_ty, false);
    Type* ctorp_fty = PointerType::getUnqual(ctor_fty);
    StructType* ctor_sty = StructType::get(
      (Type*) i32_ty, (Type*) PointerType::getUnqual(ctor_fty), (Type*) i8p_ty);

    
    const char* gv_name;
    switch (type) {
      
    case GLOBAL_DTOR: // dtor
      gv_name = GLOBAL_DTOR_ARR_NAME;
      break;

    default: // ctor
      gv_name = GLOBAL_CTOR_ARR_NAME;
      break;
    }

    
    GlobalVariable* gv = module.getNamedGlobal(gv_name);
    SmallVector<Constant *, 1024> init_elem;

    if (gv != nullptr) {
      if (ConstantArray* init_old = dyn_cast<ConstantArray>(gv->getInitializer())) {
        unsigned int i = 0;
        while (Constant* elem = init_old->getAggregateElement(i++)) {
          init_elem.push_back(elem);
        }
      }
    
      gv->eraseFromParent();
      gv = nullptr;
    }
    


    // append created function to global ctor
    
    Constant* gv_new[] = {
      ConstantInt::get(i32_ty, priority, false),
      ConstantExpr::getBitCast(func, ctorp_fty),
      Constant::getNullValue(i8p_ty)
    };
    init_elem.push_back(ConstantStruct::get(ctor_sty, gv_new));
    
    
    ArrayRef<Constant *> init_elem_arr(init_elem.begin(), init_elem.end());
    ArrayType* cdtor_arr_ty = ArrayType::get(ctor_sty, init_elem.size());
    
    new GlobalVariable(module, cdtor_arr_ty, false,
                       GlobalValue::AppendingLinkage,
                       ConstantArray::get(cdtor_arr_ty, init_elem_arr),
                       gv_name);

    return true;
  }

  
  bool registerFuncToGetDebugData(Module& module, const std::string kernel_name) {

    std::string varname = getSymbolNameForKernel(kernel_name, SYMBOL_DATA_VAR);
    std::string funcname = getSymbolNameForKernel(kernel_name, SYMBOL_DATA_FUNC);

    // add function
    LLVMContext& ctx = module.getContext();
    Function* func = dyn_cast<Function>(module.getOrInsertFunction(
                                          funcname.c_str(),
                                          Type::getVoidTy(ctx)).getCallee() );
    if (!func || !func->empty()) return false;
    func->setCallingConv(CallingConv::C);
    


    // add entry block for function
    BasicBlock* block = BasicBlock::Create(ctx, "entry", func);
    IRBuilder<> irb(block);

    GlobalVariable* var_link = module.getNamedGlobal(varname);
    if (var_link == nullptr) {
      var_link = new GlobalVariable(module, i8p_ty, false,
                                    GlobalValue::LinkOnceAnyLinkage,
                                    Constant::getNullValue(i8p_ty),
                                    varname.c_str());
    }
    Value* var_sym = irb.CreatePointerCast(var_link, i8p_ty);

    
    Value* varlen_alloca = irb.CreateAlloca(size_ty);
    Value* varlen_voidptr = irb.CreatePointerCast(varlen_alloca, i8p_ty);
    irb.CreateStore(ConstantInt::get(size_ty, 0), varlen_alloca);
    
    Value* cuda_get_symbol_size_args[] = {varlen_voidptr, var_sym};
    irb.CreateCall(cuda_get_symbol_size, cuda_get_symbol_size_args);
    
    Value* varlen_val = irb.CreateLoad(size_ty, varlen_alloca);


    
    Value* var_alloca = irb.CreateAlloca(i8_ty, varlen_val);
    Value* cuda_memcpy_from_symbol_args[] = {
      var_alloca, var_sym,
      varlen_val,
      ConstantInt::get(size_ty, 0),
      ConstantInt::get(cuda_memcpy_kind_ty, cudaMemcpyDeviceToHost)
    };
    irb.CreateCall(cuda_memcpy_from_symbol, cuda_memcpy_from_symbol_args);

    Value* accdat_append_args[] = {var_alloca, varlen_val};
    irb.CreateCall(accdat_append, accdat_append_args);
    
    irb.CreateRetVoid();



    // append created function to global ctor
    appendGlobalCtorDtor(module, func, GLOBAL_CTOR, GLOBAL_CDTOR_DEFAULT_PRIORITY + 1);
    
    
    return true;
  }

  
  void registerVars(Function* cuda_setup_func, ArrayRef<GlobalVariable*> vars) {
    Module& module = *cuda_setup_func->getParent();
    LLVMContext& ctx = module.getContext();
    IRBuilder<> irb(module.getContext());

    irb.SetInsertPoint(&cuda_setup_func->back().back());

    /** Get declaration of __cudaRegisterVar.
     * Protype:
     *  extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle,
     *   char  *hostVar, char  *deviceAddress, const char  *deviceName,
     *   int ext, int size, int constant, int global);
     */
    // no void*/* in llvm, we use i8*/* instead
    Type* voidpp_ty = i8p_ty->getPointerTo();
    Type* charp_ty = i8p_ty;
    Type* int_ty = IntegerType::get(ctx, sizeof(int)*8);
    Type* fnty_arg[] = {voidpp_ty,
                        charp_ty, charp_ty, charp_ty,
                        int_ty, int_ty, int_ty, int_ty};
    FunctionType* fty = FunctionType::get(int_ty, fnty_arg, false);
    FunctionCallee func = module.getOrInsertFunction("__cudaRegisterVar", fty);
    assert(func.getCallee() != nullptr);

    for (GlobalVariable* gv : vars) {

      Value* gv_name_literal = irb.CreateGlobalString(gv->getName());
      Value* gv_name = irb.CreateBitCast(gv_name_literal, charp_ty);
      Value* gv_addr = irb.CreateBitCast(gv, charp_ty);
      uint64_t gv_size = module.getDataLayout().getTypeStoreSize(gv->getType());
      Value* cubin_handle = &*cuda_setup_func->arg_begin();

      Value* fn_args[] = {cubin_handle, gv_addr, gv_name, gv_name,
                          irb.getInt32(0), irb.getInt32(gv_size), irb.getInt32(0), irb.getInt32(0)};
      irb.CreateCall(func, fn_args);
    }
  }


  void createAndRegisterTraceVars(Function* cuda_setup_func, Type* ty,
                                  SmallVector<Function*, 32> kernel_list) {
    Module& module = *cuda_setup_func->getParent();
    //SmallVector<Function*, 8> registeredKernels;
    SmallVector<GlobalVariable*, 32> gvs;

  
    // do works on kernels for access data info
    for (SmallVector<Function*, 32>::iterator kernel = kernel_list.begin();
         kernel != kernel_list.end();
         ++kernel) {
      StringRef kernel_name = (*kernel)->getName();
      registerFuncToGetDebugData(module, kernel_name.str());

      std::string accdat_var_str = getSymbolNameForKernel(kernel_name.str(), SYMBOL_DATA_VAR);
      if (GlobalVariable* accdat_var_gv = module.getNamedGlobal(accdat_var_str)) {
        gvs.push_back(accdat_var_gv);
      }
    }

  
    for (Instruction& inst : instructions(cuda_setup_func)) {
      CallInst* callinst = dyn_cast<CallInst>(&inst);
      if (callinst == nullptr) {
        continue;
      }
      Function* callee = callinst->getCalledFunction();
      if (!callee || callee->getName() != "__cudaRegisterFunction") {
        continue;
      }

      // 0: ptx image, 1: wrapper, 2: name, 3: name again, 4+: ?
      Value* wrapper_val = callinst->getOperand(1)->stripPointerCasts();
      assert(wrapper_val != nullptr &&
             "__cudaRegisterFunction called without wrapper");
      Value* wrapper = dyn_cast<Function>(wrapper_val);
      assert(wrapper != nullptr &&
             "__cudaRegisterFunction called with something other than a wrapper");

      StringRef kernel_name = wrapper->getName();
      std::string var_name = getSymbolNameForKernel(kernel_name.str());
      GlobalVariable *gv = getOrInsertGlobalVar(module, ty, var_name);
      gvs.push_back(gv);
    }

    registerVars(cuda_setup_func, gvs);
  }


  
/***********************
 * Module IR Insertion *
 ***********************/


  void patchKernelCall(CallInst* configure_call, Instruction* launch,
                       const StringRef kernel_name) {
    assert(configure_call->getNumArgOperands() == 6);
    Value* stream = configure_call->getArgOperand(5);

    IRBuilder<> irb(configure_call->getNextNode());

    Value* kernel_name_val = irb.CreateGlobalStringPtr(kernel_name);

    // try adding in global symbol + cuda registration
    Module& module = *configure_call->getModule();

    Type* gv_ty = trace_info_ty;
    std::string kernel_symbol_name = getSymbolNameForKernel(kernel_name.str());

    GlobalVariable* gv = getOrInsertGlobalVar(module, gv_ty, kernel_symbol_name);

    Value* gv_ptr = irb.CreateBitCast(gv, i8p_ty);
    Value* stream_ptr = irb.CreateBitCast(stream, i8p_ty);



    // Thread block size of the current kernel call
      
    Instruction::CastOps castop_i64_i16 = CastInst::getCastOpcode(
      Constant::getNullValue(i64_ty), false, i16_ty, false);
    Instruction::CastOps castop_i32_i16 = CastInst::getCastOpcode(
      Constant::getNullValue(i32_ty), false, i16_ty, false);
      
    Value* blockSize = configure_call->getArgOperand(2);   // <32-bit: y> <32-bit: x>
    Value* blockSize_z = configure_call->getArgOperand(3); // <32-bit: z>
    blockSize = irb.CreateMul(
      irb.CreateAnd(blockSize, 0xFFFFFFFF),
      irb.CreateLShr(blockSize, 32)
      ); // x * y
    blockSize = irb.CreateMul(
      irb.CreateCast(castop_i64_i16, blockSize, i16_ty),
      irb.CreateCast(castop_i32_i16, blockSize_z, i16_ty)
      ); // <uint16_t>(x*y) * <uint16_t>z

    

    Value* trace_touch_args[] = {stream_ptr};
    Value* trace_start_args[] = {stream_ptr, kernel_name_val, blockSize};
    irb.CreateCall(trc_touch, trace_touch_args);
    irb.CreateCall(trc_start, trace_start_args);

    const DataLayout& dat_layout = configure_call->getParent()->getParent()->getParent()->getDataLayout();

    size_t buflen = dat_layout.getTypeStoreSize(gv_ty);
    Value* buf_info = irb.CreateAlloca(i8_ty, irb.getInt32(buflen));
    
    Value* trace_fill_info_args[] = {buf_info, stream_ptr};
    irb.CreateCall(trc_fill_info, trace_fill_info_args);
    
    Value* trace_copy_to_symbol_args[] = {stream_ptr, gv_ptr, buf_info};
    irb.CreateCall(trc_copy_to_symbol, trace_copy_to_symbol_args);

    irb.SetInsertPoint(launch->getNextNode());
    
    Value* trace_stop_args[] = {stream_ptr};
    irb.CreateCall(trc_stop, trace_stop_args);
  }
  

  
/*******************
 * Module Analysis *
 *******************/

  
  bool isKernelToBeTraced(Function* kernel) {

    const std::string kernel_name_sym = kernel->getName().str();
    DISubprogram* kernel_debuginfo = kernel->getSubprogram();
    std::string kernel_name_orig;
    if (kernel_debuginfo) {
      kernel_name_orig = kernel_debuginfo->getName().str();
    }

    // stop instrumenting if not listed on enabled kernel
    if (std::find(args.kernel.begin(),
                  args.kernel.end(),
                  kernel_name_sym) == args.kernel.end() &&
        std::find(args.kernel.begin(),
                  args.kernel.end(),
                  kernel_name_orig) == args.kernel.end()) {
      return false;
    }
        
    //fprintf(stderr, "cuprof: Selective kernel tracing enabled (%s)\n",
    //        kernel_name_sym.c_str());

    return true;
  }
  


/**************
 * Pass Entry *
 **************/
  
  
  bool runOnModule(Module& module) override {

    bool is_cuda = module.getTargetTriple().find("nvptx") != std::string::npos;
    if (is_cuda) return false;

      
    // type / function call init
    initTypes(module);
    findOrInsertRuntimeFunctions(module);

    

    // if kernel args is set, kernel filtering is enabled
    bool kernel_filtering = (args.kernel.size() != 0);



    // patch calls
    for (KCall& kcall : getAnalysis<LocateKCallsPass>().getLaunchList()) {
      
      // kernel filtering
      if (kernel_filtering && !isKernelToBeTraced(kcall.kernel_obj))
        continue;

      
      // patch kernel call
      patchKernelCall(kcall.configure_call,
                      kcall.kernel_launch,
                      kcall.kernel_obj->getName());
    }

    
    // register global variables for trace info for all kernels registered
    // in this module
    appendGlobalCtorDtor(module, dyn_cast<Function>(accdat_ctor.getCallee()), GLOBAL_CTOR);
    appendGlobalCtorDtor(module, dyn_cast<Function>(accdat_dtor.getCallee()), GLOBAL_DTOR);


    
    // get all kernels
    SmallVector<Function*, 32> kernel_list =
      getAnalysis<LocateKCallsPass>().getKernelList();
    
    Function* cuda_setup_func = module.getFunction("__cuda_register_globals");
    if (cuda_setup_func != nullptr) {
      createAndRegisterTraceVars(cuda_setup_func, trace_info_ty, kernel_list);
    }

    
    return true;
  }

  void getAnalysisUsage(AnalysisUsage& usage) const override {
    usage.addRequired<LocateKCallsPass>();
  }

};

char InstrumentHostPass::ID = 0;

namespace llvm {
  Pass *createInstrumentHostPass(InstrumentPassArg args) {
    return new InstrumentHostPass(args);
  }
}

static RegisterPass<InstrumentHostPass> X("cuprof-host",
                                          "inserts host-side instrumentation for cuprof",
                                          false, false);
