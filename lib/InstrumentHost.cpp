#include <set>
#include <iostream>
#include <cuda_runtime_api.h>


#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/DebugInfoMetadata.h"


#include "common.h"
#include "PassCommon.h"
#include "Passes.h"
#include "LocateKCalls.h"

#include "compat/LLVM-8.h" // for backward compatibility

#define DEBUG_TYPE "cuprof-host"




using namespace llvm;

namespace cuprof {

  
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
    Type* ip_ty = nullptr;
    Type* i8p_ty = nullptr;
    Type* i32p_ty = nullptr;
    Type* i64p_ty = nullptr;
    Type* i_ty = nullptr;
    Type* i8_ty = nullptr;
    Type* i16_ty = nullptr;
    Type* i32_ty = nullptr;
    Type* i64_ty = nullptr;

    FunctionCallee cuprof_init = nullptr;
    FunctionCallee cuprof_term = nullptr;
    FunctionCallee cuprof_gvsym_set_up = nullptr;
    FunctionCallee cuprof_kernel_set_up = nullptr;
    FunctionCallee trc_start = nullptr;
    FunctionCallee trc_stop = nullptr;
    FunctionCallee cuda_get_device = nullptr;
    FunctionCallee cuda_memcpy_to_symbol = nullptr;
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
      ip_ty = Type::getIntNPtrTy(ctx, sizeof(int) * 8);
      i8p_ty = Type::getInt8PtrTy(ctx);
      i32p_ty = Type::getInt32PtrTy(ctx);
      i64p_ty = Type::getInt64PtrTy(ctx);
      i_ty = Type::getIntNTy(ctx, sizeof(int) * 8);
      i8_ty = Type::getInt8Ty(ctx);
      i16_ty = Type::getInt16Ty(ctx);
      i32_ty = Type::getInt32Ty(ctx);
      i64_ty = Type::getInt64Ty(ctx);
    }

  
    void findOrInsertRuntimeFunctions(Module& module) {
    
      cuprof_init =
        module.getOrInsertFunction("___cuprof_init",
                                   void_ty);
      cuprof_term =
        module.getOrInsertFunction("___cuprof_term",
                                   void_ty);
      cuprof_gvsym_set_up =
        module.getOrInsertFunction("___cuprof_gvsym_set_up",
                                   void_ty, i8p_ty);
      cuprof_kernel_set_up =
        module.getOrInsertFunction("___cuprof_kernel_set_up",
                                   void_ty, i8p_ty, i8p_ty);

      trc_start =
        module.getOrInsertFunction("___cuprof_trace_start",
                                   void_ty, i_ty, i8p_ty, i8p_ty, i64_ty, i16_ty);
      trc_stop =
        module.getOrInsertFunction("___cuprof_trace_stop",
                                   void_ty, i_ty, i8p_ty);
    
      cuda_get_device =
        module.getOrInsertFunction("cudaGetDevice",
                                   cuda_err_ty, ip_ty);
      cuda_memcpy_to_symbol =
        module.getOrInsertFunction("cudaMemcpyToSymbol",
                                   cuda_err_ty, i8p_ty, i8p_ty,
                                   size_ty, size_ty, cuda_memcpy_kind_ty);
      cuda_memcpy_from_symbol =
        module.getOrInsertFunction("cudaMemcpyFromSymbol",
                                   cuda_err_ty, i8p_ty, i8p_ty,
                                   size_ty, size_ty, cuda_memcpy_kind_ty);
      cuda_get_symbol_size =
        module.getOrInsertFunction("cudaGetSymbolSize",
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

    ///////////////////////////////////////////////////////////////
    bool registerFuncToGetDebugData(Module& module, const std::string kernel_name) {

      std::string varname_kdata = getSymbolName(kernel_name, CUPROF_SYMBOL_DATA_VAR);
      std::string varname_kid = getSymbolName(kernel_name, CUPROF_SYMBOL_KERNEL_ID);
      std::string funcname = getSymbolName(kernel_name, CUPROF_SYMBOL_DATA_FUNC);

      
      // get symbols for arguments of a function to be called
      GlobalVariable* gv_kdata =
        getOrInsertGlobalVar(module, i8p_ty, varname_kdata.c_str());
      //Value* kdata_sym = irb.CreatePointerCast(gv_kdata, i8p_ty);
      GlobalVariable* gv_kid =
        getOrInsertGlobalVar(module, i8p_ty, varname_kid.c_str());
      //Value* kid_sym = irb.CreatePointerCast(gv_kid, i8p_ty);

      registerFuncToGlobalCtor(module, cuprof_kernel_set_up, {gv_kdata, gv_kid}, funcname);

      return true;
      /*
      // add a function
      LLVMContext& ctx = module.getContext();
      FunctionCallee func_callee = module.getOrInsertFunction(funcname.c_str(), void_ty);
      Function* func = dyn_cast<Function>(func_callee.getCallee());
      if (!func || !func->empty()) return false;
      func->setCallingConv(CallingConv::C);
    


      // add entry block for the function
      BasicBlock* block = BasicBlock::Create(ctx, "entry", func);
      IRBuilder<> irb(block);


      
      //Value* varlen_alloca = irb.CreateAlloca(size_ty);
      //Value* varlen_voidptr = irb.CreatePointerCast(varlen_alloca, i8p_ty);
      //irb.CreateStore(ConstantInt::get(size_ty, 0), varlen_alloca);
    
      //Value* cuda_get_symbol_size_args[] = {varlen_voidptr, var_sym};
      //irb.CreateCall(cuda_get_symbol_size, cuda_get_symbol_size_args);
    
      //Value* varlen_val = irb.CreateLoad(size_ty, varlen_alloca);


    
      //Value* var_alloca = irb.CreateAlloca(i8_ty, varlen_val);
      //Value* cuda_memcpy_from_symbol_args[] = {
      //  var_alloca, var_sym,
      //  varlen_val,
      //  ConstantInt::get(size_ty, 0),
      //  ConstantInt::get(cuda_memcpy_kind_ty, cudaMemcpyDeviceToHost)
      //};
      //irb.CreateCall(cuda_memcpy_from_symbol, cuda_memcpy_from_symbol_args);

      Value* cuprof_kernel_set_up_args[] = {kdata_sym, kid_sym};
      irb.CreateCall(cuprof_kernel_set_up, cuprof_kernel_set_up_args);
    
      irb.CreateRetVoid();



      // append created function to global ctor
      appendGlobalCtorDtor(module, func, GLOBAL_CTOR, GLOBAL_CDTOR_DEFAULT_PRIORITY + 1);
    
      
      return true;
      */
    }

    
    bool registerFuncToGlobalCtor(Module& module,
                                  FunctionCallee func_callee, ArrayRef<Value*> args,
                                  const Twine& ctor_name) {

      // add a function
      LLVMContext& ctx = module.getContext();
      
      FunctionCallee ctor_callee =
        module.getOrInsertFunction(ctor_name.getSingleStringRef(), void_ty);
      Function* ctor = dyn_cast<Function>(ctor_callee.getCallee());
      if (!ctor || !ctor->empty()) return false;
      ctor->setCallingConv(CallingConv::C);


      
      // add entry block for the function
      BasicBlock* block = BasicBlock::Create(ctx, "entry", ctor);
      IRBuilder<> irb(block);

      
      FunctionType* func_callee_ty = func_callee.getFunctionType();
      int args_count = func_callee_ty->getNumParams();
      std::vector<Value*> args_casted;
      for (int i = 0; i < args_count; i++) {
        args_casted.push_back(
          irb.CreateBitOrPointerCast(args[i], func_callee_ty->getParamType(i))
          );
      }
      
      
      irb.CreateCall(func_callee, args_casted);
      irb.CreateRetVoid();

      
      // append created function to global ctor
      appendGlobalCtorDtor(module, ctor, GLOBAL_CTOR, GLOBAL_CDTOR_DEFAULT_PRIORITY + 1);
    
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


    void createAndRegisterTraceVars(Function* cuda_setup_func,
                                    SmallVector<Function*, 32> kernel_list) {
      Module& module = *cuda_setup_func->getParent();
      //SmallVector<Function*, 8> registeredKernels;
      SmallVector<GlobalVariable*, 32> gvs;

  
      // push metadata for each kernel to the list
      for (SmallVector<Function*, 32>::iterator kernel = kernel_list.begin();
           kernel != kernel_list.end();
           ++kernel) {
        StringRef kernel_name_ref = (*kernel)->getName();
        std::string kernel_name = kernel_name_ref.str();
        registerFuncToGetDebugData(module, kernel_name); /////////////////////////

        std::string kdata_name = getSymbolName(kernel_name,
                                               CUPROF_SYMBOL_DATA_VAR);
        if (GlobalVariable* gv_kdata = module.getNamedGlobal(kdata_name)) {
          gvs.push_back(gv_kdata);
        }
        std::string kid_name = getSymbolName(kernel_name,
                                             CUPROF_SYMBOL_KERNEL_ID);
        if (GlobalVariable* gv_kid = module.getNamedGlobal(kid_name)) {
          gvs.push_back(gv_kid);
        }
      }

      // push the traceinfo var to the list
      GlobalVariable *gv = getOrInsertGlobalVar(module, trace_info_ty,
                                                CUPROF_TRACE_BASE_INFO);
      
      std::string symbol_name = getSymbolName(module.getModuleIdentifier(),
                                              CUPROF_SYMBOL_BASE_NAME);
      registerFuncToGlobalCtor(module, cuprof_gvsym_set_up, {gv}, symbol_name);
      gvs.push_back(gv);

      // register all items in the list
      registerVars(cuda_setup_func, gvs);
    }


  
/***********************
 * Module IR Insertion *
 ***********************/


    void patchKernelCall(CallInst* configure_call, Instruction* launch,
                         const StringRef kernel_name) {
      assert(configure_call->getNumArgOperands() == 6);
      
      IRBuilder<> irb(configure_call->getNextNode());

      
      Value* device_num_alloc = irb.CreateAlloca(i_ty);
      Value* cuda_get_device_args[] = {device_num_alloc};
      irb.CreateCall(cuda_get_device, cuda_get_device_args);
      Value* device_num = irb.CreateLoad(i_ty, device_num_alloc);

      Value* stream = configure_call->getArgOperand(5);
      Value* stream_ptr = irb.CreateBitCast(stream, i8p_ty);
      Value* kernel_name_val = irb.CreateGlobalStringPtr(kernel_name);


      // Thread block size of the current kernel call
      
      Instruction::CastOps castop_i32_i64 = CastInst::getCastOpcode(
        Constant::getNullValue(i32_ty), false, i64_ty, false);
      Instruction::CastOps castop_i64_i16 = CastInst::getCastOpcode(
        Constant::getNullValue(i64_ty), false, i16_ty, false);
      Instruction::CastOps castop_i32_i16 = CastInst::getCastOpcode(
        Constant::getNullValue(i32_ty), false, i16_ty, false);

      Value* grid_dim = configure_call->getArgOperand(0);   // <32-bit: y> <32-bit: x>
      Value* grid_dim_z = configure_call->getArgOperand(1); // <32-bit: z>
      grid_dim_z = irb.CreateShl(
        irb.CreateCast(castop_i32_i64, grid_dim_z, i64_ty),
        48
        ); // z: shift 48 left
      grid_dim = irb.CreateOr(
        irb.CreateAnd(grid_dim, 0xFFFFFFFFFFFF),
        grid_dim_z
        ); // <16-bit: z> <16-bit: y> <32-bit: x>

      
      Value* cta_size = configure_call->getArgOperand(2);   // <32-bit: y> <32-bit: x>
      Value* cta_size_z = configure_call->getArgOperand(3); // <32-bit: z>
      cta_size = irb.CreateMul(
        irb.CreateAnd(cta_size, 0xFFFFFFFF),
        irb.CreateLShr(cta_size, 32)
        ); // x * y
      cta_size = irb.CreateMul(
        irb.CreateCast(castop_i64_i16, cta_size, i16_ty),
        irb.CreateCast(castop_i32_i16, cta_size_z, i16_ty)
        ); // <uint16_t>(x*y) * <uint16_t>z

    

      Value* trace_start_args[] = {device_num, stream_ptr, kernel_name_val, grid_dim, cta_size};
      irb.CreateCall(trc_start, trace_start_args);

      irb.SetInsertPoint(launch->getNextNode());
    
      Value* trace_stop_args[] = {device_num, stream_ptr};
      irb.CreateCall(trc_stop, trace_stop_args);
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
      //bool kernel_filtering = (args.kernel.size() != 0);



      // patch calls
      /*
        for (KCall& kcall : getAnalysis<LocateKCallsPass>().getLaunchList()) {
      
        // kernel filtering
        if (kernel_filtering && !isKernelToBeTraced(kcall.kernel_obj, args.kernel))
        continue;

      
        // patch kernel call
        patchKernelCall(kcall.configure_call,
        kcall.kernel_launch,
        kcall.kernel_obj->getName());
        }
      */

    
      // register global variables of trace info for all kernels registered in this module
      appendGlobalCtorDtor(module,
                           dyn_cast<Function>(cuprof_init.getCallee()),
                           GLOBAL_CTOR);
      appendGlobalCtorDtor(module,
                           dyn_cast<Function>(cuprof_term.getCallee()),
                           GLOBAL_DTOR);
      
      //GlobalVariable *gv = getOrInsertGlobalVar(module, trace_info_ty,
      //                                          CUPROF_TRACE_BASE_INFO);
      //registerFuncToGlobalCtor(module, cuprof_gvsym_set_up, {gv}, "___cuprof_base_name");


    
      // get all kernels
      SmallVector<Function*, 32> kernel_list =
        getAnalysis<LocateKCallsPass>().getKernelList();
    
      Function* cuda_setup_func = module.getFunction("__cuda_register_globals");
      if (cuda_setup_func != nullptr) {
        createAndRegisterTraceVars(cuda_setup_func, kernel_list);
      }

    
      return true;
    }

    void getAnalysisUsage(AnalysisUsage& usage) const override {
      usage.addRequired<LocateKCallsPass>();
    }

  };

  char InstrumentHostPass::ID = 0;



  
  
  Pass *createInstrumentHostPass(InstrumentPassArg args) {
    return new InstrumentHostPass(args);
  }

  static RegisterPass<InstrumentHostPass>
  X("cuprof-host",
    "inserts host-side instrumentation for cuprof",
    false, false);

}
