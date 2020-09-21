#include <set>
#include <iostream>
#include <fstream>
#include <memory>

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "common.h"
#include "trace-io.h"
#include "PassCommon.h"
#include "Passes.h"

#include "compat/LLVM-8.h" // for backward compatibility

#define DEBUG_TYPE "cuprof-device"
#define TRACE_DEBUG_DATA "___cuprof_accdat_instmd"

#define ADDRESS_SPACE_GENERIC 0
#define ADDRESS_SPACE_GLOBAL 1
#define ADDRESS_SPACE_INTERNAL 2
#define ADDRESS_SPACE_SHARED 3
#define ADDRESS_SPACE_CONSTANT 4
#define ADDRESS_SPACE_LOCAL 5

#include "llvm/IR/IntrinsicInst.h"




using namespace llvm;

namespace cuprof {

  struct InstrumentDevicePass : public ModulePass {
    static char ID;
    InstrumentPassArg args;

    InstrumentDevicePass(InstrumentPassArg passargs = args_default)
      : ModulePass(ID), args(passargs) { }



  
/*****************
 * Pass Variable *
 *****************/

  
    struct TraceInfoValues {
      Value* alloc;
      Value* commit;
      Value* flushed;
      Value* signal;
      Value* records;
      Value* grid;
      Value* cta_serial;
      Value* warpv;
      Value* lane;
      Value* kernel;
      Value* filter_sm;
      Value* filter_warpp;
      Value* filter_sm_count;
      Value* filter_warpp_count;
    
      Value* to_be_traced;
    };

  
  
    Type* void_ty = nullptr;
    Type* i8p_ty = nullptr;
    Type* i32p_ty = nullptr;
    Type* i64p_ty = nullptr;
    Type* i8_ty = nullptr;
    Type* i16_ty = nullptr;
    Type* i32_ty = nullptr;
    Type* i64_ty = nullptr;
    Type* trace_base_info_ty = nullptr;
    Type* trace_info_ty = nullptr;
    Type* trace_info_pty = nullptr;
  
    FunctionType* i32_fty = nullptr;
    FunctionType* i64_fty = nullptr;

    FunctionCallee trace_call = nullptr;
    FunctionCallee trace_ret_call = nullptr;
    FunctionCallee filter_call = nullptr;
    FunctionCallee filter_volatile_call = nullptr;

  

/***********************
 * Pass Initialization *
 ***********************/

  
    void initTypes(Module& module) {
      LLVMContext& ctx = module.getContext();
    
      void_ty = Type::getVoidTy(ctx);
      i8p_ty = Type::getInt8PtrTy(ctx);
      i32p_ty = Type::getInt32PtrTy(ctx);
      i64p_ty = Type::getInt64PtrTy(ctx);
      i8_ty = Type::getInt8Ty(ctx);
      i16_ty = Type::getInt16Ty(ctx);
      i32_ty = Type::getInt32Ty(ctx);
      i64_ty = Type::getInt64Ty(ctx);
      i32_fty = FunctionType::get(i32_ty, false);
      i64_fty = FunctionType::get(i64_ty, false);
      
      //trace_base_info_ty = getTraceBaseInfoType(ctx);
      trace_info_ty = getTraceInfoType(ctx);
      trace_info_pty = trace_info_ty->getPointerTo();
    }
  
    void findOrInsertRuntimeFunctions(Module& module) {
    
      trace_call =
        module.getOrInsertFunction("___cuprof_trace", void_ty,
                                   i32p_ty, i32p_ty,
                                   i32p_ty, i32p_ty,
                                   i8p_ty, i64_ty,
                                   i64_ty, i64_ty,
                                   i32_ty, i32_ty,
                                   i32_ty, i32_ty,
                                   i32_ty, i32_ty,
                                   i8_ty);
      if (!trace_call.getCallee()) {
        report_fatal_error("No ___cuprof_trace declaration found");
      }
    
      trace_ret_call =
        module.getOrInsertFunction("___cuprof_trace_ret", void_ty,
                                   i32p_ty, i32p_ty,
                                   i32_ty);
      if (!trace_ret_call.getCallee()) {
        report_fatal_error("No ___cuprof_trace declaration found");
      }
    
      filter_call =
        module.getOrInsertFunction("___cuprof_filter", void_ty,
                                   i8p_ty, i64p_ty, i64p_ty, i32p_ty,
                                   i8_ty, i8_ty, i8_ty, i64_ty, i32_ty);
      if (!filter_call.getCallee()) {
        report_fatal_error("No ___cuprof_filter declaration found");
      }
    
      filter_volatile_call =
        module.getOrInsertFunction("___cuprof_filter_volatile", void_ty,
                                   i8p_ty, i32p_ty, i32p_ty, i8_ty,
                                   i8_ty, i32_ty, i32_ty);
      if (!filter_volatile_call.getCallee()) {
        report_fatal_error("No ___cuprof_filter_volatile declaration found");
      }
    }


  
/*************************
 * Module Initialization *
 *************************/

  
    template <typename T>
    GlobalVariable* getOrInsertGlobalArray(Module& module, std::vector<T> input,
                                           const char* name) {
    
      LLVMContext& ctx = module.getContext();

      GlobalVariable* gv = module.getNamedGlobal(name);
    
      if (!gv) {
        ArrayRef<T> dref = ArrayRef<T>(input.data(), input.size());
        Constant* init = ConstantDataArray::get(ctx, dref);
        gv = new GlobalVariable(module, init->getType(), true,
                                GlobalValue::InternalLinkage,
                                init, name);
      }

      return gv;
    }

  
    GlobalVariable* getOrInsertGlobalVariableExtern(Module& module, Type* ty,
                                                    const char* name) {
      GlobalVariable* gv = module.getNamedGlobal(name);

      if (!gv) {
        Constant* zero = Constant::getNullValue(ty);
        gv = new GlobalVariable(module, ty, false,
                                GlobalValue::ExternalLinkage,
                                zero, name, nullptr,
                                GlobalVariable::NotThreadLocal,
                                1, true);
      }
      
      return gv;
    }

    void setKernelHeader(Function* kernel,
                         std::vector<trace_header_inst_t>& inst_debugdata) {

      
      // allocate space for kernel header struct
      size_t kernel_header_size =
        sizeof(trace_header_kernel_t) +
        sizeof(trace_header_inst_t) * inst_debugdata.size() +
        4;
      std::unique_ptr<byte[]> kernel_header_buf =
        std::make_unique<byte[]>(kernel_header_size);
      if (!kernel_header_buf) {
        fprintf(stderr, "cuprof: Failed to build debug data!\n");
        abort();
      }
      trace_header_kernel_t* kernel_header =
        (trace_header_kernel_t*) kernel_header_buf.get();
      //memset(kernel_header, 0, kernel_header_size);


      // init kernel header struct
      const StringRef kernel_name_ref = kernel->getName();
      const std::string kernel_name = kernel_name_ref.str();
      uint8_t kernel_name_len = std::min(kernel_name.length(), (size_t)TRACE_KERNELNAME_MAXLEN);
      
      kernel_header->insts_count = inst_debugdata.size();
      kernel_header->kernel_name_len = kernel_name_len;
      memcpy(kernel_header->kernel_name, kernel_name.c_str(), kernel_name_len);
      memcpy(kernel_header->insts, inst_debugdata.data(),
             sizeof(trace_header_inst_t) * kernel_header->insts_count);

      
      // get serialized header data
      size_t kernel_data_size;
      byte* kernel_data = header_serialize(&kernel_data_size, kernel_header);
      

      // set kernel header to global constant
      Module& module = *kernel->getParent();
      LLVMContext& ctx = module.getContext();

      const std::string varname_str =
        getSymbolName(kernel_name, CUPROF_SYMBOL_DATA_VAR);
      
      GlobalVariable* debugdata = module.getNamedGlobal(varname_str.c_str());
      if (debugdata != nullptr) {
        debugdata->eraseFromParent();
        debugdata = nullptr;
      }
      
      ArrayRef<byte> data_arr_ref = ArrayRef<byte>(kernel_data, kernel_data_size);
      Constant* var_init = ConstantDataArray::get(ctx, data_arr_ref);
      new GlobalVariable(module, var_init->getType(), false,
                         GlobalValue::ExternalLinkage,
                         var_init, varname_str.c_str(), nullptr,
                         GlobalValue::ThreadLocalMode::NotThreadLocal,
                         1, false);
      
      
      free(kernel_data);
    }


  
/***********************
 * Module IR Insertion *
 ***********************/
  

    Value* getSm(IRBuilder<> irb) {
      InlineAsm* sm_asm = InlineAsm::get(i32_fty,
                                         "mov.u32 $0, %smid;", "=r", false,
                                         InlineAsm::AsmDialect::AD_ATT );
      return irb.CreateCall(sm_asm);
    }

  
    Value* getWarpp(IRBuilder<> irb) {
      InlineAsm* warpp_asm =InlineAsm::get(i32_fty,
                                           "mov.u32 $0, %warpid;", "=r", false,
                                           InlineAsm::AsmDialect::AD_ATT );
      return irb.CreateCall(warpp_asm);
    }

    Value* getLane(IRBuilder<> irb) {
      InlineAsm* laneid_asm = InlineAsm::get(i32_fty,
                                             "mov.u32 $0, %laneid;", "=r", false,
                                             InlineAsm::AsmDialect::AD_ATT );
      return irb.CreateCall(laneid_asm);
    }

    Value* getGrid(IRBuilder<> irb) {
      InlineAsm* gridid_asm = InlineAsm::get(i64_fty,
                                             "mov.u64 $0, %gridid;", "=l", false,
                                             InlineAsm::AsmDialect::AD_ATT );
      return irb.CreateCall(gridid_asm);
    }

    void getCta(IRBuilder<> irb, Value* (* cta)[3]) {

      InlineAsm* cta_asm[3] = {
        InlineAsm::get(i32_fty, "mov.u32 $0, %ctaid.x;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT ),
        InlineAsm::get(i32_fty, "mov.u32 $0, %ctaid.y;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT ),
        InlineAsm::get(i32_fty, "mov.u32 $0, %ctaid.z;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT )
      };
      
      for (int i = 0; i < 3; i++) {
        (*cta)[i] = irb.CreateCall(cta_asm[i]);
        (*cta)[i] = irb.CreateZExt((*cta)[i], i64_ty);
      }
    }

    Value* getCtaSerial(IRBuilder<> irb, Value* cta[3]) {
    
      // cta_serial: <32-bit: cta_x> <16-bit: cta_y> <16-bit: cta_z>
      Value* cta_serial_x = irb.CreateShl(cta[0], 32);
      Value* cta_serial_y = irb.CreateAnd(cta[1], 0xFFFF);
      cta_serial_y = irb.CreateShl(cta_serial_y, 16);
      Value* cta_serial_z = irb.CreateAnd(cta[2], 0xFFFF);
      Value* cta_serial = irb.CreateOr(
        irb.CreateOr(cta_serial_x, cta_serial_y),
        cta_serial_z
        );

      return cta_serial;
    }

    Value* getCtaIndex(IRBuilder<> irb, Value* cta[3]) {

      
      InlineAsm* ncta_asm[2] = {
        InlineAsm::get(i32_fty, "mov.u32 $0, %nctaid.x;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT ),
        InlineAsm::get(i32_fty, "mov.u32 $0, %nctaid.y;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT )
      };
      
      Value* ncta[2];
      for (int i = 0; i < 2; i++) {
        ncta[i] = irb.CreateCall(ncta_asm[i]);
        ncta[i] = irb.CreateZExt(ncta[i], i64_ty);
      }

      // cta_i: cta_x + (cta_y * ncta_x) + (cta_z * ncta_x * ncta_y)
      Value* first = cta[0];
      Value* second = irb.CreateMul(cta[2], ncta[1]);
      second = irb.CreateAdd(second, cta[1]);
      second = irb.CreateMul(second, ncta[0]);
      Value* cta_i = irb.CreateAdd(first, second);

      return irb.CreateTrunc(cta_i, i32_ty);;
    }

    Value* getWarpv(IRBuilder<> irb) {

      InlineAsm* tid_asm[3] = {
        InlineAsm::get(i32_fty, "mov.u32 $0, %tid.x;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT ),
        InlineAsm::get(i32_fty, "mov.u32 $0, %tid.y;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT ),
        InlineAsm::get(i32_fty, "mov.u32 $0, %tid.z;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT )
      };
    
      Value* tid[3];
      for (int i = 0; i < 3; i++) {
        tid[i] = irb.CreateCall(tid_asm[i]);
        tid[i] = irb.CreateZExt(tid[i], i64_ty);
      }
    
      InlineAsm* ntid_asm[2] = {
        InlineAsm::get(i32_fty, "mov.u32 $0, %ntid.x;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT ),
        InlineAsm::get(i32_fty, "mov.u32 $0, %ntid.y;", "=r", false,
                       InlineAsm::AsmDialect::AD_ATT )
      };

      Value* ntid[2];
      for (int i = 0; i < 2; i++) {
        ntid[i] = irb.CreateCall(ntid_asm[i]);
        ntid[i] = irb.CreateZExt(ntid[i], i64_ty);
      }

      // tid_x + ntid_x * (tid_y + tid_z * ntid_y)
      Value* thread_i = irb.CreateAdd(
        irb.CreateMul(tid[2], ntid[1]),
        tid[1]
        );
      thread_i = irb.CreateMul(ntid[0], thread_i);
      thread_i = irb.CreateAdd(tid[0], thread_i);

      Value* warpv = irb.CreateTrunc(
        irb.CreateUDiv(thread_i, ConstantInt::get(i64_ty, 32)),
        i32_ty
        );

      return warpv;
    }


    void insertFilterVolatile(IRBuilder<> irb, Value** to_be_traced_p,
                              TraceInfoValues* info, Value* sm, Value* warpp) {
    
      // apply volatile filters if exists
      if (args.sm.size() > 0 || args.warpp.size() > 0) {
        Value* to_be_traced_volatile_ptr = irb.CreateAlloca(i8_ty);
                       
        Value* filter_volatile_call_args[] = {
          to_be_traced_volatile_ptr,
          info->filter_sm, info->filter_warpp,
          info->filter_sm_count,
          info->filter_warpp_count,
          sm, warpp
        };
        irb.CreateCall(filter_volatile_call, filter_volatile_call_args);

        Value* to_be_traced_volatile = irb.CreateLoad(i8_ty, to_be_traced_volatile_ptr);
        *to_be_traced_p = irb.CreateAnd(*to_be_traced_p, to_be_traced_volatile);
      }

    }

  

/*******************
 * Module Analysis *
 *******************/

  
    std::vector<Function*> getKernelFunctions(Module& module) {
      std::set<Function*> kernels;
      NamedMDNode* kernel_md = module.getNamedMetadata("nvvm.annotations");
      if (kernel_md) {
        // MDNodes in NamedMDNode
        for (const MDNode* node : kernel_md->operands()) {
          // MDOperands in MDNode
          for (const MDOperand& op : node->operands()) {
            Metadata* md = op.get();
            ValueAsMetadata* val = dyn_cast_or_null<ValueAsMetadata>(md);
            if (!val) continue;
            Function* func = dyn_cast<Function>(val->getValue());
            if (!func) continue;
            kernels.insert(func);
          }
        }
      }
      return std::vector<Function*>(kernels.begin(), kernels.end());
    }

    enum PointerKind {
      PK_OTHER = 0,
      PK_GLOBAL,
      PK_UNINITIALIZED,
    };

    PointerKind mergePointerKinds(PointerKind pk1, PointerKind pk2) {
      return pk1 < pk2 ? pk1 : pk2;
    }

    PointerKind getPointerKind(Value* val, bool is_kernel) {
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
        while (BitCastOperator* cast = dyn_cast<BitCastOperator>(node)) {
          node = cast->getOperand(0);
        }
        if (isa<AllocaInst>(node)) {
          kind = mergePointerKinds(kind, PK_OTHER);
        } else if (isa<GlobalValue>(node)) {
          kind = mergePointerKinds(kind, PK_GLOBAL);
        } else if (isa<Argument>(node)) {
          kind = mergePointerKinds(kind, is_kernel ? PK_GLOBAL : PK_OTHER);
        } else if (GEPOperator* gep = dyn_cast<GEPOperator>(node)) {
          stack.push_back(gep->getPointerOperand());
        } else if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(node)) {
          stack.push_back(gep->getPointerOperand());
        } else if (AtomicRMWInst* atomic = dyn_cast<AtomicRMWInst>(node)) {
          stack.push_back(atomic->getPointerOperand());
        } else if (isa<CallInst>(node)) {
          report_fatal_error("Base Pointer is result of function. No.");
        } else if (PHINode* phi = dyn_cast<PHINode>(node)) {
          int num_incomning = phi->getNumIncomingValues();
          for (int i = 0; i < num_incomning; ++i) {
            stack.push_back(phi->getIncomingValue(i));
          }
        }
      }

      return kind;
    }
  

    std::vector<Instruction*> collectGlobalMemAccesses(Function* kernel) {
      std::vector<Instruction*> result;
      for (BasicBlock& basicblock : *kernel) {
        for (Instruction& inst : basicblock) {
          PointerKind kind = PK_OTHER;
          if (LoadInst* load = dyn_cast<LoadInst>(&inst)) {
            kind = getPointerKind(load->getPointerOperand(), true);
          } else if (StoreInst* store = dyn_cast<StoreInst>(&inst)) {
            kind = getPointerKind(store->getPointerOperand(), true);
          } else if (AtomicRMWInst* atomic = dyn_cast<AtomicRMWInst>(&inst)) {
            // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
            kind = getPointerKind(atomic->getPointerOperand(), true);
          } else if (AtomicCmpXchgInst* atomic = dyn_cast<AtomicCmpXchgInst>(&inst)) {
            // ATOMIC CAS //
            kind = getPointerKind(atomic->getPointerOperand(), true);
          } else if (CallInst* call = dyn_cast<CallInst>(&inst)) {
            Function* callee = call->getCalledFunction();
            if (callee == nullptr) continue;
            StringRef callee_name = callee->getName();
            if (callee_name.startswith("llvm.nvvm.atomic")) {
              // ATOMIC Inc/Dec //
              kind = getPointerKind(call->getArgOperand(0), true);
            } else if ( callee_name == "___cuprof_trace") {
              report_fatal_error("already instrumented!");
            } else if ( !callee_name.startswith("llvm.") ) {
              std::string error = "call to non-intrinsic: ";
              error.append(callee_name);
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
      for (BasicBlock& basicblock : *kernel) {
        for (Instruction& inst : basicblock) {
          if (isa<ReturnInst>(&inst)) {
            result.push_back(&inst);
          }
        }
      }
      return result;
    }



/**************************
 * Module Instrumentation *
 **************************/
  
  
    IRBuilderBase::InsertPoint setupTraceInfo(Function* kernel, TraceInfoValues* info) {
      IRBuilder<> irb(kernel->getEntryBlock().getFirstNonPHI());

      Module& module = *kernel->getParent();


    
      // get basic info
      Value* lane = getLane(irb);
      Value* warpv = getWarpv(irb);
      Value* cta[3];
      getCta(irb, &cta);
      Value* cta_serial = getCtaSerial(irb, cta);
      Value* cta_i = getCtaIndex(irb, cta);
      Value* grid = getGrid(irb);

    

      // initialize alloc / commit / count / records
    
      //GlobalVariable* gv_tmp = getOrInsertGlobalVariableExtern(module, trace_base_info_ty, CUPROF_TRACE_BASE_INFO);
      //assert(gv_tmp != nullptr);
      
      //Value* trace_info = irb.CreateAlloca(trace_info_ty);
      //irb.CreateCall(allocate_call, {trace_info});

      
      //std::string kernel_name = ;
      GlobalVariable* kernel_id_ptr = getOrInsertGlobalVariableExtern(
        module, i32_ty,
        getSymbolName(kernel->getName().str(),
                      CUPROF_SYMBOL_KERNEL_ID).c_str()
        );
      Value* kernel_id = irb.CreateLoad(kernel_id_ptr, "kernel_id");
      GlobalVariable* trace_info = getOrInsertGlobalVariableExtern(
        module, trace_info_ty, CUPROF_TRACE_BASE_INFO
        );

      

      
      //////////////////////////////////

      
      Value* slot = irb.CreateAnd(cta_i, irb.getInt32(SLOTS_PER_STREAM_IN_A_DEV - 1));

      //Value* base_i = irb.CreateMul(slot, ConstantInt::get(i32_ty, CACHELINE));
      //Value* slot_i = irb.CreateMul(irb.CreateZExt(slot, i64_ty),
      //                              ConstantInt::get(i64_ty, RECORDS_PER_SLOT));
      //slot_i = irb.CreateMul(slot_i, ConstantInt::get(i64_ty, RECORD_MAX_SIZE));
      
      Value* base_i = irb.CreateMul(slot, ConstantInt::get(i32_ty, CACHELINE));
      Value* slot_i = irb.CreateMul(slot, ConstantInt::get(i32_ty, RECORDS_PER_SLOT));
      slot_i = irb.CreateMul(slot_i, ConstantInt::get(i32_ty, RECORD_MAX_SIZE));

      Value* allocs_ptr = irb.CreateStructGEP(nullptr, trace_info, 0);
      Value* alloc = irb.CreateLoad(allocs_ptr, "alloc");
      alloc = irb.CreateInBoundsGEP(i8_ty, alloc, base_i);
      alloc = irb.CreateBitCast(alloc, i32p_ty);

      Value* commits_ptr = irb.CreateStructGEP(nullptr, trace_info, 1);
      Value* commit = irb.CreateLoad(commits_ptr, "commit");
      commit = irb.CreateInBoundsGEP(i8_ty, commit, base_i);
      commit = irb.CreateBitCast(commit, i32p_ty);

      Value* flusheds_ptr = irb.CreateStructGEP(nullptr, trace_info, 2);
      Value* flushed = irb.CreateLoad(flusheds_ptr, "flushed");
      flushed = irb.CreateInBoundsGEP(i8_ty, flushed, base_i);
      flushed = irb.CreateBitCast(flushed, i32p_ty);
      
      Value* signals_ptr = irb.CreateStructGEP(nullptr, trace_info, 3);
      Value* signal = irb.CreateLoad(signals_ptr, "signal");
      signal = irb.CreateInBoundsGEP(i8_ty, signal, base_i);
      signal = irb.CreateBitCast(signal, i32p_ty);

      Value* records_ptr = irb.CreateStructGEP(nullptr, trace_info, 4);
      Value* records = irb.CreateLoad(records_ptr, "records");
      records = irb.CreateInBoundsGEP(i8_ty, records, slot_i);



      // initialize constant filters
    
      Value* to_be_traced = irb.CreateAlloca(i8_ty);
    
      Value* filter_grid = getOrInsertGlobalArray<uint64_t>(module, args.grid, "___cuprof_filter_grid");
      filter_grid = irb.CreateConstGEP1_32(filter_grid, 0);
      filter_grid = irb.CreateBitCast(filter_grid, i64p_ty);
    
      Value* filter_cta = getOrInsertGlobalArray<uint64_t>(module, args.cta, "___cuprof_filter_cta");
      filter_cta = irb.CreateConstGEP1_32(filter_cta, 0);
      filter_cta = irb.CreateBitCast(filter_cta, i64p_ty);
    
      Value* filter_warpv = getOrInsertGlobalArray<uint32_t>(module, args.warpv, "___cuprof_filter_warpv");
      filter_warpv = irb.CreateConstGEP1_32(filter_warpv, 0);
      filter_warpv = irb.CreateBitCast(filter_warpv, i32p_ty);
    
      Constant* filter_grid_count = ConstantInt::get(i8_ty, args.grid.size());
      Constant* filter_cta_count = ConstantInt::get(i8_ty, args.cta.size());
      Constant* filter_warp_count = ConstantInt::get(i8_ty, args.warpv.size());
    
      Value* filter_call_args[] = {
        to_be_traced, filter_grid, filter_cta, filter_warpv,
        filter_grid_count, filter_cta_count, filter_warp_count,
        cta_serial, warpv
      };
      irb.CreateCall(filter_call, filter_call_args);
      to_be_traced = irb.CreateLoad(to_be_traced, "to_be_traced");



      // initialize volatile filters
    
      Value* filter_sm_count = ConstantInt::get(i8_ty, args.sm.size());
      Value* filter_sm = getOrInsertGlobalArray<uint32_t>(module, args.sm, "___cuprof_filter_sm");
      filter_sm = irb.CreateConstGEP1_32(filter_sm, 0);
      filter_sm = irb.CreateBitCast(filter_sm, i32p_ty);
    
      Value* filter_warpp_count = ConstantInt::get(i8_ty, args.warpp.size());
      Value* filter_warpp = getOrInsertGlobalArray<uint32_t>(module, args.warpp, "___cuprof_filter_warpp");
      filter_warpp = irb.CreateConstGEP1_32(filter_warpp, 0);
      filter_warpp = irb.CreateBitCast(filter_warpp, i32p_ty);



      // set info
    
      info->alloc = alloc;
      info->commit = commit;
      info->flushed = flushed;
      info->signal = signal;
      info->records = records;
      info->grid = grid;
      info->cta_serial = cta_serial;
      info->warpv = warpv;
      info->lane = lane;
      info->kernel = kernel_id;
      info->filter_sm = filter_sm;
      info->filter_warpp = filter_warpp;
      info->filter_sm_count = filter_sm_count;
      info->filter_warpp_count = filter_warpp_count;
    
      info->to_be_traced = to_be_traced;
    
      return irb.saveIP();
    }




    bool appendInstHeader(std::vector<trace_header_inst_t>& inst_headers,
                          Value* target, uint32_t instid, uint32_t inst_type,
                          uint32_t meta) {

      bool debuginfo_not_found = false;

      // target basic info
      trace_header_inst_t inst_header = {};
      inst_header.instid = instid;
      inst_header.inst_type = inst_type;
      inst_header.meta = meta;

      // target debug info
      // '-g' option needed for debug info!
      if (const DebugLoc& debug = isa<Instruction>(target) ?
          ((Instruction*)target)->getDebugLoc() : nullptr) {
        
        StringRef path = debug->getFilename();
        int path_len = std::min(path.size(), (size_t)TRACE_FILENAME_MAXLEN);

        
        inst_header.row = debug->getLine();
        inst_header.col = debug->getColumn();
        inst_header.inst_filename_len = path_len;
        inst_header.inst_filename = (char*) path.data();
        //memcpy(inst_header.inst_filename, path.c_str(), path_len);
      }
      else if (const DISubprogram* debug = isa<Function>(target) ?
               ((Function*)target)->getSubprogram() : nullptr) {
        
        StringRef path = debug->getFilename();
        int path_len = std::min(path.size(), (size_t)TRACE_FILENAME_MAXLEN);

        
        inst_header.row = debug->getLine();
        inst_header.col = 0;
        inst_header.inst_filename_len = path_len;
        inst_header.inst_filename = (char*) path.data();
      }
      else {
        debuginfo_not_found = true;
      }

      // append to vector
      inst_headers.push_back(inst_header);

      return debuginfo_not_found;
    }


  
    bool instrumentMemAccess(Function* kernel, ArrayRef<Instruction*> memacc_insts,
                             TraceInfoValues* info,
                             std::vector<trace_header_inst_t>& inst_headers) {

      bool debuginfo_not_found = false;
      uint32_t instid = inst_headers.size()+1;
      const DataLayout& dat_layout = kernel->getParent()->getDataLayout();
      IRBuilder<> irb(kernel->front().getFirstNonPHI());
      
      for (Instruction* inst : memacc_insts) {

        
        // determine the pointer type
        
        Value* ptr_operand = nullptr;
        uint8_t type_num = 0;
        
        if (LoadInst* loadinst = dyn_cast<LoadInst>(inst)) {
          ptr_operand = loadinst->getPointerOperand();
          type_num = (uint8_t)RECORD_LOAD;
                             
        } else if (StoreInst* storeinst = dyn_cast<StoreInst>(inst)) {
          ptr_operand = storeinst->getPointerOperand();
          type_num = (uint8_t)RECORD_STORE;
                             
        } else if (AtomicRMWInst* atomicinst = dyn_cast<AtomicRMWInst>(inst)) {
          // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
          ptr_operand = atomicinst->getPointerOperand();
          type_num = (uint8_t)RECORD_ATOMIC;
                             
        } else if (AtomicCmpXchgInst* atomicinst = dyn_cast<AtomicCmpXchgInst>(inst)) {
          // ATOMIC CAS //
          ptr_operand = atomicinst->getPointerOperand();
          type_num = (uint8_t)RECORD_ATOMIC;
                             
        } else if (CallInst* callinst = dyn_cast<CallInst>(inst)) {
          // ATOMIC Inc/Dec //
          assert(callinst->getCalledFunction()->getName()
                 .startswith("llvm.nvvm.atomic"));
          ptr_operand = callinst->getArgOperand(0);
          type_num = RECORD_ATOMIC;
          
        } else {
          report_fatal_error("invalid access type encountered, this should not have happened");
        }

        

        // insert func call
        
        // insert argument calculation
        irb.SetInsertPoint(inst->getNextNode()); // insert after the access
        Value* addr = irb.CreatePtrToInt(ptr_operand, irb.getInt64Ty());
        Value* sm = getSm(irb);
        Value* warpp = getWarpp(irb);
        Constant* instid_const = ConstantInt::get(i32_ty, instid);

        
        // insert volatile filter if exists
        Value* to_be_traced = info->to_be_traced;
        insertFilterVolatile(irb, &to_be_traced, info, sm, warpp);

        // create call
        Value* trace_call_args[] = {
          info->alloc, info->commit,
          info->flushed, info->signal,
          info->records, addr,
          info->grid, info->cta_serial,
          info->warpv, info->lane,
          instid_const, info->kernel,
          sm, warpp,
          to_be_traced
        };
        irb.CreateCall(trace_call, trace_call_args);


        
        // append inst info to the header
        
        PointerType* p_ty = dyn_cast<PointerType>(ptr_operand->getType());
        uint32_t req_size = dat_layout.getTypeStoreSize(p_ty->getElementType());

        debuginfo_not_found = debuginfo_not_found
          || appendInstHeader(inst_headers, inst, instid++, type_num, req_size);
      }
      

      return !debuginfo_not_found;

    }


  
    bool instrumentScheduling(Function* kernel, IRBuilderBase::InsertPoint ipfront,
                              ArrayRef<Instruction*> retinsts,
                              TraceInfoValues* info,
                              std::vector<trace_header_inst_t>& inst_headers) {
    
      bool debuginfo_not_found = false;
      uint32_t instid = inst_headers.size()+1;
      IRBuilder<> irb(kernel->front().getFirstNonPHI());

      // trace call
      irb.restoreIP(ipfront);
      Value* addr = ConstantInt::get(i64_ty, -1);
      Value* instid_arg = ConstantInt::get(i32_ty, instid);
      Value* sm = getSm(irb);
      Value* warpp = getWarpp(irb);
      
      
      // insert volatile filter if exists
      Value* to_be_traced = info->to_be_traced;
      insertFilterVolatile(irb, &to_be_traced, info, sm, warpp);
    
    
      Value* trace_call_args[] = {
        info->alloc, info->commit,
        info->flushed, info->signal,
        info->records, addr,
        info->grid, info->cta_serial,
        info->warpv, info->lane,
        instid_arg, info->kernel,
        sm, warpp,
        to_be_traced
      };
      irb.CreateCall(trace_call, trace_call_args);

      debuginfo_not_found = debuginfo_not_found
        || appendInstHeader(inst_headers, kernel, instid++, RECORD_EXECUTE, 0);


      // trace ret
      //addr = ConstantInt::get(i64_ty, 0);
      //type = ConstantInt::get(i8_ty, RECORD_RETURN);
    
      for (Instruction* inst : retinsts) {
        irb.SetInsertPoint(inst);

        if (! isa<ReturnInst>(inst)) {
          report_fatal_error("invalid access type encountered, this should not have happened");
        }

        
        instid_arg = ConstantInt::get(i32_ty, instid);
        Value* sm = getSm(irb);
        Value* warpp = getWarpp(irb);
      
      
        // insert volatile filter if exists
        to_be_traced = info->to_be_traced;
        insertFilterVolatile(irb, &to_be_traced, info, sm, warpp);

        Value* trace_call_args[] = {
          info->alloc, info->commit,
          info->flushed, info->signal,
          info->records, addr,
          info->grid, info->cta_serial,
          info->warpv, info->lane,
          instid_arg, info->kernel,
          sm, warpp,
          to_be_traced
        };
        irb.CreateCall(trace_call, trace_call_args);

        
        Value* trace_ret_call_args[] = {
          info->commit, info->signal,
          info->lane
        };
        irb.CreateCall(trace_ret_call, trace_ret_call_args);

        
        
        debuginfo_not_found = debuginfo_not_found
          || appendInstHeader(inst_headers, inst, instid++, RECORD_RETURN, 0);
      }


      return debuginfo_not_found;
    }


  
/**************
 * Pass Entry *
 **************/
  
  
    bool runOnModule(Module& module) override {

      bool is_cuda = module.getTargetTriple().find("nvptx") != std::string::npos;
      if (!is_cuda) return false;

    
      // type / function call init
      initTypes(module);
      findOrInsertRuntimeFunctions(module);

    
    
      // if kernel args is set, kernel filtering is enabled
      bool kernel_filtering = (args.kernel.size() != 0);
    

      bool debug_without_problem = true; // All debug data is written without problem
      for (Function* kernel : getKernelFunctions(module)) {

      
        // kernel filtering
        if (kernel_filtering && !isKernelToBeTraced(kernel, args.kernel))
          continue;


        // kernel instrumentation
      
        std::vector<Instruction*> accesses = collectGlobalMemAccesses(kernel);
        std::vector<Instruction*> retinsts = collectReturnInst(kernel);
      
        TraceInfoValues info;
        IRBuilderBase::InsertPoint ipfront = setupTraceInfo(kernel, &info);


        
//        trace_header_kernel_t* kernel_header =
//          initKernelDebugData(kernel, accesses.size() + retinsts.size() + 1);

        std::vector<trace_header_inst_t> inst_header;
        
        
        if (args.trace_mem) {
          instrumentMemAccess(kernel, accesses, &info, inst_header);
        }

        if (args.trace_thread) {
          instrumentScheduling(kernel, ipfront, retinsts, &info, inst_header);
        }

        setKernelHeader(kernel, inst_header);
      }
    
      if (!debug_without_problem) {
        std::cerr << "cuprof: No memory access data for \""
                  << module.getModuleIdentifier()
                  << "\" found! Check if \"-g\" option is set.\n";
      }

      return true;
    }

    void getAnalysisUsage(AnalysisUsage&) const override {
    }

  };
  char InstrumentDevicePass::ID = 0;



  
  
  Pass* createInstrumentDevicePass(InstrumentPassArg args) {
    return new InstrumentDevicePass(args);
  }

  static RegisterPass<InstrumentDevicePass>
  X("cuprof-device",
    "inserts device-side instrumentation for cuprof",
    false, false);

}
