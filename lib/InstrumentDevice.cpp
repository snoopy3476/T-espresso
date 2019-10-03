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

#define INCLUDE_LLVM_CUPROF_TRACE_STUFF
#include "TraceIO.h"

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


/******************************************************************************
 * Various helper functions
 */

// Prototype
// __device__ void __mem_trace (uint8_t* records, uint8_t* allocs,
//  uint8_t* commits, uint64_t desc, uint64_t addr, uint32_t slot) {
Constant* getOrInsertTraceDecl(Module& module) {
  LLVMContext& ctx = module.getContext();

  Type* void_ty = Type::getVoidTy(ctx);
  Type* i8p_ty = Type::getInt8PtrTy(ctx);
  Type* i64_ty = Type::getInt64Ty(ctx);
  Type* i32_ty = Type::getInt32Ty(ctx);

  return module.getOrInsertFunction("___cuprof_trace", void_ty,
			       i8p_ty, i8p_ty, i8p_ty,
                               i64_ty, i64_ty, i64_ty, i32_ty, i32_ty);
}

Constant* getOrInsertSetDebugDataDecl(Module& module) {
  LLVMContext& ctx = module.getContext();

  Type* void_ty = Type::getVoidTy(ctx);
  Type* i8p_ty = Type::getInt8PtrTy(ctx);
  Type* i64_ty = Type::getInt64Ty(ctx);

  return module.getOrInsertFunction("___cuprof_set_accdat", void_ty,
			       i8p_ty, i64_ty);
}

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

GlobalVariable* defineDeviceGlobal(Module& module, Type* ty, const Twine& name) {
  Constant* zero = Constant::getNullValue(ty);
  auto* gv = new GlobalVariable(module, ty, false,
                                       GlobalValue::ExternalLinkage, zero, name, nullptr,
                                       GlobalVariable::NotThreadLocal, 1, true);
  gv->setAlignment(1);
  gv->setDSOLocal(true);
  return gv;
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
    while (auto* cast = dyn_cast<BitCastOperator>(node)) {
      node = cast->getOperand(0);
    }
    if (isa<AllocaInst>(node)) {
      kind = mergePointerKinds(kind, PK_OTHER);
    } else if (isa<GlobalValue>(node)) {
      kind = mergePointerKinds(kind, PK_GLOBAL);
    } else if (isa<Argument>(node)) {
      kind = mergePointerKinds(kind, is_kernel ? PK_GLOBAL : PK_OTHER);
    } else if (auto* gep = dyn_cast<GEPOperator>(node)) {
      stack.push_back(gep->getPointerOperand());
    } else if (auto* gep = dyn_cast<GetElementPtrInst>(node)) {
      stack.push_back(gep->getPointerOperand());
    } else if (auto* atomic = dyn_cast<AtomicRMWInst>(node)) {
      stack.push_back(atomic->getPointerOperand());
    } else if (isa<CallInst>(node)) {
      report_fatal_error("Base Pointer is result of function. No.");
    } else if (auto* phi = dyn_cast<PHINode>(node)) {
      int num_incomning = phi->getNumIncomingValues();
      for (int i = 0; i < num_incomning; ++i) {
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
    Value* allocs;
    Value* commits;
    Value* records;
    Value* desc;
    Value* slot;
  };

  

  std::vector<Instruction*> collectGlobalMemAccesses(Function* kernel) {
    std::vector<Instruction*> result;
    for (auto& basicblock : *kernel) {
      for (auto& inst : basicblock) {
        PointerKind kind = PK_OTHER;
        if (auto* load = dyn_cast<LoadInst>(&inst)) {
          kind = getPointerKind(load->getPointerOperand(), true);
        } else if (auto* store = dyn_cast<StoreInst>(&inst)) {
          kind = getPointerKind(store->getPointerOperand(), true);
        } else if (auto* atomic = dyn_cast<AtomicRMWInst>(&inst)) {
          // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
          kind = getPointerKind(atomic->getPointerOperand(), true);
        } else if (auto* atomic = dyn_cast<AtomicCmpXchgInst>(&inst)) {
          // ATOMIC CAS //
          kind = getPointerKind(atomic->getPointerOperand(), true);
        } else if (auto* call = dyn_cast<CallInst>(&inst)) {
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
    for (auto& basicblock : *kernel) {
      for (auto& inst : basicblock) {
        if (isa<ReturnInst>(&inst)) {
          result.push_back(&inst);
        }
      }
    }
    return result;
  }



  

  bool setupAndGetKernelDebugData(Function* kernel, std::vector<char>& debug_data, std::vector<Instruction*> inst_list) {
    LLVMContext& ctx = kernel->getParent()->getContext();
    bool debuginfo_not_found = false;
    //char serialbuf[8] = {0};

    //uint64_t data_size = 0;

    uint64_t kernel_header_size =
      sizeof(trace_header_kernel_t) +
      sizeof(trace_header_inst_t) * inst_list.size() +
      4;
    trace_header_kernel_t* kernel_header =
      (trace_header_kernel_t* ) malloc(kernel_header_size);
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
      trace_header_inst_t* inst_header = &kernel_header->insts[inst_id - 1];
      inst_header->inst_id = inst_id;

      // set inst debug info
      const DebugLoc& loc = inst->getDebugLoc();
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
        debuginfo_not_found = true;
      }

      MDNode* metadata = MDNode::get(ctx, MDString::get(ctx, std::to_string(inst_id)));
      inst->setMetadata(TRACE_DEBUG_DATA, metadata);

    }
    kernel_header->insts_count = inst_id;

    
    char* kernel_data = (char* ) malloc(get_max_header_bytes_after_serialize(kernel_header));
    size_t kernel_data_size = header_serialize(kernel_data, kernel_header);
    debug_data.reserve(debug_data.size() + kernel_data_size);
    debug_data.insert(debug_data.end(),
                      kernel_data,
                      kernel_data + kernel_data_size);

    free(kernel_data);
    

    return !debuginfo_not_found;
  }


  

  IRBuilderBase::InsertPoint setupTraceInfo(Function* kernel, TraceInfoValues* info) {
    LLVMContext& ctx = kernel->getParent()->getContext();
    Type* trace_info_ty = getTraceInfoType(ctx);

    IRBuilder<> irb(kernel->getEntryBlock().getFirstNonPHI());

    Module& module = *kernel->getParent();
    std::string symbol_name = getSymbolNameForKernel(kernel->getName().str());
    //errs() << "creating device symbol " << symbol_name << "\n";
    auto* gv = defineDeviceGlobal(module, trace_info_ty, symbol_name);
    assert(gv != nullptr);
    

    Value* allocs_ptr = irb.CreateStructGEP(nullptr, gv, 0);
    Value* allocs = irb.CreateLoad(allocs_ptr, "allocs");

    Value* commits_ptr = irb.CreateStructGEP(nullptr, gv, 1);
    Value* commits = irb.CreateLoad(commits_ptr, "commits");

    Value* records_ptr = irb.CreateStructGEP(nullptr, gv, 2);
    Value* records = irb.CreateLoad(records_ptr, "records");

    IntegerType* i32_ity = IntegerType::get(module.getContext(), 32);

    FunctionType* i32_fty = FunctionType::get(i32_ity, false);

    InlineAsm* smid_asm = InlineAsm::get(i32_fty,
                                        "mov.u32 $0, %smid;", "=r", false,
                                        InlineAsm::AsmDialect::AD_ATT );
    Value* smid = irb.CreateCall(smid_asm);

    auto smid64 = irb.CreateZExtOrBitCast(smid, irb.getInt64Ty(), "desc");
    auto desc = irb.CreateShl(smid64, 32);

    auto slot = irb.CreateAnd(smid, irb.getInt32(SLOTS_NUM - 1));

    
    info->allocs = allocs;
    info->commits = commits;
    info->records = records;
    info->desc = desc;
    info->slot = slot;
    
    return irb.saveIP();
  }


  

  void instrumentMemAccess(Function* func, ArrayRef<Instruction*> memacc_insts,
                        TraceInfoValues* info) {
    Module& module = *func->getParent();

    Constant* trace_call = getOrInsertTraceDecl(module);
    if (!trace_call) {
      report_fatal_error("No ___cuprof_trace declaration found");
    }


    const DataLayout& dat_layout = func->getParent()->getDataLayout();
    
    auto allocs = info->allocs;
    auto commits = info->commits;
    auto records = info->records;
    auto slot = info->slot;
    auto desc = info->desc;

    IRBuilder<> irb(func->front().getFirstNonPHI());

      
    IntegerType* i64_ty = IntegerType::get(module.getContext(), 64);
    FunctionType* clockasm_fty = FunctionType::get(i64_ty, false);
    InlineAsm* clock_asm = InlineAsm::get(clockasm_fty,
                                         "mov.u64 $0, %clock64;", "=l", true,
                                         InlineAsm::AsmDialect::AD_ATT );

    for (auto* inst : memacc_insts) {
      Value* ptr_operand = nullptr;
      Value* dat = nullptr;
      Value* ldesc = nullptr;
      irb.SetInsertPoint(inst);
      
      if (auto loadinst = dyn_cast<LoadInst>(inst)) {
        ptr_operand = loadinst->getPointerOperand();
        ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_LOAD << RECORD_TYPE_SHIFT));
                             
      } else if (auto storeinst = dyn_cast<StoreInst>(inst)) {
        ptr_operand = storeinst->getPointerOperand();
        ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_STORE << RECORD_TYPE_SHIFT));
                             
      } else if (auto atomicinst = dyn_cast<AtomicRMWInst>(inst)) {
        // ATOMIC Add/Sub/Exch/Min/Max/And/Or/Xor //
        ptr_operand = atomicinst->getPointerOperand();
        ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_ATOMIC << RECORD_TYPE_SHIFT));
                             
      } else if (auto atomicinst = dyn_cast<AtomicCmpXchgInst>(inst)) {
        // ATOMIC CAS //
        ptr_operand = atomicinst->getPointerOperand();
        ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_ATOMIC << RECORD_TYPE_SHIFT));
                             
      } else if (auto* callinst = dyn_cast<CallInst>(inst)) {
        // ATOMIC Inc/Dec //
        assert(callinst->getCalledFunction()->getName()
               .startswith("llvm.nvvm.atomic"));
        ptr_operand = callinst->getArgOperand(0);
        ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_ATOMIC << RECORD_TYPE_SHIFT));
                             
      } else {
        report_fatal_error("invalid access type encountered, this should not have happened");
      }


      dat = irb.CreatePtrToInt(ptr_operand,  irb.getInt64Ty());
      auto p_ty = dyn_cast<PointerType>(ptr_operand->getType());
      ldesc = irb.CreateOr(ldesc, (uint64_t) dat_layout.getTypeStoreSize(p_ty->getElementType()));

      uint32_t inst_id = 0;
      if (MDNode* mdnode = inst->getMetadata(TRACE_DEBUG_DATA)) {
        inst_id = (uint32_t) std::stoi(cast<MDString>(mdnode->getOperand(0))->getString().str());
      }
      Constant* inst_id_val = Constant::getIntegerValue(irb.getInt32Ty(), APInt(32, inst_id));
      
      Value* clock = irb.CreateCall(clock_asm);
      Value* trace_call_args[] = {records, allocs, commits, ldesc, dat, clock, slot, inst_id_val};
      irb.CreateCall(trace_call, trace_call_args);
    }
  }


  

  void instrumentScheduling(Function* func, IRBuilderBase::InsertPoint ipfront,
                            ArrayRef<Instruction*> retinsts,
                            TraceInfoValues* info) {
	
    Module& module = *func->getParent();

    
    Constant* trace_call = getOrInsertTraceDecl(module);
    if (!trace_call) {
      report_fatal_error("No ___cuprof_trace declaration found");
    }
    
    auto allocs = info->allocs;
    auto commits = info->commits;
    auto records = info->records;
    auto slot = info->slot;
    auto desc = info->desc;



    IntegerType* i32_ty = IntegerType::get(module.getContext(), 32);
    FunctionType* i32_fty = FunctionType::get(i32_ty, false);
    IntegerType* i64_ty = IntegerType::get(module.getContext(), 64);
    FunctionType* i64_fty = FunctionType::get(i64_ty, false);
    
    InlineAsm* clock_asm = InlineAsm::get(i64_fty,
                                         "mov.u64 $0, %clock64;", "=l", true,
                                         InlineAsm::AsmDialect::AD_ATT );
    InlineAsm* laneid_asm = InlineAsm::get(i32_fty,
                                          "mov.u32 $0, %laneid;", "=r", false,
                                          InlineAsm::AsmDialect::AD_ATT );
    
    Constant* inst_id = Constant::getIntegerValue(i32_ty, APInt(32, 0));
    
    
    IRBuilder<> irb(func->front().getFirstNonPHI());
    irb.restoreIP(ipfront);


    // trace call
    Value* clock = irb.CreateCall(clock_asm);
    Value* ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_EXECUTE << RECORD_TYPE_SHIFT));
    Value* laneid = irb.CreateCall(laneid_asm);
    Instruction::CastOps laneid_castop = CastInst::getCastOpcode(laneid, false, i64_ty, false);
    laneid = irb.CreateCast(laneid_castop, laneid, i64_ty);

    Value* trace_call_args[] = {records, allocs, commits, ldesc, laneid, clock, slot, inst_id};
    irb.CreateCall(trace_call, trace_call_args);


    // trace ret
    for (auto* inst : retinsts) {
      Value* dat = nullptr;
      Value* ldesc = nullptr;
      irb.SetInsertPoint(inst);

      if (isa<ReturnInst>(inst)) {
        dat = laneid;
        ldesc = irb.CreateOr(desc, ((uint64_t)RECORD_RETURN << RECORD_TYPE_SHIFT));
      } else {
        report_fatal_error("invalid access type encountered, this should not have happened");
      }

      Value* clock = irb.CreateCall(clock_asm);
      Value* trace_call_args[] = {records, allocs, commits, ldesc, dat, clock, slot, inst_id};
      irb.CreateCall(trace_call, trace_call_args);
    }
  }

  
  
  GlobalVariable* setDebugData(Module& module, std::vector<char> input, const llvm::Twine& kernel_name) {

    LLVMContext& ctx = module.getContext();

    const std::string varname_str = getSymbolNameForKernel(kernel_name, SYMBOL_DATA_VAR);
    
    
    GlobalVariable* debugdata = module.getNamedGlobal(varname_str.c_str());
    if (debugdata != nullptr) {
      debugdata->eraseFromParent();
      debugdata = nullptr;
    }
    
    unsigned int data_len = input.size();
    ArrayRef<char> data_arr_ref = ArrayRef<char>(input.data(), data_len);
    Constant* var_init = ConstantDataArray::get(ctx, data_arr_ref);
    debugdata = new GlobalVariable(module, var_init->getType(), false,
                                   GlobalValue::ExternalLinkage,
                                   var_init, varname_str.c_str(), nullptr,
                                   GlobalValue::ThreadLocalMode::NotThreadLocal,
                                   1, false);
    debugdata->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    debugdata->setAlignment(1);


    return debugdata;
  }

  
  bool runOnModule(Module& module) override {
    
    bool is_cuda = module.getTargetTriple().find("nvptx") != std::string::npos;
    if (!is_cuda) return false;

    
    // if kernel args is set, kernel filtering is enabled
    bool kernel_filtering = (args.kernel.size() != 0);
    

    bool debug_without_problem = true; // All debug data is written without problem
    std::vector<char> debugdata;
    for (auto* kernel : getKernelFunctions(module)) {
      std::string kernel_name_sym = kernel->getName().str();

      
      // kernel filtering
      
      if (kernel_filtering) {
        DISubprogram* kernel_debuginfo = kernel->getSubprogram();
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
      
      
      debug_without_problem &= setupAndGetKernelDebugData(kernel, debugdata, accesses);
      if (args.trace_mem) {
        instrumentMemAccess(kernel, accesses, &info);
      }
      setDebugData(module, debugdata, kernel_name_sym); //////////////////////

      if (args.trace_thread) {
        instrumentScheduling(kernel, ipfront, retinsts, &info);
      }

      
      debugdata.clear();
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

namespace llvm {
  Pass* createInstrumentDevicePass(InstrumentPassArg args) {
    return new InstrumentDevicePass(args);
  }
}

static RegisterPass<InstrumentDevicePass> X("cuprof-device",
                                            "inserts device-side instrumentation for cuprof",
                                            false, false);

