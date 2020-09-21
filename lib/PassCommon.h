#ifndef __PASS_COMMON_H__
#define __PASS_COMMON_H__


// functions need to be static because we link it into both host and device
// instrumentation

#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/Twine.h"

#include "common.h"

static llvm::StructType* getTraceInfoType(llvm::LLVMContext &ctx) {
  using llvm::Type;
  using llvm::StructType;

  Type *fields[] = {
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx)
  };

  return StructType::create(fields, "traceinfo_t");
}



enum CuprofSymbolType {
  CUPROF_SYMBOL_DATA_VAR,
  CUPROF_SYMBOL_DATA_FUNC,
  CUPROF_SYMBOL_KERNEL_ID,
  CUPROF_SYMBOL_BASE_NAME,
  CUPROF_SYMBOL_END,
};


#define CUPROF_TRACE_BASE_INFO "___cuprof_trace_base_info"
#define CUPROF_ACCDAT_VAR "___cuprof_accdat_var"
#define CUPROF_ACCDAT_VARLEN "___cuprof_accdat_varlen"
#define CUPROF_ACCDAT_CTOR "___cuprof_accdat_ctor"
#define CUPROF_ACCDAT_DTOR "___cuprof_accdat_dtor"
const char * const SYMBOL_TYPE_STR[] = {
  "___cuprof_accdat_var_",
  "___cuprof_accdat_func_",
  "___cuprof_kernel_id_",
  "___cuprof_base_name_"
};



static std::string getSymbolName(const llvm::Twine &kernel_name,
                                          CuprofSymbolType type) {
  
  if (type >= CUPROF_SYMBOL_END || type < (CuprofSymbolType)0)
    type = (CuprofSymbolType)0;

  return (SYMBOL_TYPE_STR[type] + kernel_name).str();
}

static std::string getSymbolName(const std::string &kernel_name,
                                          CuprofSymbolType type) {
  return getSymbolName(llvm::Twine(kernel_name), type);
}



static bool isKernelToBeTraced(llvm::Function* kernel, std::vector<std::string> filter_list) {

  const std::string kernel_name_sym = kernel->getName().str();
  llvm::DISubprogram* kernel_debuginfo = kernel->getSubprogram();
  std::string kernel_name_orig;
  if (kernel_debuginfo) {
    kernel_name_orig = kernel_debuginfo->getName().str();
  }

  // stop instrumenting if not listed on enabled kernel
  if (std::find(filter_list.begin(),
                filter_list.end(),
                kernel_name_sym) == filter_list.end() &&
      std::find(filter_list.begin(),
                filter_list.end(),
                kernel_name_orig) == filter_list.end()) {
    return false;
  }

  return true;
}


#endif
