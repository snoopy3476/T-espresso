/*****************************************************
 * A header for backward compatibility before LLVM 9.
 * ('FunctionCallee' is not available until LLVM 9)
 */

#if (LLVM_VERSION_MAJOR < 9)


#ifndef __FUNCTION_CALLEE_H__
#define __FUNCTION_CALLEE_H__

#include "llvm/IR/Constant.h"

#define CUDA_LAUNCH_FUNC_NAME "cudaLaunch"
#define CUDA_PUSHCONF_FUNC_NAME "cudaConfigureCall"
#define CUDA_POPCONF_FUNC_NAME "cudaSetupArgument"

namespace llvm {

  class FunctionCallee {
  public:
    template<typename T> FunctionCallee(T* callee) {
      this->callee = callee;
    }
    FunctionCallee(std::nullptr_t) {}
    
    template<typename T> FunctionCallee& operator=(T* callee) {
      this->callee = callee;
      return *this;
    }
    
    Constant* getCallee() {
      return callee;
    }

    operator Constant*() {
      return callee;
    }
  
    Constant* callee;
  };
  
}

#endif


#endif
