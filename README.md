# T-espresso

The plugin instruments CUDA code such that tracing ① all executes/returns (lifetimes) of every thread and ② all memory accesses to global memory by a kernel, on a per-stream basis.
Traces are stored in a simple run-length encoded binary format, for which we provide an io utils in a header only library.


# Usage
## Required arguments
In order to add tracing to your CUDA application, you can compile CUDA applications using clang as described in the [official manual](https://releases.llvm.org/9.0.0/docs/CompileCudaWithLLVM.html) and two additional arguments (using at least `-O1` might be necessary for some applications).
Let `$BASE` be a variable containing the base path of your llvm installation (so clang can be found as `$BASE/bin/clang`), then the required arguments are:

1. `-g -fplugin=$BASE/lib/libcuprof.so` (or `-g -fplugin=libcuprof.so`, if `LD_LIBRARY_PATH=$BASE/lib:...`) for the compilation of every `.cu` file. This instruments host and device code to emit traces.  
      
    E.g.        
    `$ clang++ -g -fplugin=$BASE/lib/libcuprof.so --cuda-path=... --cuda-gpu-arch=sm_50 -O1 -c -o code.o code.cu`  
    or just  
    `$ clang++ -g -fplugin=libcuprof.so --cuda-path=... --cuda-gpu-arch=sm_50 -O1 -c -o code.o code.cu`, if `$LD_LIBRARY_PATH` is properly set.  
      
2. `$BASE/lib/libcuprofhost.a` (or `-lcuprofhost`, if `LD_LIBRARY_PATH=$BASE/lib:...`) when linking your application. This links the host side runtime (receiving traces from devices and writing them to disk) into your application.  
      
    E.g.  
    `$ clang++ -o application code.o $BASE/lib/libcuprofhost.a`  
    or just  
    `$ clang++ -o application code.o -lcuprofhost`, if `$LD_LIBRARY_PATH` is properly set.
    
E.g.:
```bash
$ LD_LIBRARY_PATH="$BASE/lib:$LD_LIBRARY_PATH" \
  $BASE/bin/clang++ --cuda-gpu-arch=sm_50 -O1 \
                    ... \
                    --cuda-path="$CUDA_DIR" -L"$CUDA_DIR/lib64" \
                    -lcudart -lpthread \
                    -g -fplugin=libcuprof.so -lcuprofhost
```

## Optional arguments
You can compile CUDA application with optional arguments if you need selective tracing (which makes CUPROF to trace only thread/mem-access, or specific kernel/sm/cta/warp). If you want to trace selectively, add the line `-Xclang -plugin-arg-cuprof -Xclang (Selective-tracing-arguments)` for the compilation of every `.cu` file.  
    What `(Selective-tracing-arguments)` you can use are as follows:
1. `thread-only`. Trace only thread executes/returns, and do not trace memory accesses.
2. `mem-only`. Trace only memory access, and do not trace thread executes/returns.
3. `kernel=(kernel_name)`. Trace only specific kernel(s). '(kernel_name)' can be either symbol name (function name after compilation) or original name (function name from the source).
4. `sm=(smid)`. Trace only specific SM(s).
5. `cta=(ctaid)`. Trace only specific CTA(s). '(ctaid)' is the format of `ctaid_x/ctaid_y/ctaid_z` (E.g. cta=3/2/5)
6. `warp=(warpid)`. Trace only specific warp(s).

Selective tracing arguments can be used multiple times, which are separated with commas(,) between different argument types (`thread-only,kernel=...,warp=...`), and separated with spaces( ) between different argument values in the same argument type if enclosed in quotes(") (`warp="0 2 4 7"`).

E.g.:

```bash
$ $BASE/bin/clang++ --cuda-gpu-arch=sm_50 -O1 \
                    ... \
                    -g -fplugin=libcuprof.so \
                    -Xclang -plugin-arg-cuprof -Xclang mem-only,kernel="your_kernel1 _Z12your_kernel2PfS_S_ your_kernel3",sm=0,cta="0/0/0 8/0/0 9/1/2",warp="3 6"
```

The argument types `thread-only` and `mem-only` are exclusive. Multiple argument types are processed with AND conditions, and multiple argument values in the same argument type are processed with OR conditions.
In the example above, multiple arguments are processed like: `(mem-only) && (kernel: your_kernel1 || your_kernel2 || your_kernel3) && (sm: 0) && (cta: 0/0/0 || 8/0/0 || 9/1/2) && (warp: 3 || 6)`


## Outputs
Afterwards, just run your application.
Traces are written to files named `trace-<your application>-<CUDA stream number>.trc`.
One file is created per stream.

You can take a quick look at your traces by using the tool `$BASE/bin/cutracedump`.
Its source code can also be used as a reference for your own analysis tools.

Outputs from cutracedump are the raw data without any descriptions. If you want to change how data is printed, there are some example awk scripts that can handle and manipulate cutracedump outputs in `tools/trc2*.sh` of the CUPROF source root directory. `trc2detail.sh` prepends titles to every fields in cutracedump outputs, so you can use this script to check what each fields in cutracedump means. E.g. `./trc2detail.sh trace-binary-0.trc`

Outputs from cutracedump is as follows:
- Kernel call - `K <kernel_name>`
- Thread trace - `T <optype> <sm> <cta_size> <cta[x]> <cta[y]> <cta[z]> <warp> <clock>`
- Memory trace - `M <optype> <sm> <cta_size> <cta[x]> <cta[y]> <cta[z]> <warp> <clock> <requested_size_per_addr> <addr_1> <addr_2> ... <addr_32> <inst_id> <kernel_name> <inst_line> <inst_col> <srcfile_name> <srcfile_name2 (if spaces exist on srcfile_name)> ...`

`<inst_id>` is an ID per kernel, so '<kernel_name> + <inst_id>' is a unique value of each instruction.
All `<inst_id>`s are determined at compile time (existing as constants in binary), so you can find which memory access instruction (on .ptx or .sass file) is issued at certain memory access trace.


# Compatibility

The CUPROF was developed and tested against LLVM and Clang version 9.0.0 and CUDA toolkit 10.1, and is compatible with CUDA toolkit 9 and 10.

If you need CUPROF with LLVM and Clang version 8.0.1, switch branch to 'cuprof-llvm-8.0.1'.


# Building

The CUPROF is an external project to LLVM (like Clang) that lives in the 
`llvm-project/llvm/tools` directory of the llvm-project tree (also like Clang).

First, download and checkout llvm, and the CUPROF. E.g.:

```bash
  # Clone llvm project
$ cd where-you-want-llvm-to-live
$ git clone https://github.com/llvm/llvm-project

  # Switch to release 9.0.0
$ cd llvm-project
$ git checkout llvmorg-9.0.0

  # Symlink clang
$ cd llvm/tools
$ ln -s ../../clang ./

  # Download CUPROF
$ git clone https://github.com/snoopy3476/cuprof
```

Next, add the line `add_llvm_external_project(cuprof)` to `llvm-project/llvm/tools/CMakeLists.txt`, to make CMake aware of the new external project and include it in the build.  
We typically add it after the block containing the other tools, as in the following diff:

```diff
diff --git a/llvm/tools/CMakeLists.txt b/llvm/tools/CMakeLists.txt                                 
index b654b8c..7eef359 100644
--- a/llvm/tools/CMakeLists.txt
+++ b/llvm/tools/CMakeLists.txt
@@ -46,6 +46,7 @@ add_llvm_external_project(clang)                                       
 add_llvm_external_project(llgo)
 add_llvm_external_project(lld)
 add_llvm_external_project(lldb)
+add_llvm_external_project(cuprof)

 # Automatically add remaining sub-directories containing a 'CMakeLists.txt'             
 # file as external projects.
```

Lastly, configure from the directory `llvm-project/llvm` using cmake and build LLVM. E.g.:

```bash
$ CUDA_DIR=/usr/local/cuda-10.1
$ BASE=build

  # Configure from llvm-project/llvm to $BASE
$ cmake -S llvm-project/llvm -B "$BASE" \
        -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCUPROF_CUDA_FLAGS=--cuda-path="$CUDA_DIR"

  # Build LLVM and CUPROF
$ make -C "$BASE" -j`nproc` \
       PATH="$CUDA_DIR"/bin:"$PATH" \
       CPATH="$CUDA_DIR"/include:"$CPATH" \
       LD_LIBRARY_PATH="$CUDA_DIR"/lib64:"$LD_LIBRARY_PATH"
```

The configuration requires the following flags:

- `-DBUILD_SHARED_LIBS=ON` - The CUPROF is implemented as a plugin, which
  currently does not support static builds of LLVM (linker error message:
  duplicate symbols).
- `-DLLVM_ENABLE_ASSERTIONS=ON` - The current analysis pass to locate kernel
  launch sites relies on the basic block labels set by gpucc. Value/BB labels
  are only set in +Assert builds (or with the `-fno-discard-value-names` clang
  flag), so instrumentation fails with disabled assertions (which is the
  default).
- `-DCUPROF_CUDA_FLAGS=--cuda-path=/path/to/cuda/dir` - required if your
  CUDA installation is located somewhere other than `/usr/local/cuda` (E.g.
  `/opt/cuda-10.1`).

The resulting LLVM build includes the cuprof libraries and can be used as described above.


# Software Authors

- Kim Hwiwon (CUPROF)
####
- Alexander Matz (CUDA Memtrace)
- Dennis Rieber (CUDA Memtrace)
