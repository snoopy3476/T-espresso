#include "../lib/common.h"
#include "../lib/trace-io.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>


#define always_assert(cond) do {                                        \
    if (!(cond)) {                                                      \
      printf("assertion failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      abort();                                                          \
    }                                                                   \
  } while(0)

#define cudaChecked(code) do {                                  \
    cudaError_t err = code;                                     \
    if (err != cudaSuccess) {                                   \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
             cudaGetErrorString(err));                          \
      abort();                                                  \
    }                                                           \
  } while(0)

extern "C" {
  static char* ___cuprof_accdat_var = NULL;
  static uint64_t ___cuprof_accdat_varlen = 0;
}

static const char* getexename() {
  static char* cmdline = NULL;

  if (cmdline != NULL) {
    return cmdline;
  }

  FILE* file = fopen("/proc/self/cmdline", "r");
  if (!file) {
    return NULL;
  }
  size_t n;
  getdelim(&cmdline, &n, 0, file);
  fclose(file);
  cmdline = basename(cmdline);
  return cmdline;
}

/** Allows to specify base name for traces. The first occurence of
 * "?" is replaced with an ID unique to each stream.
 * Default pattern: "./trace-?.bin"
 */
static std::string traceName(std::string id) {
  static const char* exename = getexename(); // initialize once
  const char* pattern_env = getenv("CUPROF_TRACE_PATTERN");
  std::string pattern;
  if (pattern_env) {
    pattern = pattern_env;
  } else if (exename) {
    pattern = "./trace-" + std::string(exename) + "-?.trc";
  } else {
    pattern = "./trace-?.trc";
  }

  size_t pos = pattern.find("?");
  if (pos != std::string::npos) {
    pattern.replace(pos, 1, id);
  }
  return pattern;
}

/*******************************************************************************
 * TraceConsumer sets up and consumes a queue that can be used by kernels to
 * to write their traces into.
 * Only correct when accessed by a single cuda stream.
 * Usage must follow a strict protocol:
 * - one call to TraceConsumer()
 * - zero or more calls to start() ALWAYS followed by stop()
 * - one call to ~TraceConsumer()
 * Trying to repeatedly start or stop a consumer results in process termination.
 *
 * The queue is a multiple producer, single consumer key. Circular queues do not
 * work as expected because we cannot reliably update the in-pointer with a single
 * atomic operation. The result would be corrupted data as the host begins reading
 * data that is falsely assumed to have been committed.
 *
 * Instead we use buffers that are alternatingly filled up by the GPU and cleared
 * out by the CPU.
 * Two pointers are associated with each buffer, an allocation and a commit pointer.
 * A GPU warp first allocates spaces in the buffer using an atomic add on the
 * allocation pointer, then writes its data and increases the commit buffer by the
 * same amount, again using atomic add.
 * The buffered is considered full 
 * a) by the GPU if the allocation pointer is within 32 elements of capacity, and
 * b) by the host if the commit pointer is within 32 elements of capacity.
 * When the buffer is full, all elements are read by the host and the commit and
 * allocation buffer are reset to 0 in this order.
 * 
 * Since a maximum of 1 warp is writing some of the last 32 elements, the commit
 * pointer pointing in this area signals that all warps have written their data.
 * 
 * Several buffers, called "slots", exist in order to reduce contention.
 *
 * Allocation and commit pointers are uint32_t with 64 Byte padding to avoid cache thrashing.
 */


typedef struct kernel_trace_arg_t {
  const char* kernel_name;
  uint64_t kernel_grid_dim;
  uint16_t kernel_cta_size;
} kernel_trace_arg_t;


class TraceConsumer {
public:
  static char* debugdata;
  
  //TraceConsumer(std::string suffix, const char* header_info) {
  TraceConsumer(std::string suffix) {

    //printf("___CUDATRACE_DEBUG_DATA"); //
    //printf(" (%p) = ", ___CUDATRACE_DEBUG_DATA); //
    //printf("%s\n", ___CUDATRACE_DEBUG_DATA); //
    
    this->suffix = suffix;

    cudaChecked(cudaHostAlloc(&records_host, SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE, cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&records_device, records_host, 0));

    cudaChecked(cudaHostAlloc(&allocs_host, SLOTS_NUM * CACHELINE, cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&allocs_device, allocs_host, 0));
    memset(allocs_host, 0, SLOTS_NUM * CACHELINE);

    cudaChecked(cudaHostAlloc(&commits_host, SLOTS_NUM * CACHELINE, cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&commits_device, commits_host, 0));
    memset(commits_host, 0, SLOTS_NUM * CACHELINE);

    should_run = false;
    does_run = false;

    pipe_name = traceName(suffix);

    output = fopen(this->pipe_name.c_str(), "wb");
    if (output == nullptr) {
      printf("unable to open trace file '%s' for writing\n", pipe_name.c_str());
      abort();
    }

    trace_write_header(output, ___cuprof_accdat_var, ___cuprof_accdat_varlen);
  }

  virtual ~TraceConsumer() {
    always_assert(!should_run);
    fclose(output);

    cudaFreeHost(records_host);
    cudaFreeHost(allocs_host);
    cudaFreeHost(commits_host);
  }

  void start(const char* name, uint64_t grid_dim, uint16_t cta_size) {
    always_assert(!should_run);
    should_run = true;

    // reset all buffers and pointers
    memset(allocs_host, 0, SLOTS_NUM * CACHELINE);
    memset(commits_host, 0, SLOTS_NUM * CACHELINE);
    // just for testing purposes
    memset(records_host, 0, SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE);

    trace_write_kernel(output, name, grid_dim, cta_size);

    worker_thread = std::thread(consume, this);

    while (!does_run) {}
  }

  void stop() {
    always_assert(should_run);
    should_run = false;
    while (does_run) {}
    worker_thread.join();
  }

  void fillTraceinfo(traceinfo_t* info) {
    info->allocs = allocs_device;
    info->commits = commits_device;
    info->records = records_device;
    info->slot_size = SLOTS_SIZE;
  }

protected:

  static uint64_t rdtsc(){
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
  }

  // clear up a slot if it is full
  static int consumeSlot(uint8_t* alloc_ptr, uint8_t* commit_ptr, uint8_t* records_ptr,
                         FILE* out, bool is_kernel_active, trace_record_t* acc) {
    // allocs/commits is written by threads on the GPU, so we need it volatile
    volatile uint32_t* vcommit = (uint32_t*)commit_ptr;
    volatile uint32_t* valloc = (uint32_t*)alloc_ptr;

    // flush only if kernel is not active and there are records to write,
    // or kernel is active but slot is full and all allocs are committed (no writes anymore)
    uint32_t rec_count = *vcommit;

    
    if (is_kernel_active) {
      if (! (rec_count >= SLOTS_SIZE && rec_count == *valloc))
        return 1;
    } else {
      if (rec_count == 0)
        return 1;
    }
    
    
    static char rec_orig[TRACE_RECORD_SIZE(32)];
    trace_record_t* const rec = (trace_record_t* const) rec_orig;
    
    
    trace_record_addr_t* addr_unit_cur;
    
    for (int32_t i = 0; i < rec_count; ++i) {

      trace_deserialize((record_t*)&records_ptr[i * RECORD_SIZE], rec);

      rec->addr_len = 1;
      addr_unit_cur = rec->addr_unit;

      // lane 0 is already set, so starts with lane 1
      for (int32_t lane = 1; lane < 32; lane++) {

        uint64_t addr_new = rec->addr_unit[lane].addr;
        
        // set offset
        if (addr_unit_cur->count == 1) {
          int64_t offset = (int64_t) addr_new - addr_unit_cur->addr;
          if ((offset & 0xFFFFFFFFFF000000) == 0x0 ||
              (offset & 0xFFFFFFFFFF000000) == 0xFFFFFFFFFF000000) {
            addr_unit_cur->offset = (int32_t) (offset & 0xFFFFFFFF);
          }
        }

        // same offset - increment current addr_unit count
        if (addr_new == addr_unit_cur->addr +
            (addr_unit_cur->offset * addr_unit_cur->count)) {
          addr_unit_cur->count += 1;
        }

        // different offset - add new addr_unit
        else {
          rec->addr_len += 1;
          addr_unit_cur = &rec->addr_unit[rec->addr_len - 1];
          addr_unit_cur->count = 1;
          addr_unit_cur->addr = addr_new;
        }
 
      }
      
      
      trace_write_record(out, rec);
    }
    
    memset(records_ptr, 0, SLOTS_SIZE * RECORD_SIZE);
    *vcommit = 0;
    // ensure records and commits are reset first
    std::atomic_thread_fence(std::memory_order_release);
    *valloc = 0;

    return 0;
  }

  // payload function of queue consumer
  static void consume(TraceConsumer* obj) {
    obj->does_run = true;
    
    char record_acc_orig[TRACE_RECORD_SIZE(32)] = {0};
    trace_record_t* const record_acc = (trace_record_t* const) record_acc_orig;
    // record_acc->addr_len == 0 -> uninitialized

    uint8_t* allocs = obj->allocs_host;
    uint8_t* commits = obj->commits_host;
    uint8_t* records = obj->records_host;

    FILE* sink = obj->output;


    while(obj->should_run) {
      for(int slot = 0; slot < SLOTS_NUM; slot++) {
        uint32_t allocs_offset = slot * CACHELINE;
        uint32_t commits_offset = slot * CACHELINE;
        uint32_t records_offset = slot * SLOTS_SIZE * RECORD_SIZE;
        consumeSlot(&allocs[allocs_offset], &commits[commits_offset],
                    &records[records_offset], sink, true, record_acc);
      }
    }

    // after should_run flag has been reset to false, no warps are writing, but
    // there might still be data in the buffers
    for(int slot = 0; slot < SLOTS_NUM; slot++) {
      uint32_t allocs_offset = slot * CACHELINE;
      uint32_t commits_offset = slot * CACHELINE;
      uint32_t records_offset = slot * SLOTS_SIZE * RECORD_SIZE;
      consumeSlot(&allocs[allocs_offset], &commits[commits_offset],
                  &records[records_offset], sink, false, record_acc);
    }

    // flush accumulator and reset to uninitialized (if at all initialized)
    if (record_acc->addr_len > 0) {
      trace_write_record(sink, record_acc);
      record_acc->addr_len = 0;
    }

    obj->does_run = false;
    return;
  }

  std::string suffix;

  std::atomic<bool> should_run;
  std::atomic<bool> does_run;

  FILE* output;
  std::thread       worker_thread;
  std::string       pipe_name;

  uint8_t* allocs_host, * allocs_device;
  uint8_t* commits_host, * commits_device;
  uint8_t* records_host, * records_device;
};

/*******************************************************************************
 * TraceManager acts as a cache for TraceConsumers and ensures only one consumer
 * per stream is exists. RAII on global variable closes files etc.
 * CUDA API calls not allowed inside of stream callback, so TraceConsumer
 * initialization must be performed explicitly;
 */
class TraceManager {
public:
  /** Creates a new consumer for a stream if necessary. Returns true if a new
   * consumer had to be created, false otherwise.
   */
  //bool touchConsumer(cudaStream_t stream, const char* header_info) {
  bool touchConsumer(cudaStream_t stream) {
    for (auto &consumer_pair : consumers) {
      if (consumer_pair.first == stream) {
        return false;
      }
    }

    char* suffix;
    asprintf(&suffix, "%d", (int)consumers.size());
    auto new_pair = std::make_pair(stream, new TraceConsumer(suffix));
    free(suffix);
    consumers.push_back(new_pair);
    return true;
  }

  /** Return *already initialized* TraceConsumer for a stream. Aborts application
   * if stream is not initialized.
   */
  TraceConsumer* getConsumer(cudaStream_t stream) {
    for (auto &consumer_pair : consumers) {
      if (consumer_pair.first == stream) {
        return consumer_pair.second;
      }
    }
    always_assert(0 && "trying to get non-existent consumer");
    return nullptr;
  }

  virtual ~TraceManager() {
    for (auto &consumer_pair : consumers) {
      delete consumer_pair.second;
    }
  }
private:
  std::vector<std::pair<cudaStream_t, TraceConsumer*>> consumers;
};
char* TraceConsumer::debugdata = nullptr;

TraceManager __trace_manager;

/*******************************************************************************
 * C Interface
 */

extern "C" {
  
  void ___cuprof_accdat_ctor() {
    if (!___cuprof_accdat_var) {
      ___cuprof_accdat_var = (char*) malloc(sizeof(char));
    }
  }
  
  void ___cuprof_accdat_dtor() {
    if (___cuprof_accdat_var) {
      free(___cuprof_accdat_var);
      ___cuprof_accdat_var = NULL;
    }
  }

  void ___cuprof_accdat_append(const char* data, uint64_t data_len) {
    char* var_tmp = (char*) realloc(___cuprof_accdat_var,
                                    ___cuprof_accdat_varlen + data_len + 1);
    if (!var_tmp) {
      fprintf(stderr, "cuprof: Failed to initialize memory access data!\n");
      abort();
    }

    
    memcpy(var_tmp + ___cuprof_accdat_varlen, data, data_len);
    var_tmp[___cuprof_accdat_varlen + data_len] = '\0';
    
    ___cuprof_accdat_var = var_tmp;
    ___cuprof_accdat_varlen += data_len;
  }


  
  static void __trace_start_callback(cudaStream_t stream, cudaError_t status, void* vargs) {
    auto* consumer = __trace_manager.getConsumer(stream);
    kernel_trace_arg_t* vargs_cast = (kernel_trace_arg_t*)vargs;
    consumer->start(vargs_cast->kernel_name, vargs_cast->kernel_grid_dim,
                    vargs_cast->kernel_cta_size);
    free(vargs_cast);
  }

  static void __trace_stop_callback(cudaStream_t stream, cudaError_t status, void* vargs) {
    auto* consumer = __trace_manager.getConsumer(stream);
    consumer->stop();
  }

  
  
  void __trace_fill_info(const void* info, cudaStream_t stream) {
    auto* consumer = __trace_manager.getConsumer(stream);
    consumer->fillTraceinfo((traceinfo_t*) info);
  }

  void __trace_copy_to_symbol(cudaStream_t stream, const void* symbol, const void* info) {
    cudaChecked(cudaMemcpyToSymbolAsync(symbol, info, sizeof(traceinfo_t), 0, cudaMemcpyHostToDevice, stream));
  }

  void __trace_touch(cudaStream_t stream) {
    __trace_manager.touchConsumer(stream);
  }

  void __trace_start(cudaStream_t stream, const char* kernel_name,
                     uint64_t grid_dim, uint16_t cta_size) {
    kernel_trace_arg_t* arg = (kernel_trace_arg_t*) malloc(sizeof(kernel_trace_arg_t));
    if (arg == nullptr) {
      printf("unable to allocate memory\n");
      abort();
    }
    
    *arg = (kernel_trace_arg_t){kernel_name, grid_dim, cta_size};
    cudaChecked(cudaStreamAddCallback(stream,
                                      __trace_start_callback, (void*)arg, 0));
  }

  void __trace_stop(cudaStream_t stream) {
    cudaChecked(cudaStreamAddCallback(stream,
                                      __trace_stop_callback, (void*)nullptr, 0));
  }

}
