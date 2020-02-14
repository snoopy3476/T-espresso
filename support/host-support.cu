#include "../lib/common.h"
#include "../lib/trace-io.h"

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <libgen.h>

//***************
#define always_assert(cond) do {                        \
    if (!(cond)) {                                      \
      fprintf(stderr, "Assertion failed: %s\n", #cond); \
      abort();                                          \
    }                                                   \
  } while(0)
//*********************
#define cudaChecked(code) do {                  \
    cudaError_t err = code;                     \
    if (err != cudaSuccess) {                   \
      fprintf(stderr, "CUDA error: (%d) %s\n", __LINE__,        \
              cudaGetErrorString(err));         \
      abort();                                  \
    }                                           \
  } while(0)

extern "C" {
  static char* ___cuprof_accdat_var = NULL;
  static uint64_t ___cuprof_accdat_varlen = 0;
  
  extern traceinfo_t* ___cuprof_trace_base_info;
}
//********************
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
//*************************************
static std::string traceName(int device) {
  
  static const char* exename = getexename();
  
  std::string device_str = std::to_string(device);
  
  const char* pattern_env = getenv("CUPROF_TRACE_PATTERN");
  std::string pattern;
  if (pattern_env) {
    pattern = pattern_env;
  } else if (exename) {
    pattern = "./trace-" + std::string(exename) + "-%d";
  } else {
    pattern = "./trace-%d";
  }

  
  size_t pos = pattern.find("%d");
  if (pos != std::string::npos) {
    pattern.replace(pos, 2, device_str);
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
  int device;
} kernel_trace_arg_t;


class TraceConsumer {
public:

  //*********************************
  TraceConsumer(int device) {
    printf("TraceConsumer(%d)\n", device);/////////////////////////

    // backup the currently set device before the constructor,
    // then set device for class initialization
    int device_initial;
    cudaChecked(cudaGetDevice(&device_initial));
    cudaChecked(cudaSetDevice(device));

    mtx_refresh_consume.lock();
    int range[2];
    cudaChecked(cudaDeviceGetStreamPriorityRange(&range[0], &range[1]));
    cudaChecked(cudaStreamCreateWithPriority(&cudastream_trace,
                                             cudaStreamNonBlocking,
                                             range[1]));

    
    // allocate and initialize traceinfo of the host
    cudaChecked(cudaMalloc(&traceinfo.info_d.allocs_d,
                           SLOTS_PER_STREAM_IN_A_DEV * CACHELINE));
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.allocs_d, 0,
                                SLOTS_PER_STREAM_IN_A_DEV * CACHELINE, cudastream_trace));
    
    cudaChecked(cudaMalloc(&traceinfo.info_d.commits_d,
                           SLOTS_PER_STREAM_IN_A_DEV * CACHELINE));
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.commits_d, 0,
                                SLOTS_PER_STREAM_IN_A_DEV * CACHELINE, cudastream_trace));

    cudaChecked(cudaHostAlloc(&traceinfo.counts_h,
                              SLOTS_PER_STREAM_IN_A_DEV * CACHELINE,
                              cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&traceinfo.info_d.counts_d,
                                         traceinfo.counts_h, 0));
    memset(traceinfo.counts_h, 0,
           SLOTS_PER_STREAM_IN_A_DEV * CACHELINE);

    cudaChecked(cudaHostAlloc(&traceinfo.records_h,
                              SLOTS_PER_STREAM_IN_A_DEV * RECORDS_PER_SLOT * RECORD_SIZE,
                              cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&traceinfo.info_d.records_d,
                                         traceinfo.records_h,
                                         0));
    memset(traceinfo.records_h, 0,
           SLOTS_PER_STREAM_IN_A_DEV * RECORDS_PER_SLOT * RECORD_SIZE);

    

    // initialize traceinfo of the device
    cudaChecked(cudaMemcpyToSymbolAsync(___cuprof_trace_base_info, &traceinfo.info_d,
                                        sizeof(traceinfo_t), 0, cudaMemcpyHostToDevice, cudastream_trace));

    

    
    should_run = true;
    does_run = false;
    to_be_terminated = false;

    pipe_name = traceName(device);
    tracefile = trace_write_open(this->pipe_name.c_str());
    if (tracefile == NULL) {
      fprintf(stderr, "unable to open trace file '%s' for writing\n", pipe_name.c_str());
      abort();
    }

      
    this->device = device;
    
    

    header_written = false;

    
    cudaDeviceSynchronize();
    //std::atomic_thread_fence(std::memory_order_release);
    worker_thread = std::thread(consume, this);

    mtx_refresh_consume.unlock();

    start(NULL, "(tmp)", 0, 0);
    
    // revert set device to the device before the constructor
    cudaChecked(cudaSetDevice(device_initial));
  }

  virtual ~TraceConsumer() {
    printf("~TraceConsumer(%d)\n", device);/////////////////////////
    stop(NULL);
    to_be_terminated = true;
    std::atomic_thread_fence(std::memory_order_release);
    should_run = false;
    cv_refresh_consume.notify_all();
    worker_thread.join();
    
    //always_assert(!should_run);
    trace_write_close(tracefile);

    cudaChecked(cudaStreamDestroy(cudastream_trace));

    cudaFree(traceinfo.info_d.allocs_d);
    cudaFree(traceinfo.info_d.commits_d);
    cudaFreeHost(traceinfo.counts_h);
    cudaFreeHost(traceinfo.records_h);
  }

  //******************************
  void start(cudaStream_t stream_target, const char* name,
             uint64_t grid_dim, uint16_t cta_size) {
    printf("start (%d, %zu)\n", device, stream.size());///////////////////////
    stream_mutex.lock();
    
    
    if (!header_written) {
      
      trace_write_header(tracefile, ___cuprof_accdat_var, ___cuprof_accdat_varlen);
      header_written = true;
    }
    
    trace_write_kernel(tracefile, name, grid_dim, cta_size);
    
    
    bool added = addStream(stream_target);
    if ((stream.size() == 1) && added) {
      std::unique_lock<std::mutex> lock_refresh_consume(mtx_refresh_consume);
      should_run = true;
      cv_refresh_consume.notify_all();
      //worker_thread.join();
    }

    //std::size_t size = stream.size();

    stream_mutex.unlock();

    //while (!does_run);
    printf("~start (%d)\n", device);///////////////////////
  }

  void stop(cudaStream_t stream_target) {

    printf("stop (%d, %zu)\n", device, stream.size());///////////////////////
    stream_mutex.lock();

    //std::size_t old_size = stream.size();    
    bool removed = removeStream(stream_target);

    
    if ((stream.size() == 0) && removed) {
      //unique_lock<std::mutex> lock_consume_ended(mtx_consume_ended);
      //mtx_consume_ended.lock();
      should_run = false;
      //worker_thread.join();
      //mtx_refresh_consume.unlock();
      //cv_consume_ended.wait(lock_consume_ended);
    }

    
    stream_mutex.unlock();
    

    //if (size == 0) {
      
      //while (does_run);
    //}
    
    printf("~stop (%d)\n", device);///////////////////////
      //}
  }

  //*****************************
  bool refreshConsumeImmediately() {
    return should_run || to_be_terminated;
  }

  
  
protected:

  //**************************************
  bool addStream(cudaStream_t stream_target) {
    printf("addStream\n");///////////////////////////////////
    bool return_value = false;
    
    auto found_target = std::find(stream.begin(), stream.end(), stream_target);
    if (found_target == stream.end()) {
      stream.push_back(stream_target);
      return_value = true;
    }
    
    return return_value;
  }

  bool removeStream(cudaStream_t stream_target) {
    printf("removeStream\n");/////////////////////////////////
    bool return_value = false;
    
    auto found_target = std::find(stream.begin(), stream.end(), stream_target);
    if (found_target != stream.end()) {
      stream.erase(found_target);
      return_value = true;
    }
    
    return return_value;
  }

  
  // clear up a slot if it is full
  static int consumeSlot(uint8_t* allocs_d, uint8_t* commits_d,
                         uint8_t* counts_h, uint8_t* records_h,
                         tracefile_t out, bool is_kernel_active,
                         cudaStream_t cudastream_trace) {
    
    volatile uint32_t* vcount = (uint32_t*)counts_h;

    
    uint32_t rec_count = *vcount;
    
    if ( (is_kernel_active && rec_count != RECORDS_PER_SLOT) || (!is_kernel_active && !rec_count))
      return 1;
    

    // tmp file buf
    //static char rec_buf[RECORD_SIZE * RECORDS_PER_SLOT];
    //memset(rec_buf, 0, RECORD_SIZE * RECORDS_PER_SLOT);
    //uint32_t rec_ptr = 0;

    static char rec_orig[TRACE_RECORD_SIZE(32)];
    trace_record_t* const rec = (trace_record_t* const) rec_orig;
    
    
    trace_record_addr_t* addr_unit_cur;
    //printf("Start consume!\n");//////////////////////////////////////
    for (int32_t i = 0; i < rec_count; ++i) {
      //printf(" - %i\n", i);////////////////////////////////////
      trace_deserialize((record_t*)&records_h[i * RECORD_SIZE], rec);

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
      
      
      //printf("   - write!\n");////////////////////////////////////
      trace_write_record(out, rec);
      //char buf[TRACE_RECORD_SIZE(32)]; // mem space for addr_len == threads per warp
      //trace_serialize(rec, (record_t*)buf);
      //uint32_t record_size = RECORD_RAW_SIZE(rec->addr_len);
      //memcpy(rec_buf + rec_ptr, buf, record_size);
      //rec_ptr += record_size;
      //printf("   - written!\n"); ///////////////////////////////////
    }

    //TRACEFILE_WRITE(out, rec_buf, rec_ptr);


    //cudaMemsetAsync(commits_d, 0, sizeof(uint32_t), cudastream_trace);
    
    // reset the read slot
    memset(records_h, 0, RECORDS_PER_SLOT * RECORD_SIZE); // records (H)
    
    // ensure commits, counts, records are reset first
    std::atomic_thread_fence(std::memory_order_release);
    //cudaStreamSynchronize(cudastream_trace);
    //cudaMemsetAsync(allocs_d, 0, sizeof(uint32_t), cudastream_trace);
    *vcount = 0; // counts (H)

    return 0;
  }

  // payload function of queue consumer
  static void consume(TraceConsumer* obj) {
    printf("consume (%d)\n", obj->device); ////////////////////////
    

    cudaSetDevice(obj->device);
    obj->does_run = true;

    uint8_t* allocs_d = obj->traceinfo.info_d.allocs_d;
    uint8_t* commits_d = obj->traceinfo.info_d.commits_d;
    uint8_t* counts_h = obj->traceinfo.counts_h;
    uint8_t* records_h = obj->traceinfo.records_h;
    cudaStream_t cudastream_trace = obj->cudastream_trace;

    tracefile_t tracefile = obj->tracefile;
    
    cudaChecked(cudaMemsetAsync(allocs_d, 0,
                                SLOTS_PER_STREAM_IN_A_DEV * CACHELINE, cudastream_trace));
    cudaChecked(cudaMemsetAsync(commits_d, 0,
                                SLOTS_PER_STREAM_IN_A_DEV * CACHELINE, cudastream_trace));
    
    
    
    {
      std::unique_lock<std::mutex> lock_refresh_consume(obj->mtx_refresh_consume);
      obj->cv_refresh_consume.wait(lock_refresh_consume, [obj](){return obj->refreshConsumeImmediately();}); // wait for refresh
    }

    printf("Initial Lock Freed (%d)\n", obj->device); ////////////////////////
    
    
    while (!obj->to_be_terminated) {

      printf("New Iteration (%d)\n", obj->device);//////////////////////////////////
      /*
      uint8_t allocs_local[CACHELINE * SLOTS_PER_STREAM_IN_A_DEV];
      uint8_t commits_local[CACHELINE * SLOTS_PER_STREAM_IN_A_DEV];
      cudaChecked(cudaMemcpyAsync(allocs_local, allocs_d, CACHELINE * SLOTS_PER_STREAM_IN_A_DEV,
                                  cudaMemcpyDeviceToHost, cudastream_trace));
      cudaChecked(cudaMemcpyAsync(commits_local, commits_d, CACHELINE * SLOTS_PER_STREAM_IN_A_DEV,
                                  cudaMemcpyDeviceToHost, cudastream_trace));
                                  cudaStreamSynchronize(cudastream_trace);*/
      /*
      for (int i = 0; i < SLOTS_PER_STREAM_IN_A_DEV; i++) {
        printf("%u/%u/%u ", *(uint32_t*)(&allocs_local[i * CACHELINE]), *(uint32_t*)(&allocs_local[i * CACHELINE]), *(uint32_t*)&counts_h[i * CACHELINE]);/////////////////
      }
      printf("\n"); /////////////////////////
      */
      
      while(obj->should_run) {
        for(int slot = 0; slot < SLOTS_PER_STREAM_IN_A_DEV; slot++) {
          uint32_t offset = slot * CACHELINE;
          uint32_t records_offset = slot * RECORDS_PER_SLOT * RECORD_SIZE;
          consumeSlot(&allocs_d[offset], &commits_d[offset], &counts_h[offset],
                      &records_h[records_offset], tracefile, true, cudastream_trace);
        }
        //printf("S\n");/////////
        //cudaChecked(cudaMemsetAsync(allocs_d, 0,
        //                            SLOTS_PER_STREAM_IN_A_DEV * CACHELINE, cudastream_trace));
        //cudaChecked(cudaMemsetAsync(commits_d, 0,
        //                            SLOTS_PER_STREAM_IN_A_DEV * CACHELINE, cudastream_trace));
        //printf("s\n");/////////////
      }
      
      printf("Stopping... (%d)\n", obj->device);/////////////////////////////

      // after should_run flag has been reset to false, no warps are writing, but
      // there might still be data in the buffers
      for(int slot = 0; slot < SLOTS_PER_STREAM_IN_A_DEV; slot++) {
        uint32_t offset = slot * CACHELINE;
        uint32_t records_offset = slot * RECORDS_PER_SLOT * RECORD_SIZE;
        consumeSlot(&allocs_d[offset], &commits_d[offset], &counts_h[offset],
                    &records_h[records_offset], tracefile, false, cudastream_trace);
      }

      //unique_lock<std::mutex> lock_consume_ended(obj->mtx_consume_ended);
      //obj->cv_consume_ended.notify_all(); // notify ended
      //bool (* pred)() = &obj->doNotWait;

      printf("Stopped (%d)\n", obj->device);/////////////////////////////
      std::unique_lock<std::mutex> lock_refresh_consume(obj->mtx_refresh_consume);
      obj->cv_refresh_consume.wait(lock_refresh_consume, [obj](){return obj->refreshConsumeImmediately();}); // wait for refresh
    }
    
    printf("Terminating... (%d)\n", obj->device);///////////////////////////
    
    
    obj->does_run = false;
    return;
  }

  int device;
  bool header_written;

  std::atomic<bool> should_run;
  std::atomic<bool> does_run;
  std::atomic<bool> to_be_terminated;

  
  std::mutex mtx_refresh_consume;
  std::condition_variable cv_refresh_consume;

  tracefile_t tracefile;
  std::thread worker_thread;
  std::string pipe_name;

  traceinfo_host_t traceinfo;

  cudaStream_t cudastream_trace;
  std::vector<cudaStream_t> stream;
  std::mutex stream_mutex;
  
};

/*******************************************************************************
 * TraceManager acts as a cache for TraceConsumers and ensures only one consumer
 * per stream is exists. RAII on global variable closes files etc.
 * CUDA API calls not allowed inside of stream callback, so TraceConsumer
 * initialization must be performed explicitly;
 */
#define MAX_DEV_COUNT 256
class TraceManager {
public:
  
  TraceManager() {
    printf("TraceManager()\n");///////////////////
    
    initConsumers();
    //device_count = 0;
  }

  void initConsumers() {
    printf("initconsumers()\n");//////////////////////////

    if (consumers != nullptr)
      return;
    
    cudaChecked(cudaGetDeviceCount(&device_count));
    //device_count = 1; ////////////////////////////////////////
    //printf("device_count: %d\n", device_count);///////////////////////////
    consumers = new TraceConsumer*[device_count];
    
    for (int device = 0; device < device_count; device++) {
      consumers[device] = new TraceConsumer(device);
      //printf("consumer %d: %p\n", device, consumers[device]);///////////////////
    }
  }

  
  TraceConsumer* getConsumer(int device) {
    printf("getConsumer(%d)\n", device);////////////////////////////////////////
    
    //printf("device: %d, device_count: %d\n", device, device_count);///////////////////////////
    if (device >= device_count)
      return nullptr;
    else
      return consumers[device];
  }

  
  virtual ~TraceManager() {
    printf("~TraceManager()\n");///////////////////////////////
    if (consumers == nullptr)
      return;
    
    for (int device = 0; device < device_count; device++) {
      if (consumers[device])
        delete consumers[device];
    }
    delete[] consumers;
  }
  
  
private:
  TraceConsumer** consumers;
  int device_count;
};

static TraceManager ___cuprof_trace_manager;

/*******************************************************************************
 * C Interface
 */

extern "C" {
  
  void ___cuprof_accdat_ctor() {
    printf("___cuprof_accdat_ctor()\n");////////////////////////////////
    if (!___cuprof_accdat_var) {
      ___cuprof_accdat_var = (char*) malloc(sizeof(char));
    }
  }
  
  void ___cuprof_accdat_dtor() {
    printf("___cuprof_accdat_dtor()\n");////////////////////////////////
    if (___cuprof_accdat_var) {
      free(___cuprof_accdat_var);
      ___cuprof_accdat_var = NULL;
    }
  }

  void ___cuprof_accdat_append(const char* data, uint64_t data_len) {
    printf("___cuprof_accdat_append()\n");////////////////////////////////
    char* var_tmp = (char*) realloc(___cuprof_accdat_var,
                                    ___cuprof_accdat_varlen + data_len + 1);
    if (!var_tmp) {
      fprintf(stderr, "Failed to initialize memory access data!\n");
      abort();
    }
    
    
    memcpy(var_tmp + ___cuprof_accdat_varlen, data, data_len);
    var_tmp[___cuprof_accdat_varlen + data_len] = '\0';
    
    ___cuprof_accdat_var = var_tmp;
    ___cuprof_accdat_varlen += data_len;
  }



  
  static void ___cuprof_trace_start_callback(cudaStream_t stream, cudaError_t status, void* vargs) {
    kernel_trace_arg_t* vargs_cast = (kernel_trace_arg_t*)vargs;
    int device = vargs_cast->device;
    printf("trace_start_callback (%d)\n", device);/////////////////////
    
    TraceConsumer* consumer = ___cuprof_trace_manager.getConsumer(device);
    //printf("get (%p)\n", consumer);////////////////////////////////////////////
    consumer->start(stream, vargs_cast->kernel_name,
                    vargs_cast->kernel_grid_dim, vargs_cast->kernel_cta_size);
    free(vargs);
  }

  static void ___cuprof_trace_stop_callback(cudaStream_t stream, cudaError_t status, void* vargs) {
    int* device_ptr = (int*)vargs;
    int device = *device_ptr;
    printf("trace_stop_callback (%d)\n", device);/////////////////////
    
    TraceConsumer* consumer = ___cuprof_trace_manager.getConsumer(device);
    //printf("get\n");////////////////////////////////////////////
    consumer->stop(stream);
    
    free(vargs);
  }
  
  
  void ___cuprof_trace_start(int device, cudaStream_t stream, const char* kernel_name,
                            uint64_t grid_dim, uint16_t cta_size) {
    return;
    printf("trace_start (%d)\n", device);///////////////////////////////////////////

    TraceConsumer* consumer = ___cuprof_trace_manager.getConsumer(device);
    //printf("get (%p)\n", consumer);////////////////////////////////////////////
    //consumer->start(stream, kernel_name, grid_dim, cta_size);
    //printf("%p\n", stream);////////////////////////////////////////////////
    //cudaChecked(cudaStreamAddCallback(stream, tttest, NULL, 0));
    //return; ////////////////////////////////////////////
    kernel_trace_arg_t* arg = (kernel_trace_arg_t*) malloc(sizeof(kernel_trace_arg_t));
    if (arg == nullptr) {
        fprintf(stderr, "Unable to allocate memory!\n");
        abort();
      }
    
    *arg = (kernel_trace_arg_t){kernel_name, grid_dim, cta_size, device};
    //printf("device: %u, stream: %p\n", device, stream); ////////////////////
    cudaChecked(cudaStreamAddCallback(stream, ___cuprof_trace_start_callback, (void*)arg, 0));
  }

  void ___cuprof_trace_stop(int device, cudaStream_t stream) {
    return;
    printf("trace_stop (%d)\n", device);///////////////////////////////////////////

    //printf("%p\n", stream);////////////////////////////////////////////////
    //cudaChecked(cudaStreamAddCallback(stream, tttest, NULL, 0));
    //return; //////////////////////////////////////////
    
    int* arg = (int*) malloc(sizeof(int));
    if (arg == nullptr) {
      fprintf(stderr, "Unable to allocate memory!\n");
      abort();
    }
    
    //printf("device: %u, stream: %p\n", device, stream); ////////////////////
    *arg = device;
    cudaChecked(cudaStreamAddCallback(stream, ___cuprof_trace_stop_callback, (void*)arg, 0));
    //
  }

}
