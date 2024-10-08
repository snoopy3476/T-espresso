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

//*******************
//int32_t count_stats[100000] = {0};
//uint8_t* buffer = NULL;

#include <sys/time.h>
static inline double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

static inline long long rtclocku()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return Tp.tv_usec + Tp.tv_sec*1.0e+6;
}

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
  static uint32_t ___cuprof_kernel_count = 0;
  
  traceinfo_t* ___cuprof_trace_base_info = NULL;
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
  
  const char* pattern_env = getenv("CUPROF_TRACE_FILE");
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

//////////
//static uint8_t buf_tmp[RECORDS_PER_SLOT * RECORD_MAX_SIZE];

class TraceConsumer {
public:

  TraceConsumer() {
  }

  //*********************************
  TraceConsumer(int device) {

    int debug_count = 0;
    //printf("%lf - TraceConsumer[%d] (%d)\n", rtclock(), device, debug_count++); //////////////////////////////////////
    
    // backup the currently set device before the constructor,
    // then set device for class initialization
    int device_initial;
    cudaChecked(cudaGetDevice(&device_initial));
    cudaChecked(cudaSetDevice(device));

    //mtx_refresh_consume.lock();
    int range[2];
    cudaChecked(cudaDeviceGetStreamPriorityRange(&range[0], &range[1]));
    cudaChecked(cudaStreamCreateWithPriority(&cudastream_trace,
                                             cudaStreamNonBlocking,
                                             range[1]));

    
    // allocate and initialize traceinfo of the host
    cudaChecked(cudaMalloc(&traceinfo.info_d.allocs_d,
                           SLOTS_PER_DEV * CACHELINE));
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.allocs_d, 0,
                                SLOTS_PER_DEV * CACHELINE,
                                cudastream_trace));
    
    cudaChecked(cudaMalloc(&traceinfo.info_d.commits_d,
                           SLOTS_PER_DEV * CACHELINE));
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.commits_d, 0,
                                SLOTS_PER_DEV * CACHELINE,
                                cudastream_trace));
    /*
    cudaChecked(cudaMallocManaged(&traceinfo.flusheds_h,
                                  SLOTS_PER_DEV * CACHELINE));
    traceinfo.info_d.flusheds_d = traceinfo.flusheds_h;
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.flusheds_d, 0,
                                SLOTS_PER_DEV * CACHELINE,
                                cudastream_trace));
    */
    /*
    cudaChecked(cudaHostAlloc(&traceinfo.flusheds_h,
                              SLOTS_PER_DEV * CACHELINE,
                              cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&traceinfo.info_d.flusheds_d,
                                         traceinfo.flusheds_h, 0));
    memset(traceinfo.flusheds_h, 0,
           SLOTS_PER_DEV * CACHELINE);
    */
    
      
    cudaChecked(cudaMalloc(&traceinfo.info_d.flusheds_d,
                           SLOTS_PER_DEV * CACHELINE));
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.flusheds_d, 0,
                                SLOTS_PER_DEV * CACHELINE,
                                cudastream_trace));
    traceinfo.flusheds_h = traceinfo.info_d.flusheds_d;
    //printf("%p\n", traceinfo.info_d.flusheds_d);



    

    traceinfo.flusheds_old =
      (uint8_t*) malloc(SLOTS_PER_DEV * CACHELINE);
    always_assert(traceinfo.flusheds_old);
    memset(traceinfo.flusheds_old, 0, SLOTS_PER_DEV * CACHELINE);

    cudaChecked(cudaHostAlloc(&traceinfo.signals_h,
                              SLOTS_PER_DEV * CACHELINE,
                              cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&traceinfo.info_d.signals_d,
                                         traceinfo.signals_h, 0));
    memset(traceinfo.signals_h, 0,
           SLOTS_PER_DEV * CACHELINE);


#ifndef CUPROF_RECBUF_MANAGED

    cudaChecked(cudaHostAlloc(&traceinfo.records_h,
                              SLOTS_PER_DEV
                              * SLOT_SIZE,
                              cudaHostAllocMapped));
    cudaChecked(cudaHostGetDevicePointer(&traceinfo.info_d.records_d,
                                         traceinfo.records_h,
                                         0));
    memset(traceinfo.records_h, 0,
           SLOTS_PER_DEV * SLOT_SIZE);

    
#else

    cudaChecked(cudaMallocManaged(&traceinfo.records_h,
                                  SLOTS_PER_DEV
                                  * SLOT_SIZE));
    traceinfo.info_d.records_d = traceinfo.records_h;
    cudaChecked(cudaMemsetAsync(traceinfo.info_d.records_d, 0,
                                SLOTS_PER_DEV * SLOT_SIZE,
                                cudastream_trace));
    
#endif

    
    // initialize traceinfo of the device
    if (___cuprof_trace_base_info) {
      cudaChecked(cudaMemcpyToSymbolAsync(___cuprof_trace_base_info,
                                          &traceinfo.info_d,
                                          sizeof(traceinfo_t), 0,
                                          cudaMemcpyHostToDevice,
                                          cudastream_trace));
    }

    

    
    should_run = true;
    does_run = false;
    //to_be_terminated = false;

    pipe_name = traceName(device);
    tracefile = trace_write_open(this->pipe_name.c_str());
    if (tracefile == NULL) {
      fprintf(stderr, "unable to open trace file '%s' for writing\n",
              pipe_name.c_str());
      abort();
    }

      
    this->device = device;
    
    
    trace_write_header(tracefile, ___cuprof_accdat_var, ___cuprof_accdat_varlen);
    header_written = false;

    
    cudaDeviceSynchronize();
    worker_thread = std::thread(consume, this);

    //mtx_refresh_consume.unlock();

    
    // revert set device to the device before the constructor
    cudaChecked(cudaSetDevice(device_initial));
  }

  virtual ~TraceConsumer() {
    //to_be_terminated = true;
    //std::atomic_thread_fence(std::memory_order_release);
    should_run = false;
    //cv_refresh_consume.notify_all();
    worker_thread.join();

    trace_write_close(tracefile);
    
    cudaChecked(cudaStreamDestroy(cudastream_trace));
    
    
    cudaFree(traceinfo.info_d.allocs_d);
    cudaFree(traceinfo.info_d.commits_d);
    cudaFree(traceinfo.info_d.flusheds_d);
    free(traceinfo.flusheds_old);
    cudaFreeHost(traceinfo.signals_h);

#ifndef CUPROF_RECBUF_MANAGED
    cudaFreeHost(traceinfo.records_h);

#else
    cudaFree(traceinfo.info_d.records_d);
    
#endif
  }

  /*
  void start(cudaStream_t stream_target, const char* name,
             uint64_t grid_dim, uint16_t cta_size) {
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
    }

    stream_mutex.unlock();
  }

  void stop(cudaStream_t stream_target) {

    stream_mutex.lock();

    bool removed = removeStream(stream_target);

    
    if ((stream.size() == 0) && removed) {
      should_run = false;
    }

    
    stream_mutex.unlock();
  }

  bool refreshConsumeImmediately() {
    return should_run || to_be_terminated;
  }
  */
  
  
protected:
/*
  bool addStream(cudaStream_t stream_target) {
  bool return_value = false;
    
  auto found_target = std::find(stream.begin(), stream.end(), stream_target);
  if (found_target == stream.end()) {
  stream.push_back(stream_target);
  return_value = true;
  }
    
  return return_value;
  }

  bool removeStream(cudaStream_t stream_target) {
  bool return_value = false;
    
  auto found_target = std::find(stream.begin(), stream.end(), stream_target);
  if (found_target != stream.end()) {
  stream.erase(found_target);
  return_value = true;
  }
    
  return return_value;
  }
*/
  // clear up a slot if it is full
  static void consumeSlot(uint8_t* alloc_d, uint8_t* commit_d,
                         uint8_t* signal_h,
                         uint8_t* flushed_h, uint8_t* flushed_old,
                         uint8_t* records_d, uint8_t* records_h,
                         tracefile_t out, bool is_kernel_active,
                         cudaStream_t cudastream_trace) {
    
    volatile uint32_t* signal_v = (uint32_t*)signal_h;
    volatile uint32_t* flushed_v = (uint32_t*)flushed_h;
    volatile uint32_t* flushed_old_v = (uint32_t*)flushed_old;
    volatile uint64_t* records_h_lu = (uint64_t*) records_h;
    uint32_t* flushed_d = (uint32_t*)flushed_h;


    
    uint32_t signal = *signal_v;
    uint32_t signal_old = *flushed_old_v;

    // get commit value manually when exit program
    if (!is_kernel_active) {
      cudaChecked(cudaMemcpyAsync(&signal, commit_d, sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
      cudaChecked(cudaStreamSynchronize(cudastream_trace));
    }


    // check if to be flushed
    if ( (uint32_t)(signal - signal_old - 1) >= SLOT_SIZE ) {
      return;
    }

    // change old flushed value on host
    *flushed_old_v = signal;
    

    

    // set flush ranges
    
    uint32_t start_i = signal_old % SLOT_SIZE;
    uint32_t end_i = signal % SLOT_SIZE;
    uint32_t area[2][2];

    if (start_i < end_i) {
      // range of area 1
      area[0][0] = start_i;
      area[0][1] = end_i - start_i;
      // range of area 2
      area[1][0] = 0;
      area[1][1] = 0;
    }
    else {
      // range of area 1
      area[0][0] = start_i;
      area[0][1] = SLOT_SIZE - start_i;
      // range of area 2
      area[1][0] = 0;
      area[1][1] = end_i;
    }
    



    // flush
    
    // flush for area 1
#ifndef CUPROF_RMSYNC_DISABLE
    for (int i = area[0][0]/sizeof(uint64_t);
         i < (area[0][0] + area[0][1])/sizeof(uint64_t);
         i++) {
      while ( !records_h_lu[i] );
    }
#endif
    
    if (! tracefile_write(out, records_h + area[0][0], area[0][1])) {
      fprintf(stderr, "Trace Write Error!\n");
    }

    memset(records_h + area[0][0], 0, area[0][1]);
    
    

    // flush for area 2
    if (area[1][1] != 0) {

#ifndef CUPROF_RMSYNC_DISABLE
      for (int i = area[1][0]/sizeof(uint64_t);
           i < (area[1][0] + area[1][1])/sizeof(uint64_t);
           i++) {
        while ( !records_h_lu[i] );
      }
#endif
    
      if (! tracefile_write(out, records_h + area[1][0], area[1][1])) {
        fprintf(stderr, "Trace Write Error!\n");
      }

      memset(records_h + area[1][0], 0, area[1][1]);
    
    }



    
    // send signal
    
    // guarantee all work before flush signal to device
    std::atomic_thread_fence(std::memory_order_release);
    cudaChecked(cudaStreamSynchronize(cudastream_trace));

    // flush signal to device
    cudaChecked(cudaMemcpyAsync(flushed_d,
                                &signal,
                                sizeof(uint32_t), cudaMemcpyHostToDevice,
                                cudastream_trace));
    
    
    return;
  }

  // payload function of queue consumer
  static void consume(TraceConsumer* obj) {

    cudaSetDevice(obj->device);
    obj->does_run = true;

    uint8_t* allocs_d = obj->traceinfo.info_d.allocs_d;
    uint8_t* commits_d = obj->traceinfo.info_d.commits_d;
    uint8_t* flusheds_h = obj->traceinfo.flusheds_h;
    uint8_t* flusheds_old = obj->traceinfo.flusheds_old;
    uint8_t* signals_h = obj->traceinfo.signals_h;
    uint8_t* records_d = obj->traceinfo.info_d.records_d;
    uint8_t* records_h = obj->traceinfo.records_h;
    cudaStream_t cudastream_trace = obj->cudastream_trace;

    tracefile_t tracefile = obj->tracefile;

    
    /*
    {
      std::unique_lock<std::mutex> lock_refresh_consume(obj->mtx_refresh_consume);
      obj->cv_refresh_consume.wait(lock_refresh_consume,
                                   [obj](){
                                     return obj->refreshConsumeImmediately();
                                   }); // wait for refresh
    }
    */
    
    uint32_t offset[SLOTS_PER_DEV];
    uint32_t records_offset[SLOTS_PER_DEV];

    for(int slot = 0; slot < SLOTS_PER_DEV; slot++) {
      offset[slot] = slot * CACHELINE;
      records_offset[slot] = slot * SLOT_SIZE;
    }

    //while (!obj->to_be_terminated) {
      
    while(obj->should_run) {
      for(int slot = 0; slot < SLOTS_PER_DEV; slot++) {
        consumeSlot(&allocs_d[offset[slot]], &commits_d[offset[slot]],
                    &signals_h[offset[slot]], &flusheds_h[offset[slot]],
                    &flusheds_old[offset[slot]], &records_d[records_offset[slot]],
                    &records_h[records_offset[slot]],
                    tracefile, true, cudastream_trace);
          
      }
    }

    // after should_run flag has been reset to false, no warps are writing, but
    // there might still be data in the buffers
    for(int slot = 0; slot < SLOTS_PER_DEV; slot++) {
      consumeSlot(&allocs_d[offset[slot]], &commits_d[offset[slot]],
                  &signals_h[offset[slot]], &flusheds_h[offset[slot]],
                  &flusheds_old[offset[slot]], &records_d[records_offset[slot]],
                  &records_h[records_offset[slot]],
                  tracefile, false, cudastream_trace);
    }
    /*
      std::unique_lock<std::mutex> lock_refresh_consume(obj->mtx_refresh_consume);
      obj->cv_refresh_consume.wait(lock_refresh_consume,
      [obj](){
      return obj->refreshConsumeImmediately();
      }); // wait for refresh
    */
    //}

    obj->does_run = false;
    return;
  }

  int device;
  bool header_written;

  std::atomic<bool> should_run;
  std::atomic<bool> does_run;
  //std::atomic<bool> to_be_terminated;

  
  //std::mutex mtx_refresh_consume;
  //std::condition_variable cv_refresh_consume;

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
    initConsumers();
  }

  void initConsumers() {
    if (consumers != nullptr)
      return;
    
    cudaChecked(cudaGetDeviceCount(&device_count));
    consumers = new TraceConsumer*[device_count];
    
    for (int device = 0; device < device_count; device++) {
      consumers[device] = new TraceConsumer(device);
    }
  }

  
  TraceConsumer* getConsumer(int device) {
    if (device >= device_count)
      return nullptr;
    else
      return consumers[device];
  }

  
  virtual ~TraceManager() {
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
  
  void ___cuprof_init() {
    if (!___cuprof_accdat_var) {
      ___cuprof_accdat_var = (char*) malloc(sizeof(char));
    }
  }
  
  void ___cuprof_term() {
    if (___cuprof_accdat_var) {
      free(___cuprof_accdat_var);
      ___cuprof_accdat_var = NULL;
    }
  }

  void ___cuprof_gvsym_set_up(const void* basedat_sym) {
    if (!___cuprof_trace_base_info) {
      ___cuprof_trace_base_info = (traceinfo_t*) basedat_sym;
    }
  }

  void ___cuprof_kernel_set_up(const void* kdata_sym, const void* kid_sym) {

    ___cuprof_kernel_count++;

    // set kernel id for all devices
    int initial_device;
    cudaChecked(cudaGetDevice(&initial_device));
    int dev_count;
    cudaChecked(cudaGetDeviceCount(&dev_count));

    for (int i = 0; i < dev_count; i++) {
      cudaChecked(cudaSetDevice(i));
      cudaChecked(cudaMemcpyToSymbol(kid_sym, &___cuprof_kernel_count, sizeof(uint32_t)));
    }

    cudaChecked(cudaSetDevice(initial_device));
    

    // get kernel access data from device, and append to host-side var
    size_t kdata_size;
    cudaChecked(cudaGetSymbolSize(&kdata_size, kdata_sym));
    
    
    char* var_tmp = (char*) realloc(___cuprof_accdat_var,
                                    ___cuprof_accdat_varlen + kdata_size + 1);
    if (!var_tmp) {
      fprintf(stderr, "Failed to initialize memory access data!\n");
      abort();
    }
    
    
    const char* kdata;
    cudaChecked(cudaMemcpyFromSymbol(var_tmp + ___cuprof_accdat_varlen,
                                     kdata_sym, kdata_size));
    var_tmp[___cuprof_accdat_varlen + kdata_size] = '\0';
    
    ___cuprof_accdat_var = var_tmp;
    ___cuprof_accdat_varlen += kdata_size;
  }


/*
  
  static void ___cuprof_trace_start_callback(cudaStream_t stream, cudaError_t status, void* vargs) {
    kernel_trace_arg_t* vargs_cast = (kernel_trace_arg_t*)vargs;
    int device = vargs_cast->device;
    
    TraceConsumer* consumer = ___cuprof_trace_manager.getConsumer(device);
    consumer->start(stream, vargs_cast->kernel_name,
                    vargs_cast->kernel_grid_dim, vargs_cast->kernel_cta_size);
    free(vargs);
  }

  static void ___cuprof_trace_stop_callback(cudaStream_t stream, cudaError_t status, void* vargs) {
    int* device_ptr = (int*)vargs;
    int device = *device_ptr;
    
    TraceConsumer* consumer = ___cuprof_trace_manager.getConsumer(device);
    consumer->stop(stream);
    
    free(vargs);
  }
  
  
  void ___cuprof_trace_start(int device, cudaStream_t stream, const char* kernel_name,
                            uint64_t grid_dim, uint16_t cta_size) {
    return;
    
    TraceConsumer* consumer = ___cuprof_trace_manager.getConsumer(device);
    kernel_trace_arg_t* arg = (kernel_trace_arg_t*) malloc(sizeof(kernel_trace_arg_t));
    if (arg == nullptr) {
        fprintf(stderr, "Unable to allocate memory!\n");
        abort();
      }
    
    *arg = (kernel_trace_arg_t){kernel_name, grid_dim, cta_size, device};
    cudaChecked(cudaStreamAddCallback(stream, ___cuprof_trace_start_callback, (void*)arg, 0));
  }

  void ___cuprof_trace_stop(int device, cudaStream_t stream) {
    return;
    
    int* arg = (int*) malloc(sizeof(int));
    if (arg == nullptr) {
      fprintf(stderr, "Unable to allocate memory!\n");
      abort();
    }
    
    *arg = device;
    cudaChecked(cudaStreamAddCallback(stream, ___cuprof_trace_stop_callback, (void*)arg, 0));
  }
*/
}
