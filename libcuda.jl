# Julia Wrapper for libcuda.so (tested against version 4.2)
# USAGE: 
#  You must call cuInit() first
#  to get a list of device call jlcuDeviceList()

# references
# http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/index.html
# http://www.dmi.unict.it/~bilotta/gpgpu/notes/10-driver.html

libcuda = dlopen("libcuda")
#libtestcuda = dlopen("libtestcuda")

typealias CUdevice Int32
typealias CUcontext Ptr{Void}
typealias CUmodule Ptr{Void}
typealias CUfunction Ptr{Void}
typealias CUdeviceptr Ptr{Void} 
typealias CUevent Ptr{Void}
typealias CUstream Ptr{Void}

typealias CUfunc_cache  Uint32 # enum
  const CU_FUNC_CACHE_PREFER_NONE   = 0; # no preference for shared memory or L1 cache size
  const CU_FUNC_CACHE_PREFER_SHARED = 1; # prefer larger shared memory and smaller L1
  const CU_FUNC_CACHE_PREFER_L1     = 2; # prefer smaller shared memory and larger L1
  const CU_FUNC_CACHE_PREFER_EQUAL  = 4; # prefer equal shared memory and L1 size

typealias CUlimit Uint32 # enum
  const CU_LIMIT_STACK_SIZE = 0;
  const CU_LIMIT_PRINTF_FIFO_SIZE = 1;
  const CU_LIMIT_MALLOC_HEAP_SIZE = 2;
 
typealias CUsurfref Ptr{Void};

typealias CUtexref Ptr{Void};

typealias CUsize_t Uint #64bit / 32bit

typealias CUresult Int32
  const CUresult_success  = 0
  const CUresult_invalid_value = 1
  const CUresult_out_of_memory = 2
  const CUresult_not_initialized = 3
  const CUresult_invalid_image = 200 
  const CUresult_invalid_context = 201
  const CUresult_context_already_current = 202
  const CUresult_map_failed = 205
  const CUresult_unmap_failed = 206
  const CUresult_array_is_mapped = 207
  const CUresult_already_mapped = 208
  const CUresult_no_binary_for_gpu = 209
  const CUresult_invalid_handle = 400
  

typealias CUjit_option Uint32;
  const CU_JIT_MAX_REGISTERS = 0;         #Uint32   #(Input) Maximum number of registers each thread may use #(Output) NA
  const CU_JIT_THREADS_PER_BLOCK = 1;     #Uint32   #(Input) minimum number of threads per block             #(Output) Actual number compiler came up with
  const CU_JIT_WALL_TIME = 2;             #Float32  #(Input) NA                                              #(Output) time spent compiling
  const CU_JIT_INFO_LOG_BUFFER = 3;       #char*    #(Input) Pointer to buffer for log message storage       #(Output) NA
  const CU_JIT_INFO_LOG_BUFFER_SIZE = 4;  #Uint32   #(Input) Size of Log Buffer in bytes                     #(Output) Bytes actually used
  const CU_JIT_ERROR_LOG_BUFFER = 5;      #char*    #(Input) Pointer to buffer for error message storage     #(Output) NA
  const CU_JIT_ERROR_LOG_BUFFER_SIZE = 6; #Uint32   #(Input) Size of Error Buffer in bytes                   #(Output) Bytes actually used
  const CU_JIT_OPTIMIZATION_LEVEL = 7;    #Uint32   #(Input) Level of optimize wanted 4 is max and default   #(Output) NA
  const CU_JIT_TARGET_FROM_CUCONTEXT = 8; #NONE     #(Input) NA  	                                     #(Output) NA
  const CU_JIT_TARGET = 9;                #Uint32   #(Input) Choose target based on this input               #(Output) NA
    const CU_TARGET_COMPUTE_10 = 0;
    const CU_TARGET_COMPUTE_11 = 1;
    const CU_TARGET_COMPUTE_12 = 2;
    const CU_TARGET_COMPUTE_13 = 3;
    const CU_TARGET_COMPUTE_20 = 4;
    const CU_TARGET_COMPUTE_21 = 5;
    const CU_TARGET_COMPUTE_30 = 6;
  const CU_JIT_FALLBACK_STRATEGY = 10;    #Uint32   #(Input) Choose fallback stratagy based on parameter     #(Output) NA 
    const CU_PREFER_PTX     = 0;
    const CU_PREFER_BINARY  = 1;

# ID => (in type, out type)
#TODO CUDA expects individual values to be passed (Ints,Floats) as pointers, for now this is explicitly encoded here, at some point this should be automatic...
const CUjit_option_c_types = 
{ CU_JIT_MAX_REGISTERS         => (Uint32,())
  CU_JIT_THREADS_PER_BLOCK     => (Uint32,Uint32)
  CU_JIT_WALL_TIME             => ((),Float32)
  CU_JIT_INFO_LOG_BUFFER       => (Ptr{Uint8},Ptr{Uint8})
  CU_JIT_INFO_LOG_BUFFER_SIZE  => (Uint32,Uint32)
  CU_JIT_ERROR_LOG_BUFFER      => (Ptr{Uint8},Ptr{Uint8})
  CU_JIT_ERROR_LOG_BUFFER_SIZE => (Uint32,Uint32)
  CU_JIT_OPTIMIZATION_LEVEL    => (Uint32,())
  CU_JIT_TARGET_FROM_CUCONTEXT => ((),())
  CU_JIT_TARGET                => (Uint32,())
  CU_JIT_FALLBACK_STRATEGY     => (Uint32,())
}

const CUjit_option_julia_types = 
{ CU_JIT_MAX_REGISTERS         => (Uint32,())
  CU_JIT_THREADS_PER_BLOCK     => (Uint32,Uint32)
  CU_JIT_WALL_TIME             => ((),Float32) 
  CU_JIT_INFO_LOG_BUFFER       => (Array{Uint8,1},ASCIIString) 
  CU_JIT_INFO_LOG_BUFFER_SIZE  => (Uint32,Uint32) 
  CU_JIT_ERROR_LOG_BUFFER      => (Array{Uint8,1},ASCIIString)
  CU_JIT_ERROR_LOG_BUFFER_SIZE => (Uint32,Uint32)
  CU_JIT_OPTIMIZATION_LEVEL    => (Uint32,())
  CU_JIT_TARGET_FROM_CUCONTEXT => ((),())
  CU_JIT_TARGET                => (Uint32,())
  CU_JIT_FALLBACK_STRATEGY     => (Uint32,())
}

## custom functions

# give useful string for errors
#TODO Make into dict
function jlcuCheck(v)
  if CUresult_success == v
  elseif CUresult_invalid_value == v
    error("CUDA Call Error($v): Invalid Value")
  elseif CUresult_out_of_memory == v
    error("CUDA Call Error($v): Out Of Memory")
  elseif CUresult_not_initialized == v
    error("CUDA Call Error($v): Not Initialized")
  elseif CUresult_invalid_image == v
    error("CUDA Call Error($v): Invalid Image")
  elseif CUresult_invalid_context == v
    error("CUDA Call Error($v): Invalid Context")
  elseif CUresult_context_already_current == v
    error("CUDA Call Error($v): Context Already Current")
  elseif CUresult_map_failed == v
    error("CUDA Call Error($v): Context Map Failed")
  elseif CUresult_unmap_failed == v
    error("CUDA Call Error($v): Context Unmap Failed")
  elseif CUresult_array_is_mapped == v
    error("CUDA Call Error($v): Array Is Mapped")
  elseif CUresult_already_mapped == v
    error("CUDA Call Error($v): Already Mapped")
  elseif CUresult_no_binary_for_gpu == v
    error("CUDA Call Error($v): No Binary For GPU")
  elseif CUresult_invalid_handle == v
    error("CUDA Call Error($v): Invalid Handle")
  else
    error("CUDA Call Error($v) #### NOT TRANSLATED ####")
  end
  return v
end

# list avaiable cuda devices
function jlcuDeviceList()
  n = cuDeviceGetCount()
  for i = 0:(n-1)
    hnd = cuDeviceGet(i)
    name = cuDeviceGetName(hnd)
    ccap = cuDeviceComputeCapability(hnd)
    mem = cuDeviceTotalMem(hnd)
    mem /= 1024.0;
    mem /= 1024.0;
    println("$hnd : $name : CC = $ccap : Mem (MiB) = $mem")
  end

  return n
end


## Driver function

# call to cuInit() not needed before calling cuDriverGetVersion
function cuDriverGetVersion()
  #parameter is actually a pointer to Uint, but we use int to display human readable value
  version = Array(Int32,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDriverGetVersion),CUresult,(Ptr{Int32},), version))
  return version[1]
end

const CUDA_API_VERSION = cuDriverGetVersion();



## Library functions

function cuInit(flags)
  jlcuCheck(ccall(dlsym(libcuda,:cuInit),CUresult,(Uint,), flags))
end

cuInit() = cuInit(0)



## Device Management

function cuDeviceGetCount()
  deviceCount = Array(Int32,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceGetCount),CUresult,(Ptr{Int32},), deviceCount))
  return deviceCount[1]
end

function cuDeviceComputeCapability(deviceHandle)
  major = Array(Int32,1)
  minor = Array(Int32,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceComputeCapability),CUresult,(Ptr{Int32},Ptr{Int32},Int32),major,minor,deviceHandle))
  return (major[1],minor[1])
end

function cuDeviceGet(index)
  deviceHandle = Array(CUdevice,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceGet),CUresult,(Ptr{CUdevice},Int32),deviceHandle,index))
  return deviceHandle[1]
end

#TODO implement cuDeviceGetAttibute, needs lots of enums defined...
#TODO implement cuDeviceGetProperties, needs a structure defined...

function cuDeviceGetName(hdev::CUdevice)
  deviceName = Array(Uint8, 128)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceGetName),CUresult,(Ptr{Uint8},Int32,CUdevice),deviceName, length(deviceName),hdev))
  return bytestring(convert(Ptr{Uint8},deviceName))
end

function cuDeviceTotalMem(hdev::CUdevice)
  sizeBytes = Array(Uint64,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceTotalMem),CUresult,(Ptr{Uint64},Int32),sizeBytes,hdev))
  return sizeBytes[1]
end

## Context Management

if CUDA_API_VERSION >= 3020
  function cuCtxCreate(flags,hdev)
    context = Array(CUcontext,1)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxCreate_v2),CUresult,(Ptr{CUcontext},Uint32,CUdevice),context,flags,hdev))
    return context[1]
  end
else
  function cuCtxCreate(flags,hdev)
    context = Array(CUcontext,1)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxCreate),CUresult,(Ptr{CUcontext},Uint32,CUdevice),context,flags,hdev))
    return context[1]
  end
end

if CUDA_API_VERSION >= 4000
  function cuCtxDestroy(context::CUcontext)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxDestroy_v2),CUresult,(CUcontext,),context))
    return ()
  end
else
  function cuCtxDestroy(context::CUcontext)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxDestroy),CUresult,(CUcontext,),context))
    return ()
  end
end

function cuCtxGetApiVersion(context::CUcontext)
  version = Array(Uint32,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxGetApiVersion),CUresult,(CUcontext,Ptr{Uint32}),context,version))
  return version[1]
end

function cuCtxGetCacheConfig()
  config = Array(CUfunc_cache,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxGetCacheConfig),CUresult,(Ptr{CUfunc_cache},),config))
  return config[1]
end

function cuCtxGetDevice()
  device = Array(CUdevice,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxGetDevice),CUresult,(Ptr{CUdevice},),device))
  return device[1]
end

function cuCtxGetLimit(limit::CUlimit)
  value = Array(CUsize_t,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxGetLimit),CUresult,(Ptr{CUsize_t},CUlimit),value,limit))
  return value[1]
end

if CUDA_API_VERSION >= 4000
  function cuCtxPopCurrent()
    ctx = Array(CUcontext,1)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxPopCurrent_v2),CUresult,(Ptr{CUcontext},),ctx))
    return ctx[1]
  end
else
 function cuCtxPopCurrent()
    ctx = Array(CUcontext,1)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxPopCurrent),CUresult,(Ptr{CUcontext},),ctx))
    return ctx[1]
  end
end

if CUDA_API_VERSION >= 4000
  function cuCtxPushCurrent(ctx::CUcontext)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxPushCurrenti_v2),CUresult,(CUcontext,),ctx))
  end
else
  function cuCtxPushCurrent(ctx::CUcontext)
    jlcuCheck(ccall(dlsym(libcuda,:cuCtxPushCurrent),CUresult,(CUcontext,),ctx))
  end
end

function cuCtxSetCacheConfig(config::CUfunc_cache)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxSetCacheConfig),CUresult,(CUfunc_cache,),config))
end

function cuCtxSetCurrent(ctx::CUcontext)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxSetCurrent),CUresult,(CUcontext,),ctx))
end

function cuCtxSetLimit(limit::CUlimit,value::CUsize_t)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxSetLimit),CUresult,(CUlimit,CUsize_t),limit,value))
end

function cuCtxSynchronize()
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxSynchronize),CUresult,()))
end

### Module Management
function cuModuleGetFunction(hmod::CUmodule,name::String)
  func = Array(CUfunction,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleGetFunction),CUresult,(Ptr{CUfunction},CUmodule,Ptr{Uint8}),func,hmod,bytestring(name)))
  return func[1]
end

if CUDA_API_VERSION >= 3020
  function cuModuleGetGlobal(hmod::CUmodule,name::String)
    dptr = Array(CUdeviceptr,1);
    bytes = Array(CUsize_t,1);
    jlcuCheck(ccall(dlsym(libcuda,:cuModuleGetGlobal_v2),CUresult,(Ptr{CUdeviceptr},Ptr{CUsize_t},CUmodule,Ptr{Uint8}),dptr,bytes,hmod,bytestring(name)))
    return (dptr[1],bytes[1])
  end
else
  function cuModuleGetGlobal(hmod::CUmodule,name::String)
    dptr = Array(CUdeviceptr,1);
    bytes = Array(CUsize_t,1);
    jlcuCheck(ccall(dlsym(libcuda,:cuModuleGetGlobal),CUresult,(Ptr{CUdeviceptr},Ptr{CUsize_t},CUmodule,Ptr{Uint8}),dptr,bytes,hmod,bytestring(name)))
    return (dptr[1],bytes[1])
  end
end

function cuModuleGetSurfRef(hmod::CUmodule,name::String)
  surfRef = Array(CUsurfref,1);
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleGetSurfRef),CUresult,(Ptr{CUsurfref},CUmodule,Ptr{Uint8}),surfRef,hmod,bytestring(name)))
  return surfRef[1]
end

function cuModuleGetTexRef(hmod::CUmodule,name::String)
  texRef = Array(CUtexref,1);
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleGetTexRef),CUresult,(Ptr{CUtexref},CUmodule,Ptr{Uint8}),texRef,hmod,bytestring(name)))
  return texRef[1]
end

function cuModuleLoad(fname::String)
  hmod = Array(CUmodule,1);
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleLoad),CUresult,(Ptr{CUmodule},Ptr{Uint8}),hmod,bytestring(name)))
  return hmod[1]
end

function cuModuleLoadData_base(image)
  hmod = Array(CUmodule,1);
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleLoadData),CUresult,(Ptr{CUmodule},Ptr{Uint8}),hmod,image))
  return hmod[1]
end

function cuModuleLoadData(image::String)
  return cuModuleLoadData_base(bytestring(image))
end

function cuModuleLoadData(image::Array{Uint8})
  return cuModuleLoadData_base(image)
end


# reference http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MODULE_g9e8047e9dbf725f0cd7cafd18bfd4d12.html
function cuModuleLoadDataEx_base(image,options...)
  hmod = Array(CUmodule,1);
  hmod[1] = 0;
  # build options ids and values storage
  optionIds = Array(CUjit_option,0);
  optionValues = Array(Any,0);
  optionValuePtrs = Array(Ptr{Void},0);

  local idx = 1;
  while idx <= numel(options)
    option = options[idx];
#    println("Param: $idx $option")
    if !has(CUjit_option_julia_types,options[idx])
      error("Unknown Options: ",option);
    end

    (optionJuliaTypeIn,optionJuliaTypeOut) = CUjit_option_julia_types[option];
    (optionCTypeIn,optionCTypeOut) = CUjit_option_c_types[option];

    push(optionIds,option);

    if optionJuliaTypeIn == ()
      push(optionValues,());
      push(optionValuePtrs,0);
    elseif idx >= numel(options)
      error("Missing Value after: ",option);
    else
      idx += 1;
      optionValue = options[idx];
      optionValue = convert(optionJuliaTypeIn,optionValue);
      push(optionValues,optionValue);
      push(optionValuePtrs,convert(optionCTypeIn,optionValues[end]));
    end
    idx += 1;
  end 
  numOptions = numel(optionIds)

#  println("image: $(typeof(image)) $image")
#  println("numOptions: $numOptions")
#  println("optionIds: $optionIds")
#  println("optionValues: $optionValues")
#  println("optionValuePtrs: $optionValuePtrs")
#  for i = 1:numel(optionIds)
#    println("$i : $(optionIds[i]) , $(optionValues[i]) , $(optionValuePtrs[i])")
#  end

  result = ccall(dlsym(libcuda,:cuModuleLoadDataEx),CUresult,(Ptr{CUmodule},Ptr{Uint8},Uint32,Ptr{CUjit_option},Ptr{Ptr{Void}}),hmod,image,numOptions,optionIds,optionValuePtrs)

#  println("After Call: $(numel(optionIds))")
  optionValuesOut = Array(Any,numel(optionIds))
  for idx = 1:numel(optionIds)
#    println("$idx : $(optionIds[idx]) , $(optionValuePtrs[idx])")
    option = optionIds[idx];  
    (optionJuliaTypeIn,optionJuliaTypeOut) = CUjit_option_julia_types[option];
    (optionCTypeIn,optionCTypeOut) = CUjit_option_c_types[option];
    
    if optionJuliaTypeOut == ()
      optionValuesOut[idx] = ()
    elseif optionJuliaTypeOut == Float32
      optionValuesOut[idx] = reinterpret(Float32,convert(Uint32,optionValuePtrs[idx]))
    elseif optionJuliaTypeOut == ASCIIString
      optionValuesOut[idx] = convert(ASCIIString,optionValues[idx])
    else
      optionValuesOut[idx] = convert(optionJuliaTypeOut,optionValuePtrs[idx])
    end
#    println("Value: $(optionValuesOut[idx])")
  end

  return (result,hmod[1],optionIds,optionValuesOut)
end

function cuModuleLoadDataEx(image::String,options...)
  return cuModuleLoadDataEx_base(bytestring(image),options...)
end

function cuModuleLoadData(image::Array{Uint8},options...)
  return cuModuleLoadDataEx_base(image,options...)
end

function cuModuleLoadFatBinary(image::Array{Uint8})
  hmod = Array(CUmodule,1);
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleLoadFatBinary),CUresult,(Ptr{CUmodule},Ptr{Uint8}),hmod,image))
  return hmod[1]
end

function cuModuleUnload(hmod::CUmodule)
  jlcuCheck(ccall(dlsym(libcuda,:cuModuleUnload),CUresult,(CUmodule,),hmod))
end


### Memory Management
cuMemGetInfo_symbol = :cuMemGetInfo;
cuMemAlloc_symbol = :cuMemAlloc;
cuMemFree_symbol = :cuMemFree;
cuMemcpyHtoD_symbol = :cuMemcpyHtoD;
cuMemcpyDtoH_symbol = :cuMemcpyDtoH;
cuMemcpyDtoD_symbol = :cuMemcpyDtoD;
cuMemcpyHtoDAsync_symbol = :cuMemcpyHtoDAsync;
cuMemcpyDtoHAsync_symbol = :cuMemcpyDtoHAsync;
cuMemcpyDtoDAsync_symbol = :cuMemcpyDtoDAsync;
cuMemsetD8_symbol = :cuMemsetD8;
cuMemsetD8Async_symbol = :cuMemsetD8Async;
cuMemsetD16_symbol = :cuMemsetD16;
cuMemsetD16Async_symbol = :cuMemsetD16Async;
cuMemsetD32_symbol = :cuMemsetD32;
cuMemsetD32Async_symbol = :cuMemsetD32Async;

if CUDA_API_VERSION >= 3020
  cuMemGetInfo_symbol = :cuMemGetInfo_v2;
  cuMemAlloc_symbol = :cuMemAlloc_v2;
  cuMemFree_symbol = :cuMemFree_v2;
  cuMemcpyHtoD_symbol = :cuMemcpyHtoD_v2;
  cuMemcpyDtoH_symbol = :cuMemcpyDtoH_v2;
  cuMemcpyDtoD_symbol = :cuMemcpyDtoD_v2;
  cuMemcpyHtoDAsync_symbol = :cuMemcpyHtoDAsync_v2;
  cuMemcpyDtoHAsync_symbol = :cuMemcpyDtoHAsync_v2;
  cuMemcpyDtoDAsync_symbol = :cuMemcpyDtoDAsync_v2;
  cuMemsetD8_symbol = :cuMemsetD8_v2;
  cuMemsetD16_symbol = :cuMemsetD16_v2;
  cuMemsetD32_symbol = :cuMemsetD32_v2;
end


@eval function cuMemGetInfo()
  freeD = Array(CUsize_t,1);
  totalD = Array(CUsize_t,1);
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemGetInfo_symbol)),CUresult,(Ptr{CUsize_t},Ptr{CUsize_t}),freeD,totalD))
  return (freeD[1],totalD[1])
end

@eval function cuMemAlloc(byteSize)
  dptr = Array(CUdeviceptr,1);
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemAlloc_symbol)),CUresult,(Ptr{CUdeviceptr},CUsize_t),dptr,byteSize))
  return dptr[1]
end

@eval function cuMemFree(dptr)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemFree_symbol)),CUresult,(CUdeviceptr,),dptr))
end

@eval function cuMemcpyDtoD(dstDevice,srcDevice,byteCount)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemcpyDtoD_symbol)),CUresult,(CUdeviceptr,CUdeviceptr,CUsize_t),dstDevice,srcDevice,byteCount))
end

@eval function cuMemcpyDtoDAsync(dstDevice,srcDevice,byteCount,hStream) 
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemcpyDtoDAsync_symbol)),CUresult,(CUdeviceptr,CUdeviceptr,CUsize_t,CUstream),dstDevice,srcDevice,byteCount,hStream))
end

@eval function cuMemcpyDtoH(dstHost,srcDevice,byteCount)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemcpyDtoH_symbol)),CUresult,(Ptr{Void},CUdeviceptr,CUsize_t),dstHost,srcDevice,byteCount))
end

@eval function cuMemcpyDtoHAsync(dstHost,srcDevice,byteCount,hStream)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemcpyDtoHAsync_symbol)),CUresult,(Ptr{Void},CUdeviceptr,CUsize_t,CUstream),dstHost,srcDevice,byteCount,hStream))
end

@eval function cuMemcpyHtoD(dstDevice,srcHost,byteCount)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemcpyHtoD_symbol)),CUresult,(CUdeviceptr,Ptr{Void},CUsize_t),dstDevice,srcHost,byteCount))
end

@eval function cuMemcpyHtoDAsync(dstDevice,srcHost,byteCount,hStream)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemcpyHtoDAsync_symbol)),CUresult,(CUdeviceptr,Ptr{Void},CUsize_t,CUstream),dstDevice,srcHost,byteCount,hStream))
end


@eval function cuMemsetD8(dstDevice,val,count)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemsetD8_symbol)),CUresult,(CUdeviceptr,Uint8,CUsize_t),dstDevice,val,count))
end

@eval function cuMemcpyD8Async(dstDevice,val,count,hStream)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemsetD8Async_symbol)),CUresult,(CUdeviceptr,Uint8,CUsize_t,CUstream),dstDevice,val,count,hStream))
end

@eval function cuMemsetD16(dstDevice,val,count)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemsetD16_symbol)),CUresult,(CUdeviceptr,Uint16,CUsize_t),dstDevice,val,count))
end

@eval function cuMemcpyD16Async(dstDevice,val,count,hStream)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemsetD16Async_symbol)),CUresult,(CUdeviceptr,Uint16,CUsize_t,CUstream),dstDevice,val,count,hStream))
end

@eval function cuMemsetD32(dstDevice,val,count)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemsetD32_symbol)),CUresult,(CUdeviceptr,Uint32,CUsize_t),dstDevice,val,count))
end

@eval function cuMemcpyD32Async(dstDevice,val,count,hStream)
  jlcuCheck(ccall(dlsym(libcuda,:($cuMemsetD32Async_symbol)),CUresult,(CUdeviceptr,Uint32,CUsize_t,CUstream),dstDevice,val,count,hStream))
end


#CUresult       cuArray3DCreate (CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
#       Creates a 3D CUDA array.
#CUresult       cuArray3DGetDescriptor (CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
#       Get a 3D CUDA array descriptor.
#CUresult       cuArrayCreate (CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
#       Creates a 1D or 2D CUDA array.
#CUresult       cuArrayDestroy (CUarray hArray)
#       Destroys a CUDA array.
#CUresult       cuArrayGetDescriptor (CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
#       Get a 1D or 2D CUDA array descriptor.
#CUresult       cuDeviceGetByPCIBusId (CUdevice *dev, char *pciBusId)
#       Returns a handle to a compute device.
#CUresult       cuDeviceGetPCIBusId (char *pciBusId, int len, CUdevice dev)
#       Returns a PCI Bus Id string for the device.
#CUresult       cuIpcCloseMemHandle (CUdeviceptr dptr)
#CUresult       cuIpcGetEventHandle (CUipcEventHandle *pHandle, CUevent event)
#       Gets an interprocess handle for a previously allocated event.
#CUresult       cuIpcGetMemHandle (CUipcMemHandle *pHandle, CUdeviceptr dptr)
#CUresult       cuIpcOpenEventHandle (CUevent *phEvent, CUipcEventHandle handle)
#       Opens an interprocess event handle for use in the current process.
#CUresult       cuIpcOpenMemHandle (CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)
#CUresult 	cuMemAllocHost (void **pp, size_t bytesize)
# 	Allocates page-locked host memory.
#CUresult 	cuMemAllocPitch (CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
# 	Allocates pitched device memory.
#CUresult 	cuMemcpy (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
# 	Copies memory.
#CUresult 	cuMemcpy2D (const CUDA_MEMCPY2D *pCopy)
# 	Copies memory for 2D arrays.
#CUresult 	cuMemcpy2DAsync (const CUDA_MEMCPY2D *pCopy, CUstream hStream)
# 	Copies memory for 2D arrays.
#CUresult 	cuMemcpy2DUnaligned (const CUDA_MEMCPY2D *pCopy)
# 	Copies memory for 2D arrays.
#CUresult 	cuMemcpy3D (const CUDA_MEMCPY3D *pCopy)
# 	Copies memory for 3D arrays.
#CUresult 	cuMemcpy3DAsync (const CUDA_MEMCPY3D *pCopy, CUstream hStream)
# 	Copies memory for 3D arrays.
#CUresult 	cuMemcpy3DPeer (const CUDA_MEMCPY3D_PEER *pCopy)
# 	Copies memory between contexts.
#CUresult 	cuMemcpy3DPeerAsync (const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream)
# 	Copies memory between contexts asynchronously.
#CUresult 	cuMemcpyAsync (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
# 	Copies memory asynchronously.
#CUresult 	cuMemcpyAtoA (CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
# 	Copies memory from Array to Array.
#CUresult 	cuMemcpyAtoD (CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
# 	Copies memory from Array to Device.
#CUresult 	cuMemcpyAtoH (void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
# 	Copies memory from Array to Host.
#CUresult 	cuMemcpyAtoHAsync (void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
# 	Copies memory from Array to Host.
#CUresult 	cuMemcpyDtoA (CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
# 	Copies memory from Device to Array.
#CUresult       cuMemcpyHtoA (CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
#       Copies memory from Host to Array.
#CUresult       cuMemcpyHtoAAsync (CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream)
#       Copies memory from Host to Array.
#CUresult 	cuMemcpyPeer (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
# 	Copies device memory between two contexts.
#CUresult 	cuMemcpyPeerAsync (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
# 	Copies device memory between two contexts asynchronously.
#CUresult 	cuMemFreeHost (void *p)
# 	Frees page-locked host memory.
#CUresult 	cuMemGetAddressRange (CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
# 	Get information on memory allocations.
#CUresult 	cuMemGetInfo (size_t *free, size_t *total)
# 	Gets free and total memory.
#CUresult 	cuMemHostAlloc (void **pp, size_t bytesize, unsigned int Flags)
# 	Allocates page-locked host memory.
#CUresult 	cuMemHostGetDevicePointer (CUdeviceptr *pdptr, void *p, unsigned int Flags)
# 	Passes back device pointer of mapped pinned memory.
#CUresult 	cuMemHostGetFlags (unsigned int *pFlags, void *p)
# 	Passes back flags that were used for a pinned allocation.
#CUresult 	cuMemHostRegister (void *p, size_t bytesize, unsigned int Flags)
# 	Registers an existing host memory range for use by CUDA.
#CUresult 	cuMemHostUnregister (void *p)
# 	Unregisters a memory range that was registered with cuMemHostRegister().
#CUresult 	cuMemsetD16 (CUdeviceptr dstDevice, unsigned short us, size_t N)
# 	Initializes device memory.
#CUresult 	cuMemsetD16Async (CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
# 	Sets device memory.
#CUresult 	cuMemsetD2D16 (CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
# 	Initializes device memory.
#CUresult 	cuMemsetD2D16Async (CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
# 	Sets device memory.
#CUresult 	cuMemsetD2D32 (CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
# 	Initializes device memory.
#CUresult 	cuMemsetD2D32Async (CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
# 	Sets device memory.
#CUresult 	cuMemsetD2D8 (CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
# 	Initializes device memory.
#CUresult 	cuMemsetD2D8Async (CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
# 	Sets device memory.
#CUresult 	cuMemsetD32 (CUdeviceptr dstDevice, unsigned int ui, size_t N)
# 	Initializes device memory.
#CUresult 	cuMemsetD32Async (CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
# 	Sets device memory.


### Stream Management
#CUresult 	cuStreamCreate (CUstream *phStream, unsigned int Flags)
# 	Create a stream.
#CUresult 	cuStreamDestroy (CUstream hStream)
# 	Destroys a stream.
#CUresult 	cuStreamQuery (CUstream hStream)
# 	Determine status of a compute stream.

#CUresult 	cuStreamSynchronize (CUstream hStream)
# 	Wait until a stream's tasks are completed.
@eval function cuStreamSynchronize(hStream)
  jlcuCheck(ccall(dlsym(libcuda,:cuStreamSynchronize),CUresult,(CUstream,),hStream))
end

#CUresult 	cuStreamWaitEvent (CUstream hStream, CUevent hEvent, unsigned int Flags)
# 	Make a compute stream wait on an event. 
#
### Event Management
#CUresult 	cuEventCreate (CUevent *phEvent, unsigned int Flags)
# 	Creates an event.
#CUresult 	cuEventDestroy (CUevent hEvent)
# 	Destroys an event.
#CUresult 	cuEventElapsedTime (float *pMilliseconds, CUevent hStart, CUevent hEnd)
# 	Computes the elapsed time between two events.
#CUresult 	cuEventQuery (CUevent hEvent)
# 	Queries an event's status.
#CUresult 	cuEventRecord (CUevent hEvent, CUstream hStream)
# 	Records an event.
#CUresult 	cuEventSynchronize (CUevent hEvent)
# 	Waits for an event to complete. 
#

### Execution Control
function cuFuncSetCacheConfig (hfunc::CUfunction,config::CUfunc_cache)
  jlcuCheck(ccall(dlsym(libcuda,:cuFuncSetCacheConfig),CUresult,(CUfunction,CUfunc_cache),hfunc,config))
end

# reference http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__EXEC_gb8f3dc3031b40da29d5f9a7139e52e15.html#gb8f3dc3031b40da29d5f9a7139e52e15
function cuLaunchKernel(
	f::CUfunction,
  	gridDimX::Uint32,  gridDimY::Uint32, gridDimZ::Uint32, 
  	blockDimX::Uint32, blockDimY::Uint32, blockDimZ::Uint32,
  	sharedMemBytes::Uint32, hStream::CUstream,
  	kernelParams, extra)
  jlcuCheck(ccall(dlsym(libcuda,:cuLaunchKernel),CUresult,(CUfunction,Uint32,Uint32,Uint32,Uint32,Uint32,Uint32,Uint32,CUstream,Ptr{Ptr{Void}},Ptr{Ptr{Void}}),
        f,
  	gridDimX,gridDimY,gridDimZ,
        blockDimX,blockDimY,blockDimZ,
        sharedMemBytes,
        hStream,
        kernelParams,
        extra))
end
  
# 	Launches a CUDA function. 
#
#CUresult       cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
#       Returns information about a function.



function cudaTest()
  global ctxt = cuCtxCreate(convert(Uint32,0),convert(Int32,0));
  (freeD,totalD) = cuMemGetInfo(); println("Free: ",freeD," Total: ",totalD);

  N = 128*1024*1024;
  @time buffer = cuMemAlloc(N * 4);
  (freeD,totalD) = cuMemGetInfo(); println("Free: ",freeD," Total: ",totalD);

  tt = @elapsed a = Array(Float32,N);
  println("time: ",tt," rate(MB/s): ",(N*4.0)/(1024.0^2)/tt)
  tt = @elapsed a[:] = randn(N);
  println("time: ",tt," rate(MB/s): ",(N*4.0)/(1024.0^2)/tt)
  tt = @elapsed b = Array(Float32,N);
  println("time: ",tt," rate(MB/s): ",(N*4.0)/(1024.0^2)/tt)
  tt = @elapsed b[:] = 0;
  println("time: ",tt," rate(MB/s): ",(N*4.0)/(1024.0^2)/tt)

  println("--------------");
  
  println("a -> GPU: start: ",a[1:10]);
  tt = @elapsed cuMemcpyHtoD(buffer,a,N*4);
  println("Transfer time: ",tt," rate: ",(N * 4.0)/(1024.0 * 1024.0)/tt," MB/s");
  println("a -> GPU: done: ",a[1:10]);

  println("--------------");

  println("GPU -> b: start: ",b[1:10]);
  tt = @elapsed cuMemcpyDtoH(b,buffer,N*4);
  println("Transfer time: ",tt," rate: ",(N * 4.0)/(1024.0 * 1024.0)/tt," MB/s");
  println("GPU -> b: done: ",b[1:10]);
  
  println("--------------");

  cuMemFree(buffer)
  (freeD,totalD) = cuMemGetInfo(); println("Free: ",freeD," Total: ",totalD);
  
  cuCtxDestroy(ctxt)

  tt = @elapsed  matchRes = all(a == b);
  println(" a == b: ",matchRes)
  println("time: ",tt," rate(MB/s): ",(N*4.0)/(1024.0^2)/tt)

  println("--------------");
end
