# Julia Wrapper for libcuda.so (tested against version 4.2)
# USAGE: 
#  You must call cuInit() first
#  to get a list of device call jlcuDeviceList()

# references
# http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/index.html
# http://www.dmi.unict.it/~bilotta/gpgpu/notes/10-driver.html

libcuda = dlopen("libcuda")

#libcuda_cuInitSym = dlsym(libcuda,:cuInit)

typealias CUdevice Int32
typealias CUcontext Ptr{Void}
typealias CUmodule Ptr{Void}
typealias CUfunction Ptr{Void}
typealias CUdeviceptr Uint32
typealias CUevent Ptr{Void}
typealias CUstream Ptr{Void}
typealias CUfunc_cache  Int32 # enum
typealias CUlimit Int32 # enum 
typealias CUsurfref Ptr{Void};
typealias CUtexref Ptr{Void};

typealias CUsize_t Uint32

typealias CUresult Int32
const CUresult_success  = 0
const CUresult_invalid_value = 1
const CUresult_out_of_memory = 2
const CUresult_not_initialized = 3

## custom functions

# give useful string for errors
function jlcuCheck(v)
  if CUresult_success == v
  elseif CUresult_invalid_value == v
    error("CUDA Call Error($v): Invalid Value")
  elseif CUresult_out_of_memory == v
    error("CUDA Call Error($v): Out Of Memory")
  elseif CUresult_not_initialized == v
    error("CUDA Call Error($v): Not Initialized")
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
end

## Library functions

function cuInit(flags)
  jlcuCheck(ccall(dlsym(libcuda,:cuInit),CUresult,(Uint,), flags))
end

cuInit() = cuInit(0)

## Driver function

# call to cuInit() not needed before calling cuDriverGetVersion
function cuDriverGetVersion()
  #parameter is actually a pointer to Uint, but we use int to display human readable value
  version = Array(Int32,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDriverGetVersion),CUresult,(Ptr{Int32},), version))
  return version[1]
end

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
  return cstring(convert(Ptr{Uint8}, deviceName))
end

function cuDeviceTotalMem(hdev::CUdevice)
  sizeBytes = Array(Uint64,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceTotalMem),CUresult,(Ptr{Uint64},Int32),sizeBytes,hdev))
  return sizeBytes[1]
end

## Context Management

function cuCtxCreate(flags::Uint32,hdev::CUdevice)
  context = Array(CUcontext,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxCreate),CUresult,(Ptr{CUcontext},Uint32,CUdevice),context,flags,hdev))
  return context[1]
end

function cuCtxDestroy(context::CUcontext)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxDestroy),CUresult,(CUcontext,),context))
  return ()
end

function cuCtxGetApiVersion(context::CUcontext)
  version = Array(Uint32,1)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxGetApiVersion,CUresult,(CUcontext,Ptr{Uint32}),context,version)))
  return version[1]
end

function cuCtxGetCacheConfig()
  config = Array(CUfunc_cache,1)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxGetCacheConfig,CUresult,(Ptr{CUfunc_cache},),config)))
  return config[1]
end

function cuCtxGetDevice()
  device = Array(CUdevice,1)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxGetDevice,CUresult,(Ptr{CUdevice},),device)))
  return device[1]
end

function cuCtxGetLimit(limit::CUlimit)
  value = Array(CUsize_t,1)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxGetLimit,CUresult,(Ptr{CUsize_t},CUlimit),value,limit)))
  return value[1]
end

function cuCtxPopCurrent()
  ctx = Array(CUcontext,1)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxPopCurrent,CUresult,(Ptr{CUcontext},),ctx)))
  return ctx[1]
end

function cuCtxPushCurrent(ctx::CUcontext)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxPushCurrent,CUresult,(CUcontext,),ctx)))
end

function cuCtxSetCacheConfig(config::CUfunc_cache)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxSetCacheConfig,CUresult,(CUfunc_cache,),config)))
end

function cuCtxSetCurrent(ctx::CUcontext)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxSetCurrent,CUresult,(CUcontext,),ctx)))
end

function cuCtxSetLimit(limit::CUlimit,value::CUsize_t)
  jlcucheck(ccall(dlsym(libcuda,:cuCtxSetLimit,CUresult,(CUlimit,CUsize_t),limit,value)))
end

function cuCtxSynchronize()
  jlcucheck(ccall(dlsym(libcuda,:cuCtxSynchronize,CUresult,())))
end

### Module Management
function cuModuleGetFunction(hmod::CUmodule,name::String)
  func = Array(CUfunction,1)
  jlcucheck(ccall(dlsym(libcuda,:cuModuleGetFunction,CUresult,(Ptr{CUfunction},CUmodule,Ptr{Uint8}),func,hmod,cstring(name))))
  return func[1]
end

function cuModuleGetGlobal(hmod::CUmodule,name::String)
  dptr = Array(CUdeviceptr,1);
  bytes = Array(CUsize_t,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleGetGlobal,CUresult,(Ptr{CUdeviceptr},Ptr{CUsize_t},CUmodule,Ptr{Uint8}),dptr,bytes,hmod,cstring(name))))
  return (dptr[1],bytes[1])
end

function cuModuleGetSurfRef(hmod::CUmodule,name::String)
  surfRef = Array(CUsurfref,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleGetSurfRef,CUresult,(Ptr{CUsurfref},CUmodule,Ptr{Uint8}),surfRef,hmod,cstring(name))))
  return surfRef[1]
end

function cuModuleGetTexRef(hmod::CUmodule,name::String)
  texRef = Array(CUtexref,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleGetTexRef,CUresult,(Ptr{CUtexref},CUmodule,Ptr{Uint8}),texRef,hmod,cstring(name))))
  return texRef[1]
end

function cuModuleLoad(fname::String)
  hmod = Array(CUmodule,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleLoad,CUresult,(Ptr{CUmodule},Ptr{Uint8}),hmod,cstring(name))))
  return hmod[1]
end

function cuModuleLoadData_base(image)
  hmod = Array(CUmodule,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleLoadData,CUresult,(Ptr{CUmodule},Ptr{Uint8}),hmod,image)))
  return hmod[1]
end

function cuModuleLoadData(image::String)
  return cuModuleLoadData_base(cstring(image))
end

function cuModuleLoadData(image::Array{Uint8})
  return cuModuleLoadData_base(image)
end

function cuModuleLoadDataEx_base(image)
  hmod = Array(CUmodule,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleLoadDataEx,CUresult,(Ptr{CUmodule},Ptr{Uint8},Uint32,Ptr{CUjit_option},Ptr{Ptr{void}}),hmod,image,numOptions,options,optionValues)))
  return hmod[1]
end

#CUresult 	cuModuleLoadDataEx (CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
# 	Load a module's data with options.

function cuModuleLoadFatBinary(image::Array{Uint8})
  hmod = Array(CUmodule,1);
  jlcucheck(ccall(dlsym(libcuda,:cuModuleLoadFatBinary,CUresult,(Ptr{CUmodule},Ptr{Uint8}),hmod,image)))
  return hmod[1]
end

function cuModuleUnload(hmod::CUmodule)
  jlcucheck(ccall(dlsym(libcuda,:cuModuleUnload,CUresult,(CUmodule,),hmod)))
end


### Memory Management
#CUresult 	cuArray3DCreate (CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
# 	Creates a 3D CUDA array.
#CUresult 	cuArray3DGetDescriptor (CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
# 	Get a 3D CUDA array descriptor.
#CUresult 	cuArrayCreate (CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
# 	Creates a 1D or 2D CUDA array.
#CUresult 	cuArrayDestroy (CUarray hArray)
# 	Destroys a CUDA array.
#CUresult 	cuArrayGetDescriptor (CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
# 	Get a 1D or 2D CUDA array descriptor.
#CUresult 	cuDeviceGetByPCIBusId (CUdevice *dev, char *pciBusId)
# 	Returns a handle to a compute device.
#CUresult 	cuDeviceGetPCIBusId (char *pciBusId, int len, CUdevice dev)
# 	Returns a PCI Bus Id string for the device.
#CUresult 	cuIpcCloseMemHandle (CUdeviceptr dptr)
#CUresult 	cuIpcGetEventHandle (CUipcEventHandle *pHandle, CUevent event)
# 	Gets an interprocess handle for a previously allocated event.
#CUresult 	cuIpcGetMemHandle (CUipcMemHandle *pHandle, CUdeviceptr dptr)
#CUresult 	cuIpcOpenEventHandle (CUevent *phEvent, CUipcEventHandle handle)
# 	Opens an interprocess event handle for use in the current process.
#CUresult 	cuIpcOpenMemHandle (CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)
#CUresult 	cuMemAlloc (CUdeviceptr *dptr, size_t bytesize)
# 	Allocates device memory.
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
#CUresult 	cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
# 	Copies memory from Device to Device.
#CUresult 	cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
# 	Copies memory from Device to Device.
#CUresult 	cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
# 	Copies memory from Device to Host.
#CUresult 	cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
# 	Copies memory from Device to Host.
#CUresult 	cuMemcpyHtoA (CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
# 	Copies memory from Host to Array.
#CUresult 	cuMemcpyHtoAAsync (CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream)
# 	Copies memory from Host to Array.
#CUresult 	cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
# 	Copies memory from Host to Device.
#CUresult 	cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
# 	Copies memory from Host to Device.
#CUresult 	cuMemcpyPeer (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
# 	Copies device memory between two contexts.
#CUresult 	cuMemcpyPeerAsync (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
# 	Copies device memory between two contexts asynchronously.
#CUresult 	cuMemFree (CUdeviceptr dptr)
# 	Frees device memory.
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
#CUresult 	cuMemsetD8 (CUdeviceptr dstDevice, unsigned char uc, size_t N)
# 	Initializes device memory.
#CUresult 	cuMemsetD8Async (CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
# 	Sets device memory. 
#
### Stream Management
#CUresult 	cuStreamCreate (CUstream *phStream, unsigned int Flags)
# 	Create a stream.
#CUresult 	cuStreamDestroy (CUstream hStream)
# 	Destroys a stream.
#CUresult 	cuStreamQuery (CUstream hStream)
# 	Determine status of a compute stream.
#CUresult 	cuStreamSynchronize (CUstream hStream)
# 	Wait until a stream's tasks are completed.
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
#CUresult 	cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
# 	Returns information about a function.
#CUresult 	cuFuncSetCacheConfig (CUfunction hfunc, CUfunc_cache config)
# 	Sets the preferred cache configuration for a device function.
#CUresult 	cuLaunchKernel (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
# 	Launches a CUDA function. 
#
