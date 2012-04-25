# Julia Wrapper for libcuda.so (tested against version 4.2)
# USAGE: 
#  You must call cuInit() first
#  to get a list of device call jlcuDeviceList()

libcuda = dlopen("libcuda")

#libcuda_cuInitSym = dlsym(libcuda,:cuInit)

typealias CUdevice Int32
typealias CUcontext Ptr{Void}
typealias CUmodule Ptr{Void}
typealias CUfunction Ptr{Void}
typealias CUdeviceptr Uint32
typealias CUevent Ptr{Void}
typealias CUstream Ptr{Void}

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

# call to cuInit() not needed before calling cuDriverGetVersion
function cuDriverGetVersion()
  #parameter is actually a pointer to Uint, but we use int to display human readable value
  version = Array(Int32,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDriverGetVersion),CUresult,(Ptr{Int32},), version))
  return version[1]
end

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

function cuDeviceGetName(deviceHandle)
  deviceName = Array(Uint8, 128)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceGetName),CUresult,(Ptr{Uint8},Int32,CUdevice),deviceName, length(deviceName),deviceHandle))
  return cstring(convert(Ptr{Uint8}, deviceName))
end

function cuDeviceTotalMem(deviceHandle)
  sizeBytes = Array(Uint64,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuDeviceTotalMem),CUresult,(Ptr{Uint64},Int32),sizeBytes,deviceHandle))
  return sizeBytes[1]
end

## Context Management

function cuCtxCreate(flags,deviceHandle)
  context = Array(CUcontext,1)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxCreate),CUresult,(Ptr{CUcontext},Uint32,CUdevice),context,flags,deviceHandle))
  return context[1]
end

function cuCtxDestroy(context)
  jlcuCheck(ccall(dlsym(libcuda,:cuCtxDestroy),CUresult,(CUcontext,),context))
  return ()
end

