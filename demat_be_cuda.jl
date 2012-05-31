## Delayed Expressions CUDA Backend
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

#TODO A LOT

## setup cuda library
#TODO should auto detect if the library is missing and give up

load("libcuda.jl")


### CUDA objects, should go to different file

## Context
type jlCUContext
  function jlCUContext()
    me = new(cuCtxCreate(convert(Uint32,0),convert(Int32,0)));
    finalizer(me,reset);
    return me;
  end

  handle::CUcontext
end

function reset(ctxt::jlCUContext)
  if ctxt.handle != 0
    cuCtxDestroy(ctxt.handle);
    ctxt.handle = 0;
  end

  return ctxt;
end

## Host Memory Buffer - that supports concurrent transfers it is "pinned" (cannot be paged out)
# TODO

## Device Memory Buffer
type jlCUBuffer
  function jlCUBuffer()
    me = new(0,0);
    finalizer(me,clear);
    me
  end

  function jlCUBuffer(nSz::Number)
    me = new(0,0);
    finalizer(me,clear);
    resize(me,nSz);
    me
  end

  sz::CUsize_t
  ptr::CUdeviceptr
end

function clear(buffer::jlCUBuffer)
  if buffer.ptr != 0
     cuMemFree(buffer.ptr);
     buffer.sz = 0;
     buffer.ptr = 0;
  end
  return buffer;
end

function numel(buffer::jlCUBuffer)
  return buffer.sz;
end

function resize(buffer::jlCUBuffer,nsz)
  clear(buffer);
  buffer.ptr = cuMemAlloc(nsz);
  buffer.sz = nsz;
  return buffer;
end

function copyto{T}(dst::jlCUBuffer,src::Array{T,1})
    @assert numel(dst) == numel(src)*sizeof(T);
    @time cuMemcpyHtoD(dst.ptr,src,dst.sz);
    return dst;
end

function copyto{T}(dst::Array{T,1},src::jlCUBuffer)
    @assert numel(dst)*sizeof(T) == numel(src);
    @time cuMemcpyDtoH(dst,src.ptr,src.sz);
    return dst;
end

## Module


## Kernel


### Delayed matrix backend
type DeBackEndCuda
end

type DeArrCuda{T,N} <: DeArr{DeBackEndCuda,T,N}
  
  function DeArrCuda()
    me = new(0,jlCUBuffer());
    me
  end
  
  function DeArrCuda(a::Array{T,N})
    me = new(0,jlCUBuffer());
    resize(me,numel(a));
    me[] = a;
    me
  end

  function DeArrCuda(n::Number)
    me = new(0,jlCUBuffer());
    resize(me,n);
    me
  end
 
  sz::CUsize_t
  buffer::jlCUBuffer
end

function numel{T}(arr::DeArrCuda{T,1})
  return arr.sz;
end

function clear(arr::DeArrCuda)
  clear(arr.buffer);
  arr.sz = 0;
  return arr;
end

function resize{T}(arr::DeArrCuda{T,1},nsz)
  ncnt = sizeof(T) * nsz;
  resize(arr.buffer,ncnt);
  arr.sz = nsz;
  return arr;
end

function assign{T}(dst::DeArrCuda{T,1},src::Array{T,1})
  copyto(dst.buffer,src);
  return dst;
end

function assign{T}(dst::Array{T,1},src::DeArrCuda{T,1})
  copyto(dst,src.buffer);  
  return dst;
end

typealias DeVecCu{T} DeArrCuda{T,1}
typealias DeMatCu{T} DeArrCuda{T,2}

# DeCudaEnviroment used to count/allocate registers for PTX generation
type DePtxEnv
  function DePtxEnv()
    me = new(0,0,0,0);
    return me;
  end  

  f32Cnt::Int
  pf32Cnt::Int
  f64Cnt::Int
  pf64Cnt::Int
end

function genRegisterF32!(env::DePtxEnv)
  regStr = "%f$(env.f32Cnt)";
  env.f32Cnt +=1;
  return regStr;
end

function genRegisterPtrF32!(env::DePtxEnv)
  regStr = "%pf$(env.f32Cnt)";
  env.pf32Cnt +=1;
  return regStr;
end

function genRegisterF64!(env::DePtxEnv)
  regStr = "%d$(env.f64Cnt)";
  env.f64Cnt +=1;
  return regStr;
end

function genRegisterPtrF64!(env::DePtxEnv)
  regStr = "%pd$(env.pf64Cnt)";
  env.pf64Cnt +=1;
  return regStr;
end

# assignement
function assign(lhs::DeVecCu,rhs::DeExpr)
    return lhs
end

assign(lhs::DeArrCuda,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrCuda,rhs::Number) = assign(lhs,de_promote(rhs)...)


#initialize cuda
cuInit();

#list cuda devices, select device zero for now
deCudaDeviceCount = jlcuDeviceList();
if deCudaDeviceCount > 1
  println("More than one CUDA device found, using the first one.");
end
deCudaCtx = cuCtxCreate(0,cuDeviceGet(0));

