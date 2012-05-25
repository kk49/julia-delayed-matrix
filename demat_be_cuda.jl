## Delayed Expressions CUDA Backend
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

#TODO A LOT

## setup cuda library
#TODO should auto detect if the library is missing and give up

load("libcuda.jl")


# CUDA objects, should go to different file

# context
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

# memory buffer
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

# module

# kernel



## Delayed matrix backend
type DeBackEndCuda
end

type DeArrCuda{T,N} <: DeArr{DeBackEndCuda,T,N}
  
  function DeArrCuda()
    me = new(jlCUBuffer());
    me
  end
  
  function DeArrCuda(a::Array{T,N})
    me = new(jlCUBuffer());
    resize(me,numel(a));
    me[] = a;
    me
  end

  function DeArrCuda(n::Number)
    me = new(jlCUBuffer());
    resize(me,n);
    me
  end
 
  buffer::jlCUBuffer
end

function numel{T}(arr::DeArrCuda{T,1})
  return arr.buffer.sz / sizeof(T);
end

function clear(arr::DeArrCuda)
  clear(arr.buffer);
  return arr;
end

function resize{T}(arr::DeArrCuda{T,1},nsz)
  resize(arr.buffer,sizeof(T) * nsz);
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

