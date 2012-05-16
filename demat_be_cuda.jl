## Delayed Expressions CUDA Backend
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

#TODO A LOT

## setup cuda library
#TODO should auto detect if the library is missing and give up
load("libcuda.jl")
cuInit();
deCudaDeviceCount = jlcuDeviceList();
if deCudaDeviceCount > 1
  println("More than one CUDA device found, using the first one.");
end
deCudaCtx = cuCtxCreate(0,cuDeviceGet(0));


type DeBackEndCuda
end

type DeArrCuda{T,N} <: DeArr{DeBackEndCuda,T,N}
  
  function DeArrCuda()
    me = new(0,0);
    finalizer(me,clear);
    me
  end
  
  function DeArrCuda(a::Array{T,N})
    me = new(0,0);
    finalizer(me,clear);
    resize(me,numel(a));
    @time cuMemcpyHtoD(me.ptr,a,me.sz);
    me
  end

  function DeArrCuda(n::Number)
    me = new(0,0);
    finalizer(me,clear);
    resize(me,n);
    @time cuMemsetD8(me.ptr,0,me.sz);
    me
  end
  
  sz::CUsize_t
  ptr::CUdeviceptr
end

function clear(arr::DeArrCuda)
  if arr.ptr != 0
     cuMemFree(arr.ptr);
     arr.sz = 0;
     arr.ptr = 0;
  end
  arr
end

function resize{T,N}(arr::DeArrCuda{T,N},nsz)
  clear(arr);
  nsz = sizeof(T) * nsz;
  arr.ptr = cuMemAlloc(nsz);
  arr.sz = nsz;
  arr
end

typealias DeVecCu{T} DeArrCuda{T,1}
typealias DeMatCu{T} DeArrCuda{T,2}

