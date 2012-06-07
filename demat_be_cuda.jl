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
  if ctxt.handle != convert(typeof(ctxt.handle),0)
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
  if buffer.ptr != convert(typeof(buffer.ptr),0)
     println("CUDA Clearing $buffer");
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
    cuMemcpyHtoD(dst.ptr,src,dst.sz);
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

function eltype{T,N}(arr::DeArrCuda{T,N})
  return T;
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
const registerTypeToPrefix = {Float32=>"%f32_",Ptr{Float32}=>"%pf32_",Float64=>"%f64_",Ptr{Float64}=>"%pf64_",Uint32=>"%u32_",Ptr{Uint32}=>"%pu32_",Int32=>"%s32_",Ptr{Int32}=>"%ps32_"};

type DePtxEnv
  function DePtxEnv()
    me = new(Dict(),Array(Any,0),Array(String,0));
    return me;
  end  

  registerCounter::Dict;
  paramTypes::Array{Any,1};
  paramNames::Array{String,1};
end

type PtxRegister{dataType}
  id::String;
end

function registerAlloc!(T,env::DePtxEnv)
  prefix = registerTypeToPrefix[T];

  if !has(env.registerCounter,T)
    env.registerCounter[T] = 0;
  end

  id = env.registerCounter[T];
  env.registerCounter[T] += 1;
  
  return PtxRegister{T}("$(prefix)$(id)");    
end

function registerType{dataType}(reg::PtxRegister{dataType})
  return dataType;
end

function paramAlloc!(T,env::DePtxEnv)
  paramIndex = numel(env.paramTypes);
  paramName = "%p$(paramIndex)";
  push(env.paramTypes,T);
  push(env.paramNames,paramName);

  return (paramName,paramIndex);
end

abstract PtxOp;

type msGlobal end
type msParam end

type PtxOpLoad{memorySpace,dataType} <: PtxOp
  dst::PtxRegister{dataType};
  addr::Union(String,PtxRegister,(PtxRegister,Int32),Uint32); #named address, register, register+offset, absolute
end 

type PtxOpStore{memorySpace,dataType} <: PtxOp
  addr::Union(String,PtxRegister,(PtxRegister,Int32),Uint32); #named address, register, register+offset, absolute
  src::PtxRegister{dataType};
end 

type PtxOpBin{op,dataType} <: PtxOp
  dst::PtxRegister{dataType};
  op0::Union(Int64,Uint64,Float32,Float64,PtxRegister{dataType});
  op1::Union(Int64,Uint64,Float32,Float64,PtxRegister{dataType});
end

# (registerType,register,param setup function, IR Ops)

function de_cude_eval(a::DeConst,env,indexReg)
    # allocate destination register
    rType = typeof(a.p1);
    r = registerAlloc!(rType,env);

    # allocate space in param structure
    (paramName,paramIndex) = paramAlloc!(rType,env);

    paramSetupFunc = @eval function(param,v::DeConst) param[$paramIndex] = [v.p1] end

    ops = 
    [ PtxOpLoad{msParam,rType}(r,paramName)
    ];

    return ( rType , r , paramSetupFunc , ops );
end

function de_cuda_eval(a::DeReadOp,env,indexReg)
    # allocate ptr and destination register
    rType = eltype(a.p1);
    r = registerAlloc!(rType,env);
    srcType = Ptr{rType};
    src = registerAlloc!(srcType,env);
    offset = registerAlloc!(srcType,env);
    srcOffset = registerAlloc!(srcType,env);
    
    # allocate space in param structure
    (paramName,paramIndex) = paramAlloc!(srcType,env);
    
    paramSetupFunc = @eval function(param,v::DeReadOp) param[$paramIndex] = [v.p1.buffer.ptr] end

    ops =
    [ PtxOpLoad{msParam,srcType}(src,paramName)
      PtxOpBin{DeOpLShift,srcType}(offset,indexReg,convert(Uint64,log2(sizeof(rType))))
      PtxOpBin{DeOpAdd,srcType}(srcOffset,src,offset)
      PtxOpLoad{msGlobal,rType}(r,srcOffset)
    ];

    return ( rType , r , paramSetupFunc , ops );
end

function de_cuda_eval{OT}(v::DeBinOp{OT},env,indexReg)
  
   p1 = de_cuda_eval(v.p1,env,indexReg)
   p2 = de_cuda_eval(v.p2,env,indexReg)
   
   rType = promote_type(p1[1],p2[1]);
   r = registerAlloc!(rType,env);

   paramSetupFunc = @eval function(param,v::DeBinOp{OT}) $(p1[3])(param,v.p1); $(p2[3])(param,v.p2); end

   ops =
   [ p1[4]
     p2[4]
     PtxOpBin{OT,rType}(r,p1[2],p2[2])
   ];
  
   return ( rType , r , paramSetupFunc , ops );
end

# assignement
function assign(lhs::DeVecCu,rhs::DeExpr)
  buildTime = @elapsed begin
    env = DePtxEnv();
    indexRegister = registerAlloc!(Uint32,env);
    ret = de_cuda_eval(rhs,env,indexRegister);
    println(ret[1]);
    println(ret[2]);
    println(ret[3]);
    println(ret[4]);

    rhsType = typeof(rhs);

    #@eval function assign1(plhs::DeVecJ,($prhs)::($rhsType))
    #    rhsSz = de_check_dims($prhs)
    #    lhsSz = size(plhs)
    #    if rhsSz != lhsSz
    #       error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
    #    end

    #    N = size(plhs,1)
    #    lhsData = plhs.data
    #    $rhsPreamble
    #    for ($i) = 1:N
    #        $rhsKernel
    #        lhsData[($i)] = ($rhsResult)
    #    end

    #    return plhs
    #end

    #global assign
    #@eval assign(lhs::DeVecJ,rhs::($rhsType)) = assign1(lhs,rhs)
  end

  println("DeMatJulia: Built New Assign (took $buildTime seconds) ... $rhsType");
    
  return lhs;
end

assign(lhs::DeArrCuda,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrCuda,rhs::Number) = assign(lhs,de_promote(rhs)...)


#initialize cuda only once
if !isbound(:deCudaCtx)
    cuInit();

    #list cuda devices, select device zero for now
    deCudaDeviceCount = jlcuDeviceList();
    if deCudaDeviceCount > 1
        println("More than one CUDA device found, using the first one.");
   end
   deCudaCtx = cuCtxCreate(0,cuDeviceGet(0));
end
