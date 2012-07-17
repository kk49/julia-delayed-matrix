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

function size{T}(arr::DeArrCuda{T,1})
  return (arr.sz,);
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
  op0::Union(Int64,Uint64,Float32,Float64,PtxRegister);
  op1::Union(Int64,Uint64,Float32,Float64,PtxRegister);
end

# (registerType,register,param setup function, IR Ops)

function de_cuda_eltype(a::DeConst)
  ptype = typeof(a.p1);
  return ptype;
end

function de_cuda_eltype(a::DeReadOp)
  return eltype(a.p1);
end

function de_cuda_eltype(a::DeBinOp)
  p1type = de_cuda_eltype(a.p1);
  p2type = de_cuda_eltype(a.p2);

  if p1type == p2type
     return p1type
  else
     error("Conversion between $p1type and $p2type not yet supported")
  end
end

function de_cuda_eval(a::DeConst,env,paramOut,paramIn,indexReg)
    # allocate destination register
    rType = de_cuda_eltype(a);
    r = registerAlloc!(rType,env);

    # allocate space in param structure
    (paramName,paramIndex) = paramAlloc!(rType,env);

    paramSetup = quote ($paramOut)[$paramIndex+1] = [($paramIn).p1] end

    ops = Array(PtxOp,0);
    push(ops,PtxOpLoad{msParam,rType}(r,paramName));

    return ( rType , r , paramSetup , ops );
end

function de_cuda_eval(a::DeReadOp,env,paramOut,paramIn,indexReg)
    # allocate ptr and destination register
    rType = de_cuda_eltype(a);
    r = registerAlloc!(rType,env);
    srcType = Ptr{rType};
    src = registerAlloc!(srcType,env);
    offset = registerAlloc!(srcType,env);
    srcOffset = registerAlloc!(srcType,env);
    
    # allocate space in param structure
    (paramName,paramIndex) = paramAlloc!(srcType,env);
    
    paramSetup = quote ($paramOut)[$paramIndex+1] = [($paramIn).p1.buffer.ptr] end

    ops = Array(PtxOp,0);
    push(ops, PtxOpLoad{msParam,srcType}(src,paramName));
    push(ops, PtxOpBin{DeOpLShift,srcType}(offset,indexReg,convert(Uint64,log2(sizeof(rType)))));
    push(ops, PtxOpBin{DeOpAdd,srcType}(srcOffset,src,offset));
    push(ops, PtxOpLoad{msGlobal,rType}(r,srcOffset));
    

    return ( rType , r , paramSetup , ops );
end

function de_cuda_eval{OT}(a::DeBinOp{OT},env,paramOut,paramIn,indexReg)
  
   p1 = de_cuda_eval(a.p1,env,paramOut,quote ($paramIn).p1 end, indexReg)
   p2 = de_cuda_eval(a.p2,env,paramOut,quote ($paramIn).p2 end, indexReg)
 
   rType = de_cuda_eltype(a);
   r = registerAlloc!(rType,env);

   f1 = p1[3];
   f2 = p2[3];
   paramSetup = quote $(p1[3]); $(p2[3]); end

   ops = [ p1[4], p2[4] ];
   push(ops, PtxOpBin{OT,rType}(r,p1[2],p2[2]))
  
   return ( rType , r , paramSetup , ops );
end

# assignement
function assign(lhs::DeVecCu,rhs::DeExpr)
  buildTime = @elapsed begin
    ltype = eltype(lhs);
    env = DePtxEnv();
    (lengthName,lengthIndex) = paramAlloc!(Uint32,env)
    (dstName,dstIndex) = paramAlloc!(Ptr{ltype},env)
    indexRegister = registerAlloc!(Uint32,env)
    @gensym paramOut paramIn
    ret = de_cuda_eval(rhs,env,paramOut,paramIn,indexRegister);
    rtype = ret[1]
    rreg = ret[2]
    paramSetup = ret[3]
    ops = ret[4]

    println("return Type: $rtype");
    println("return Reg:  $rreg");
    println("ops:");
    for i = 1:numel(ops)
      println(ops[i])
    end
    println("env: $env")

    # setup parameter list
    paramString = ""
    for i = 1:numel(env.paramTypes)
      pname = env.paramNames[i]
      pt = env.paramTypes[i]
      pts =""

      et = pt
      if pt <: Ptr       
        if Ptr{Uint32} == et
          pts = ".u32"
        elseif Ptr{Uint64} == et
          pts = ".u64"
        elseif Ptr{Float32} == et
          pts = ".f32"
        elseif Ptr{Float64} == et
          pts = ".f64"
        else
          error("Unhandled CUDA parameter type $pt")
        end
        pts = "$pts.ptr.global"
      else
        if Uint32 == et
          pts = ".u32"
        elseif Uint64 == et
          pts = ".u64"
        elseif Float32 == et
          pts = ".f32"
        elseif Float64 == et
          pts = ".f64"
        else
          error("Unhandled CUDA parameter type $pt")
        end
      end
  
      if pt <: Ptr
      end    
  
      paramString = "$paramString .param $pts $pname"
      if i < numel(env.paramTypes)
        paramString = "$paramString,\n"
      end
    end

    # setup computation
    compString = ""
    
    # setup result storage
    resultString = ""

    ccode = 
".version 3.0
.target sm_11
.entry julia_func 
(
$paramString
)
{
$compString

$resultString
} 
"

    rhsType = typeof(rhs);

    infoLogSize = 1024;
    infoLog = Array(Uint8,infoLogSize);
    infoLog[1:end] = 0
    errLogSize = 1024;
    errLog = Array(Uint8,errLogSize);
    errLog[1:end] = 0
    
    println("Input PTX:")
    println("----------------------------")
    println(ccode)  
    println("----------------------------")
    retM = cuModuleLoadDataEx(
      ccode,
      CU_JIT_WALL_TIME,
      CU_JIT_TARGET_FROM_CUCONTEXT,
      CU_JIT_INFO_LOG_BUFFER_SIZE,infoLogSize,
      CU_JIT_INFO_LOG_BUFFER,infoLog,
      CU_JIT_ERROR_LOG_BUFFER_SIZE,errLogSize,
      CU_JIT_ERROR_LOG_BUFFER,errLog)
    println("Output:") 
    println("----------------------------")
    println("errno: $(retM[1])")
    println("hmod: $(retM[2])")
    for i = 1:numel(retM[3])
      println("$(retM[3][i]) : $(retM[4][i])")
    end
    println("----------------------------")

    @eval function assign1(plhs::DeVecCu,($paramIn)::($rhsType))
      rhsSz = de_check_dims($paramIn)
      lhsSz = size(plhs)
      if rhsSz != lhsSz
         error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
      end

      N = numel(plhs)

      $paramOut = Array(Any,$(numel(env.paramTypes)))
      ($paramOut)[$lengthIndex+1] = [convert(Uint32,N)]
      ($paramOut)[$dstIndex+1] = [plhs.buffer.ptr]
      $paramSetup
      params = $paramOut
      println("params: $params");
      for i = 1:numel(params)
        println("$i: $(typeof(params[i])) $(params[i])")
      end
    end

    assign1(lhs,rhs)

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
