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
    cuMemcpyDtoH(dst,src.ptr,src.sz);
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

const cudaPtr = "u32"
const juliaTypeToCudaType = {Float32=>"f32",Ptr{Float32}=>cudaPtr,Float64=>"f64",Ptr{Float64}=>cudaPtr,Uint32=>"u32",Ptr{Uint32}=>cudaPtr,Int32=>"s32",Ptr{Int32}=>cudaPtr}
const cudaBPtr = "b32"
const juliaTypeToCudaBType = {Float32=>"b32",Ptr{Float32}=>cudaBPtr,Float64=>"b64",Ptr{Float64}=>cudaBPtr,Uint32=>"b32",Ptr{Uint32}=>cudaBPtr,Int32=>"b32",Ptr{Int32}=>cudaBPtr}

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

function de_eltype{T,N}(a::Type{Array{T,N}})
  return T
end

function de_eltype{T}(a::Type{Ptr{T}})
  return T
end

function de_eltype(a::Type)
  return a
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

const DeOpToCudaOp = {DeOpAdd=>"add",DeOpLShift=>"shl",DeOpMulEle=>"mul"}
const CudaMemSpaceToString = {msGlobal=>"global",msParam=>"param"}


function de_cuda_operand(op::PtxRegister)
  return op.id;
end

function de_cuda_operand(op) 
  return "$op"
end

function de_cuda_op_to_string{OP,ST}(op::PtxOpBin{OP,ST})
  if OP == DeOpLShift
    opString = "$(DeOpToCudaOp[OP]).$(juliaTypeToCudaBType[ST])"
  else
    opString = "$(DeOpToCudaOp[OP]).$(juliaTypeToCudaType[ST])"
  end
  dst = op.dst.id
  op0 = de_cuda_operand(op.op0)
  op1 = de_cuda_operand(op.op1)

  return "$opString $dst,$op0,$op1"
end

function de_cuda_op_to_string{MS,ST}(op::PtxOpLoad{MS,ST})
  opString = "ld.$(CudaMemSpaceToString[MS]).$(juliaTypeToCudaType[ST])" 
  dst = op.dst.id
  addr = de_cuda_operand(op.addr)
  return "$opString $dst,[$addr]"
end

# assignement
function assign(lhs::DeVecCu,rhs::DeExpr)
  buildTime = @elapsed begin
    ltype = eltype(lhs);
    env = DePtxEnv();
    (lengthName,lengthIndex) = paramAlloc!(Uint32,env)
    (dstPtrName,dstPtrIndex) = paramAlloc!(Ptr{ltype},env)
    lengthReg = registerAlloc!(Uint32,env)
    indexReg = registerAlloc!(Uint32,env)
    dstPtrReg = registerAlloc!(Ptr{ltype},env)
    @gensym paramOut paramIn
    ret = de_cuda_eval(rhs,env,paramOut,paramIn,indexReg);
    rtype = ret[1]
    rreg = ret[2]
    paramSetup = ret[3]
    ops = ret[4]

#    println("return Type: $rtype");
#    println("return Reg:  $rreg");
#    println("env: $env")

    # setup parameter list
    paramString = ""
    for i = 1:numel(env.paramTypes)
      pname = env.paramNames[i]
      pt = env.paramTypes[i]
      pts =""

      et = de_eltype(pt)
      pts = ".$(juliaTypeToCudaType[et])"

      #TODO assume memory allocated by cuMalloc and is 16byte aligned
      if pt <: Ptr       
        pts = "$pts.ptr.global.align 16 "
      end
  
      paramString = "$paramString .param $pts $pname"
      if i < numel(env.paramTypes)
        paramString = "$paramString,\n"
      end
    end

    # register allocation
    regString = ""
    regKeys = keys(env.registerCounter)
    for ri = 1:numel(regKeys)
      rt = regKeys[ri]
      rct = juliaTypeToCudaType[rt]
      rc = env.registerCounter[rt]
      rp = registerTypeToPrefix[rt]
      regString = "$regString    .reg .$rct $(rp)<$rc>;\n"
    end

    # setup computation
    compString = ""
#    println("ops:");
    for i = 1:numel(ops)
      opstring = de_cuda_op_to_string(ops[i])
#      println("$(ops[i]) --> $opstring")
      compString = "$compString    $opstring;\n"
    end
    
    # setup result storage
    resultString = ""
    
    stcache = ""
    if false #sm >= sm_20
      stcache = ".cs"
    end

    ccode = 
".version 3.0
.target sm_11
.entry julia_func 
(
$paramString
)
{
// Register Allocation
$regString
    .reg .u32 %ii,%ix,%nx,%cix,%cnx; //registers for index determination
    .reg .pred p;
    
    ld.param.u32 $(lengthReg.id),[$(lengthName)]; //load length
    ld.param.u32 $(dstPtrReg.id),[$(dstPtrName)]; //load destination ptr

    // Index Setup
    //TODO OVERKILL, but it should work for all cases
    mov.u32 %ix,%tid.x;
    mov.u32 %nx,%ntid.x;
    mov.u32 %cix,%ctaid.x;
    mov.u32 %cnx,%nctaid.x;

    mad.lo.u32 $(indexReg.id),%cix,%nx,%ix;


    setp.lt.u32 p,$(indexReg.id),$(lengthReg.id);
@!p bra END; 
    // Computation
$compString
    // Store results
    shl.b32 %ii,$(indexReg.id),2;
    add.u32 $(dstPtrReg.id),$(dstPtrReg.id),%ii;
    st.global$stcache.$(juliaTypeToCudaType[ltype]) [$(dstPtrReg.id)],$(rreg.id);
$resultString
    // Jump destination for threads that are beyond data length
END:
} 
"

    rhsType = typeof(rhs);

    infoLogSize = 4096;
    infoLog = Array(Uint8,infoLogSize);
    infoLog[1:end] = 0
    errLogSize = 4096;
    errLog = Array(Uint8,errLogSize);
    errLog[1:end] = 0
    
    retM = cuModuleLoadDataEx(
      ccode,
      CU_JIT_WALL_TIME,
      CU_JIT_TARGET_FROM_CUCONTEXT,
      CU_JIT_THREADS_PER_BLOCK,1024,
      CU_JIT_INFO_LOG_BUFFER_SIZE,infoLogSize,
      CU_JIT_INFO_LOG_BUFFER,infoLog,
      CU_JIT_ERROR_LOG_BUFFER_SIZE,errLogSize,
      CU_JIT_ERROR_LOG_BUFFER,errLog)

    res = retM[1]
    hmod = retM[2]
    threadsPerBlock = 0 

    showInfoLog = false
    showErrorLog = false
    for i = 1:numel(retM[3])
      if CU_JIT_THREADS_PER_BLOCK == retM[3][i]
        threadsPerBlock = retM[4][i]
      elseif CU_JIT_WALL_TIME == retM[3][i]
        println("CUDA Build Time: $(retM[4][i]/1000) Seconds") 
      elseif CU_JIT_INFO_LOG_BUFFER_SIZE == retM[3][i]
        showInfoLog = retM[4][i] > 0
      elseif CU_JIT_INFO_LOG_BUFFER == retM[3][i]
        if showInfoLog
          println("CUDA BUILD INFO LOG: $(retM[4][i])")
        end
      elseif CU_JIT_ERROR_LOG_BUFFER_SIZE == retM[3][i]
        showErrorLog = retM[4][i] > 0
      elseif CU_JIT_ERROR_LOG_BUFFER == retM[3][i]
        if showErrorLog
          println("CUDA BUILD ERROR LOG: $(retM[4][i])")
        end
      end
    end

    displayResults = false
    if 0 != hmod
      funcHnd = cuModuleGetFunction(hmod,"julia_func")

      @eval function assign1(plhs::DeVecCu,($paramIn)::($rhsType))
        rhsSz = de_check_dims($paramIn)
        lhsSz = size(plhs)
        if rhsSz != lhsSz
          error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
        end

        N = numel(plhs)

        $paramOut = Array(Any,$(numel(env.paramTypes)))
        ($paramOut)[$lengthIndex+1] = [uint32(N)]
        ($paramOut)[$dstPtrIndex+1] = [plhs.buffer.ptr]
        $paramSetup
        params = $paramOut
        paramPtrs = Array(Ptr{Void},$(numel(env.paramTypes)))
        for i = 1:$(numel(env.paramTypes))
          paramPtrs[i] = convert(Ptr{Void},params[i])
        end
#        println("params: $params");
#        for i = 1:numel(params)
#          println("$i: $(typeof(params[i])) $(params[i])")
#        end

        nxBlock = uint32((N + $threadsPerBlock - 1) / $threadsPerBlock)
        nxThread = uint32($threadsPerBlock)
#        println("nxb: $nxBlock, nxt: $nxThread")
        cuLaunchKernel(
          $funcHnd,
          nxBlock, uint32(1), uint32(1), # gridDim
          nxThread, uint32(1), uint32(1), #blockDim
          uint32(0), #sharedMemBytes::Uint32, 
          convert(Ptr{Void},0), #hStream::CUstream,
          paramPtrs, 
          convert(Ptr{Void},0)) #extra

        cuStreamSynchronize(convert(Ptr{Void},0)) # wait for stream to finish, normally you would not do this but for timing tests it is needed

        plhs
      end

      global assign
      @eval assign(lhs::DeVecCu,rhs::($rhsType)) = assign1(lhs,rhs)
    else
      displayResults = true;
    end

    if displayResults
      println("PTX Code:")
      println("----------------------------")
      println(ccode)
      println("----------------------------")
      println("Output:") 
      println("----------------------------")
      println("errno: $res")
      println("hmod: $hmod")
      for i = 1:numel(retM[3])
        println("$(retM[3][i]) : $(retM[4][i])")
      end
      println("----------------------------")
      error("Cuda Build Error: $res")
    end
  end
  
  println("DeMatJulia: Built New Assign (took $buildTime seconds) ... $rhsType");

  return assign1(lhs,rhs);
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
