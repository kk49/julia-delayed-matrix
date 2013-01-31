## Delayed Expressions Julia Backend
# Copyright 2012-2013, Krzysztof Kamieniecki (krys@kamieniecki.com)

require("demat_base.jl")

type DeBackEndJulia  
end

type DeArrJulia{T,N} <: DeArr{DeBackEndJulia,T,N}
  DeArrJulia(a::Array{T,N}) = new(a)
  DeArrJulia(a::Number) = new([convert(T,a)])
  data::Array{T,N}
end

size(a::DeArrJulia) = size(a.data)
size(a::DeArrJulia,dim) = size(a.data,dim)

typealias DeVecJ{T} DeArrJulia{T,1}
typealias DeMatJ{T} DeArrJulia{T,2}

# de_jl_eval returns a 3-tuple that contains
#   symbol that contains the value of the extression
#   result type
#   the quoted preamble code
#   the quoted kernal code

function de_jl_eval(a::DeConst,param,idxSym)
    @gensym r
    ( r
    , typeof(a.p1)
    , quote ($r) = ($param).p1 end 
    , quote end
    )
end

function de_jl_eval(a::DeReadOp,param,idxSym)
    @gensym r src
    ( r
    , eltype(a.p1.data)
    , quote ($src) = ($param).p1.data end 
    , quote ($r) = ($src)[($idxSym)] end
    )
end

#function de_jl_eval(v::DeBinOp,param,idxSym)
function de_jl_eval(v::DeBinOp,param,idxSym)
   @gensym r
   p1 = de_jl_eval(v.p1,quote ($param).p1 end,idxSym)
   p2 = de_jl_eval(v.p2,quote ($param).p2 end,idxSym)
   ( r
   , promote_type(p1[2],p2[2])
   , quote $(p1[3]);$(p2[3]) end
   , quote $(p1[4]);$(p2[4]);($r) = de_jl_do_op($v,$(p1[1]),$(p2[1])) end 
   )
end

for op = keys(deBinOpMap)
    opSingle = deBinOpMap[op];
    @eval de_jl_do_op{P1,P2}(v::DeBinOp{$(expr(:quote, op)),P1,P2},a,b) = ($opSingle)(a,b)
    #@eval de_jl_do_op(v::DeBinOp{$op},a,b) = ($opSingle)(a,b)
    #@eval function de_jl_do_op(v::DeBinOp{$opType},a,b) = ($opSingle)(a,b) end # does not work?
end

function assign(lhs::DeVecJ,rhs::DeExpr)
  buildTime = @elapsed begin
    @gensym i prhs
    (rhsResult,rhsResultType,rhsPreamble,rhsKernel) = de_jl_eval(rhs,prhs,i)
    rhsType = typeof(rhs);

    @eval function assign1(plhs::DeVecJ,($prhs)::($rhsType))        
        rhsSz = de_check_dims($prhs)
        lhsSz = size(plhs)
        if rhsSz != lhsSz
           error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
        end
        
        N = size(plhs,1)
        lhsData = plhs.data
        $rhsPreamble
        for ($i) = 1:N
            $rhsKernel
            lhsData[($i)] = ($rhsResult)
        end

        return plhs
    end

    global assign
    @eval assign(lhs::DeVecJ,rhs::($rhsType)) = assign1(lhs,rhs)
  end
       
  println("DeMatJulia: Built New Assign (took $buildTime seconds) ... $rhsType");
 
  return assign1(lhs,rhs)
end

assign(lhs::DeArrJulia,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrJulia,rhs::Number) = assign(lhs,de_promote(rhs)...)

## TODO quick hack for sum, should be generalized for reduce
function sum(rhs::DeExpr)
  buildTime = @elapsed begin
    @gensym i prhs
    (rhsResult,rhsResultType,rhsPreamble,rhsKernel) = de_jl_eval(rhs,prhs,i)
    rhsType = typeof(rhs);

    @eval function sum1(($prhs)::($rhsType))
        rhsSz = de_check_dims($prhs)

        N = rhsSz[1]
        sumV = convert($rhsResultType,0)
        $rhsPreamble
        for ($i) = 1:N
            $rhsKernel
            sumV += ($rhsResult)
        end

        return sumV
    end

    global sum
    @eval sum(rhs::($rhsType)) = sum1(rhs)
  end

  println("DeMatJulia: Built New Sum (took $buildTime seconds) ... $rhsType");

  return sum1(rhs)
end

sum(rhs::DeArrJulia) = sum(de_promote(rhs)...)

