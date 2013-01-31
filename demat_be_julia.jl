## Delayed Expressions Julia Backend
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

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
#   symbol that contains the value of the expression
#   the quoted preamble code
#   the quoted kernal code

function de_jl_eval(a::DeConst,idxSym)
    @gensym r
    ( r
    , quote ($r) = ($(a.p1)) end
    , quote end
    )
end

function de_jl_eval(a::DeReadOp,idxSym)
    @gensym r src
    ( r
    , quote ($src) = ($(a.p1.data)) end
    , quote ($r) = ($src)[($idxSym)] end
    )
end

function de_jl_eval(v::DeBinOp,idxSym)
   @gensym r
   p1 = de_jl_eval(v.p1,idxSym)
   p2 = de_jl_eval(v.p2,idxSym)
   ( r
   , quote $(p1[2]);$(p2[2]) end
   , quote $(p1[3]);$(p2[3]);($r) = de_jl_do_op($v,$(p1[1]),$(p2[1])) end
   )
end

for op = keys(deBinOpMap)
    opType = DeBinOp{op};
    opSingle = deBinOpMap[op];
    @eval de_jl_do_op(v::($opType),a,b) = ($opSingle)(a,b)
    #@eval de_jl_do_op(v::DeBinOp{$op},a,b) = ($opSingle)(a,b)
    #@eval function de_jl_do_op(v::DeBinOp{$opType},a,b) = ($opSingle)(a,b) end # does not work?
end

function assign(lhs::DeVecJ,rhs::DeExpr)
    rhsSz = de_check_dims(rhs)
    lhsSz = size(lhs)

    if rhsSz != lhsSz
        error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
    end

    @gensym i
    (rhsResult,rhsPreamble,rhsKernel) = de_jl_eval(rhs,i)
    rhsType = typeof(rhs);

    @eval function hiddenFunc(plhs::DeVecJ,prhs::($rhsType))
        N = size(plhs,1)
        lhsData = plhs.data
        $rhsPreamble
        for ($i) = 1:N
            $rhsKernel
            lhsData[($i)] = ($rhsResult)
        end
    end

    hiddenFunc(lhs,rhs)

    return lhs
end

assign(lhs::DeArrJulia,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrJulia,rhs::Number) = assign(lhs,de_promote(rhs)...)


