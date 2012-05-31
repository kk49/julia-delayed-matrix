## Delayed Expressions Julia Backend
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

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
    , quote ($r) = (($param).p1)::($(typeof(a.p1))) end #without type information JIT will not compile this
    , quote end
    )
end

function de_jl_eval(a::DeReadOp,param,idxSym)
    @gensym r src
    ( r
    , eltype(a.p1.data)
    , quote ($src) = (($param).p1.data)::($(typeof(a.p1.data))) end #without type information JIT will not compile this
    , quote ($r) = ($src)[($idxSym)] end
    )
end

#function de_jl_eval(v::DeBinOp,param,idxSym)
function de_jl_eval{OT}(v::DeBinOp{OT},param,idxSym)
   @gensym r
   p1 = de_jl_eval(v.p1,quote ($param).p1 end,idxSym)
   p2 = de_jl_eval(v.p2,quote ($param).p2 end,idxSym)
   ( r
   , promote_type(p1[2],p2[2])
   , quote $(p1[3]);$(p2[3]) end
   , quote $(p1[4]);$(p2[4]);($r) = de_jl_do_op($OT,($(p1[1]))::($(p1[2])),($(p2[1]))::($(p2[2]))) end #type information does not make this faster
   )
end

for op = deBinOpList
  opType = de_op_to_type[op];
  opSingle = de_op_to_scaler[op];
  @eval de_jl_do_op(::Type{$opType},a,b) = ($opSingle)(a,b)
  #@eval function de_jl_do_op(::Type{$opType},a,b) ($opSingle)(a,b) end # does not work?
end

function assign(lhs::DeVecJ,rhs::DeExpr)
    println("Building New Assign ...");
    rhsSz = de_check_dims(rhs)
    lhsSz = size(lhs)

    if rhsSz != lhsSz
        error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
    end

    @gensym i prhs
    (rhsResult,rhsResultType,rhsPreamble,rhsKernel) = de_jl_eval(rhs,prhs,i)
    rhsType = typeof(rhs);

    global assign

    @eval function assign(plhs::DeVecJ,($prhs)::($rhsType))
        N = size(plhs,1)
        lhsData = plhs.data
        $rhsPreamble
        for ($i) = 1:N
            $rhsKernel
            lhsData[($i)] = ($rhsResult)::($rhsResultType) # this does not improve speed...
        end

        return plhs
    end

    return assign(lhs,rhs)
end

assign(lhs::DeArrJulia,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrJulia,rhs::Number) = assign(lhs,de_promote(rhs)...)


