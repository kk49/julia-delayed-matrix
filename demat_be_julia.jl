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

function de_check_dims(a::DeConst)
  ()
end

function de_check_dims(a::DeReadOp)
  size(a.p1)
end

function de_check_dims(a::DeUniOp)
  de_check_dims(a.p1)
end

function de_check_dims(a::DeBinOp)
  r1 = de_check_dims(a.p1)
  r2 = de_check_dims(a.p2)

  if length(r1) == 0
    return r2
  elseif length(r2) == 0
    return r1
  elseif r1 == r2
    return r1
  else
    error("BinOp Parameters do not match")
  end
end

# de_eval returns a 3-tuple that contains
#   symbol that contains the value of the extression
#   the quoted preable code
#   the quoted kernal code
#de_eval(a,idxSym) = de_eval(a,idxSym,x->x)

function de_eval(a::DeConst,idxSym::Symbol)
    @gensym r
    ( r
    , quote ($r) = ($(a.p1)) end
    , quote end
    )
end

function de_eval(a::DeReadOp,idxSym::Symbol)
    @gensym r src
    ( r
    , quote ($src) = ($a).p1.data end
    , quote ($r) = ($src)[($idxSym)] end
    )
end

for op = deBinOpList
  opType = de_op_to_type(op);
  opSingle = de_op_to_scaler(op);
  @eval function de_eval(v::DeBinOp{$opType},idxSym::Symbol)
      @gensym r
      p1 = de_eval(v.p1,idxSym)
      p2 = de_eval(v.p2,idxSym)
      preamble = quote $(p1[2]);$(p2[2]) end
      kernel = quote $(p1[3]);$(p2[3]);($r) = ($($opSingle))(($(p1[1])),($(p2[1]))) end
      ( r
      , preamble
      , kernel
      )
  end
end

function assign(lhs::DeArrJulia,rhs::DeExpr)
    #println("Delayed Expression Setup Time:")
    @gensym i
    rhsSz = de_check_dims(rhs)
    lhsSz = size(lhs)

    if rhsSz != lhsSz
        error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
    end

    @gensym i hiddenFunc
    (rhsResult,rhsPreamble,rhsKernel) = de_eval(rhs,i)
    rhsType = typeof(rhs);

    ex = quote function ($hiddenFunc)(plhs::DeArrJulia,prhs::($rhsType)) 
        N = size(plhs,1)
        lhsData = plhs.data
        $rhsPreamble
        for ($i) = 1:N
            $rhsKernel
            lhsData[($i)] = ($rhsResult)
        end
    end
    end

    println(rhsResult)
    println(rhsPreamble)
    println(rhsKernel)
    println()
    println(ex)
 
    eval(ex)
    @time ((eval(hiddenFunc))(lhs,rhs))
end

assign(lhs::DeArrJulia,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrJulia,rhs::Number) = assign(lhs,de_promote(rhs)...)


