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

function de_eval(a::DeConst)
  ((p,idx)->a.p1,())
end

function de_eval(a::DeReadOp)
  ((p,idx)->p[idx],a.p1.data)
end

function de_eval(a::DeUniOp)
  opf = eval(a.op)
  (p1f,p1p) = de_eval(a.p1)
  ((p,idx)->opf(p1f(p,idx)),p1p) 
end

function de_eval(a::DeBinOp) 
  opf = eval(a.op)
  (p1f,p1p) = de_eval(a.p1)
  (p2f,p2p) = de_eval(a.p2)
  ((p,idx)->opf(p1f(p[1],idx),p2f(p[2],idx)),(p1p,p2p))
end

function assign(lhs::DeArrJulia,rhs::DeExpr)
  tic()
  rhsSz = de_check_dims(rhs)
  lhsSz = size(lhs)

  if rhsSz != lhsSz
    error("src & dst size does not match. NOT IMPLEMENTED FOR SCALARS FIX")
  end
  
  (fcall,fdata) = de_eval(rhs)
  println("Delayed Expression Setup Time ",toq())
  tic()
  for i = 1:lhsSz[1]
     lhs.data[i] = fcall(fdata,i)
  end 
  println("Delayed Expression Execution Time ",toq()) 
end

assign(lhs::DeArrJulia,rhs::DeEle) = assign(lhs,de_promote(rhs)...)
assign(lhs::DeArrJulia,rhs::Number) = assign(lhs,de_promote(rhs)...)


