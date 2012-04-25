## Delayed Expressions Base
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

## generic delayed execution types

abstract DeEle

abstract DeArr{B,T,N} <: DeEle

abstract DeExpr <: DeEle

type DeUniOp{OP,P1} <: DeExpr
  op::OP
  p1::P1
end

type DeBinOp{OP,P1,P2} <: DeExpr
  op::OP
  p1::P1
  p2::P2
end

type DeTriOp{OP,P1,P2,P3} <: DeExpr
  op::OP
  p1::P1
  p2::P2
  p3::P3
end

type DeConst{P1} <: DeExpr
  p1::P1
end

type DeReadOp{P1} <: DeExpr
  p1::P1
end

type DeWriteOp{P1} <: DeExpr
  p1::P1
end

de_promote(x::Number) = (DeConst(x),)
de_promote(x::DeArr) = (DeReadOp(x),)
de_promote(x::DeExpr) = (x,)
de_promote(x,xs...) = tuple(de_promote(x)...,de_promote(xs...)...)

typealias BinOpParams Union((DeEle,Number),(Number,DeEle),(DeEle,DeEle))

for op = (:+,:-,:.*,:./)
  @eval ($op)(a::DeEle,b::Number) = DeBinOp($op,de_promote(a,b)...)
  @eval ($op)(a::Number,b::DeEle) = DeBinOp($op,de_promote(a,b)...)
  @eval ($op)(a::DeEle,b::DeEle) =  DeBinOp($op,de_promote(a,b)...)
  #@eval ($op)(ps...::BinOpParams) =  DeBinOp($op,de_promote(ps...)...)
end

