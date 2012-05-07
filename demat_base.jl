## Delayed Expressions Base
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

## generic delayed execution types

abstract DeEle

abstract DeArr{B,T,N} <: DeEle

abstract DeExpr <: DeEle

type DeUniOp{OP,P1} <: DeExpr
  p1::P1
end

type DeBinOp{OP,P1,P2} <: DeExpr
  p1::P1
  p2::P2
end

type DeTriOp{OP,P1,P2,P3} <: DeExpr
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

type DeOpNull end
type DeOpAdd end
type DeOpSub end
type DeOpMulEle end
type DeOpDivEle end

const deBinOpList = (:+,:-,:.*,:./);

function de_op_to_type(op::Symbol)
  if op == :+ return DeOpAdd
  elseif op == :- return DeOpSub
  elseif op == :.* return DeOpMulEle
  elseif op == :./ return DeOpDivEle
  else return DeOpNull
  end
end

function de_op_to_scaler(op::Symbol)
  if op == :+ return :+
  elseif op == :- return :-
  elseif op == :.* return :*
  elseif op == :./ return :/
  else return DeOpNull
  end
end

for op = deBinOpList
  opType = de_op_to_type(op);
  @eval ($op)(a::DeEle,b::Number) = DeBinOp{$opType}(de_promote(a,b)...)
  @eval ($op)(a::Number,b::DeEle) = DeBinOp{$opType}(de_promote(a,b)...)
  @eval ($op)(a::DeEle,b::DeEle) =  DeBinOp{$opType}(de_promote(a,b)...)
  #@eval ($op)(ps...::BinOpParams) =  DeBinOp($op,de_promote(ps...)...)
end

