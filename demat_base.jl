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

type DeOpAdd end
type DeOpSub end
type DeOpMulEle end
type DeOpDivEle end

const deBinOpList = (:+,:-,:.*,:./);

de_op_to_type = dict((:+,),(DeOpAdd,))
de_op_to_type[:-] = DeOpSub
de_op_to_type[:.*] = DeOpMulEle
de_op_to_type[:./] = DeOpDivEle

de_op_to_scaler = dict((:+,),(:+,))
de_op_to_scaler[:-] = :-
de_op_to_scaler[:.*] = :*
de_op_to_scaler[:./] = :/

for op = deBinOpList
  opType = de_op_to_type[op];
  @eval ($op)(a::DeEle,b::Number) = DeBinOp{$opType}(de_promote(a,b)...)
  @eval ($op)(a::Number,b::DeEle) = DeBinOp{$opType}(de_promote(a,b)...)
  @eval ($op)(a::DeEle,b::DeEle) =  DeBinOp{$opType}(de_promote(a,b)...)
end

