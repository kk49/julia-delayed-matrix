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

de_check_dims(a::DeConst) = ()
de_check_dims(a::DeReadOp) = size(a.p1)
de_check_dims(a::DeUniOp) = de_check_dims(a.p1)
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


function DeBuildBinOp(OP,p...)
  pp = de_promote(p...)
  return DeBinOp{OP,typeof(pp[1]),typeof(pp[2])}(pp...)
end
   

for op = deBinOpList
  opType = de_op_to_type[op];
  #@eval ($op)(a::DeEle,b::Number) = DeBinOp{$opType}(de_promote(a,b)...)
  #@eval ($op)(a::Number,b::DeEle) = DeBinOp{$opType}(de_promote(a,b)...)
  #@eval ($op)(a::DeEle,b::DeEle)  = DeBinOp{$opType}(de_promote(a,b)...)
  @eval ($op)(a::DeEle,b::Number) = DeBuildBinOp($opType,a,b)
  @eval ($op)(a::Number,b::DeEle) = DeBuildBinOp($opType,a,b)
  @eval ($op)(a::DeEle,b::DeEle)  = DeBuildBinOp($opType,a,b)
end

