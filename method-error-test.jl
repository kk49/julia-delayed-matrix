abstract AT

type TT <: AT
  a
end

function +(a::AT,b::AT)
  a.a + b.a
end
