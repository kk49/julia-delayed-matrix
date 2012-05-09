load("demat.jl")

a = randn(1000000);
ad = DeVecJ{Float64}(a); 
bd =DeVecJ{Float64}(copy(a));
finfer(assign,(ad,ad+1))

