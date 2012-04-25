## Delayed Expressions Demo
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

load("demat.jl")

## test code
function demat_test()
    N = 1000000
    a = rand(N)
    b = rand(N)
    c = rand(N)
    d = rand(N)
    ad = DeVecJ{Float64}(a)
    bd = DeVecJ{Float64}(b)
    cd = DeVecJ{Float64}(c)
    dd = DeVecJ{Float64}(d)

    for i = 1:4
        println("-------------------")
        tic()
        ad[] = bd+cd.*dd
        tr = toq()
        println("Delayed Expression Total Time ",tr)

        tic()
        a = b+c.*d
        tr = toq()
        println("Standard Julia Vector Execution Time ",tr)

        tic()
        for j = 1:N
        a[j] = b[j] + c[j] .* d[j]
        end
        tr = toq()
        println("Standard Julia For Loop Execution Time ",tr)
    end
end

demat_test()
