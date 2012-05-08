## Delayed Expressions Demo
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

load("demat.jl")

## test code

function demat_test()
    N = 10000000
    a = rand(N)
    b = rand(N)
    c = rand(N)
    d = rand(N)
    ad = DeVecJ{Float64}(copy(a))
    bd = DeVecJ{Float64}(b)
    cd = DeVecJ{Float64}(c)
    dd = DeVecJ{Float64}(d)

    r1 = 0
    for i = 1:1
        println("-------------------")
        println("#1 Delayed Expression:")
        @time ad[] = bd+cd.*dd + 1.0

        println("#2 Standard Julia Vector:")
        @time a = b+c.*d + 1.0

        println("#3 Standard Julia For Loop:")
        @time for j = 1:N
            a[j] = b[j] + c[j] * d[j] + 1.0
        end

        println()
        println("error(sum((#3 - #1).^2) / abs(sum(#3)) == ",sum((a-ad.data).^2) / sum(a))
    end

    r1
end

demat_test()

