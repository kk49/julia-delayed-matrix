## Delayed Expressions Demo
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

load("demat.jl")

## test code

function demat_test()
    N = 100000000
    a = rand(N)
    b = rand(N)
    c = rand(N)
    d = rand(N)
    ad = DeVecJ{Float64}(copy(a))
    bd = DeVecJ{Float64}(b)
    cd = DeVecJ{Float64}(c)
    dd = DeVecJ{Float64}(d)

    tN = 1
    println("-------------------")
    println("#1 Delayed Expression:")
    t1 = @elapsed for i = 1:tN ad[] = bd+cd.*dd + 1.0 end
    #@time ad[] = bd + 1.0 
    println("Elapsed time: ",t1)

    println("#2 Standard Julia Vector:")
    t2 = @elapsed for i = 1:tN a = b+c.*d + 1.0 end
    #@time a = b + 1.0
    println("Elapsed time: ",t2)

    println("#3 Standard Julia For Loop:")
    t3 = @elapsed for i = 1:tN 
        for j = 1:N
            a[j] = b[j] + c[j] * d[j] + 1.0
            #a[j] = b[j] + 1.0
        end
    end
    println("Elapsed time: ",t3)

    error = sum((a-ad.data).^2) / sum(a)

    println()
    println("Estimated overhead per expression == ",(t1-t3)/tN)

    println()
    println("error(sum((#3 - #1).^2) / abs(sum(#3)) == ",error)

    error
end

demat_test()

