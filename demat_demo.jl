## Delayed Expressions Demo
# Copyright 2012, Krzysztof Kamieniecki (krys@kamieniecki.com)

load("demat.jl")

## test code

function demat_test()
    N = 10000000
    a = convert(Array{Float32,1},rand(N))
    b = convert(Array{Float32,1},rand(N))
    c = convert(Array{Float32,1},rand(N))
    d = convert(Array{Float32,1},rand(N))
    ad = DeVecJ{Float32}(copy(a))
    bd = DeVecJ{Float32}(b)
    cd = DeVecJ{Float32}(c)
    dd = DeVecJ{Float32}(d)
    ac = DeVecCu{Float32}(a);
    bc = DeVecCu{Float32}(b);
    cc = DeVecCu{Float32}(c);
    dc = DeVecCu{Float32}(d);

    tN = 10
    println("-------------------")
    println("#1 Standard Julia For Loop:")
    t1 = @elapsed for i = 1:tN 
        for j = 1:N
            a[j] = b[j] + c[j] * d[j] + 1.0
            #a[j] = b[j] + 1.0
        end
    end
    println("Elapsed time: ",t1)

    println("#2 Standard Julia Vector:")
    t2 = @elapsed for i = 1:tN a = b+c.*d + 1.0 end
    #@time a = b + 1.0
    println("Elapsed time: ",t2)


    println("#3 Delayed Expression (Julia):")
#    ad[] = bd+cd.*dd + 1.0 # give delayed expression a change to build the function
    t3 = @elapsed for i = 1:tN ad[] = bd+cd.*dd + 1.0 end
    #@time ad[] = bd + 1.0 
    println("Elapsed time: ",t3)

#    println("#4 Delayed Expression (CUDA):")
#    t4 = @elapsed for i = 1:tN ac[] = bc+cc.*dc + 1.0 end
#    #@time ad[] = bd + 1.0 
#    println("Elapsed time: ",t4)



    error = sum((a-ad.data).^2) / sum(a)

    println()
    println("Estimated overhead per expression == ",(t3-t1)/tN)

    println()
    println("error(sum((#1 - #3).^2) / abs(sum(#1)) == ",error)

    error
end

demat_test()

