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
    #the following should be included somehow in the time below...
    acReadback = copy(a);
    ac = DeVecCu{Float32}(a);
    bc = DeVecCu{Float32}(b);
    cc = DeVecCu{Float32}(c);
    dc = DeVecCu{Float32}(d);

    tN = 10
    println("ASSIGN TEST")
    println("running $tN iterations of $N sized for loops...")
    println("-------------------")
    println("#1 Standard Julia For Loop:")
    t1 = @elapsed for i = 1:tN 
        for j = 1:N
            a[j] = b[j] + c[j] * d[j] + float32(1.0)
            #a[j] = b[j] + 1.0
        end
    end
    println("Elapsed time: $t1")

    println("#1a Standard Julia For Loop (in Function):")
    function testRun(ai,bi,ci,di)
        for j = 1:N
            ai[j] = bi[j] + ci[j] * di[j] + float32(1.0)
            #a[j] = b[j] + 1.0
        end
    end
    
    t1a = @elapsed for i = 1:tN
        testRun(a,b,c,d);
    end
    println("Elapsed time: $t1a")

    println("#2 Standard Julia Vector:")
    t2 = @elapsed for i = 1:tN a = b+c.*d + float32(1.0) end
    #@time a = b + 1.0
    println("Elapsed time: $t2")


    println("#3 Delayed Expression (Julia):")
    et = @elapsed ad[] = bd+cd.*dd + float32(1.0) # give delayed expression a change to build the function
    println("  Time to build (if necessary) and run one iteration: $et seconds")
    et = @elapsed ad[] = bd+cd.*dd + float32(1.0) 
    println("  Time to only run one iteration: $et seconds")
    et = @elapsed ad[] = bd+cd.*dd + float32(2.0) 
    println("  Time to only run one iteration (Add 2 instead of 1): $et seconds")
    et = @elapsed ad[] = bd + float32(1.0) 
    println("  Time to only run one iteration (replace second operand of top operator): $et seconds")

    t3 = @elapsed for i = 1:tN ad[] = bd+cd.*dd + float32(1.0) end
    #@time ad[] = bd + 1.0 
    println("Elapsed time: $t3")

    println("#4 Delayed Expression (CUDA):")
    t4 = @elapsed for i = 1:tN ac[] = bc+cc.*dc + float32(1.0) end
    #@time ad[] = bd + 1.0
    t4rb = @elapsed acReadback[] = ac; # readback data 
    println("Elapsed time: ",t4," Readback time: ",t4rb)


    errorDej = 0
    errorDec = 0
    errorDej = sum((a-ad.data).^2) / sum(a)
    errorDec = sum((a-acReadback).^2) / sum(a)

    println()
    println("Estimated overhead per expression == $((t3-t1)/tN)")

    println()
    println("error(sum((#1 - DeJulia).^2) / abs(sum(#1)) == $errorDej")

    println("error(sum((#1 - DeCuda).^2) / abs(sum(#1)) == ",errorDec)
    println()


    println("SUM TEST")
    r1 = 0.0;
    et = @elapsed for i = 1:tN r1 = r1 + sum(d) end
    println("Julia sum(x): Time $et Result $r1")
    r1 = 0.0;
    et = @elapsed for i = 1:tN r1 = r1 + sum(dd) end
    println("Delay Expression (Julia) sum(x): Time $et Result $r1")
    
    r1 = 0.0;
    et = @elapsed for i = 1:tN r1 = r1 + sum(d .* d) end
    println("Julia sum(x .* x): Time $et Result $r1")
    r1 = 0.0;
    #et = @elapsed for i = 1:tN r1 = r1 + sum(dd - float32(1.0)) end
    et = @elapsed for i = 1:tN r1 = r1 + sum(cd .* cd) end
    println("Delay Expression (Julia) sum(x .* x): Time $et Result $r1")

    return (errorDej,errorDec)
end

demat_test()

