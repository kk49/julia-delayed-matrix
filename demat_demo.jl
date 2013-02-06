## Delayed Expressions Demo
# Copyright 2012-2013, Krzysztof Kamieniecki (krys@kamieniecki.com)

require("demat.jl")

## test code

function demat_test()
    N = 10000000
    a0 = convert(Array{Float32,1},rand(N))
    a1 = convert(Array{Float32,1},rand(N))
    a2 = convert(Array{Float32,1},rand(N))
    a3 = convert(Array{Float32,1},rand(N))
    a4 = convert(Array{Float32,1},rand(N))
    a5 = convert(Array{Float32,1},rand(N))
    a6 = convert(Array{Float32,1},rand(N))
    a7 = convert(Array{Float32,1},rand(N))

    d0 = DeVecJ{Float32}(copy(a0))
    d1 = DeVecJ{Float32}(a1)
    d2 = DeVecJ{Float32}(a2)
    d3 = DeVecJ{Float32}(a3)
    d4 = DeVecJ{Float32}(a4)
    d5 = DeVecJ{Float32}(a5)
    d6 = DeVecJ{Float32}(a6)
    d7 = DeVecJ{Float32}(a7)

    #the following should be included somehow in the time below...
    c0Readback = copy(a0);
    c0 = DeVecCu{Float32}(a0);
    c1 = DeVecCu{Float32}(a1);
    c2 = DeVecCu{Float32}(a2);
    c3 = DeVecCu{Float32}(a3);
    c4 = DeVecCu{Float32}(a4);
    c5 = DeVecCu{Float32}(a5);
    c6 = DeVecCu{Float32}(a6);
    c7 = DeVecCu{Float32}(a7);

    tN = 10
    println("ASSIGN TEST")
    println("running $tN iterations of $N sized for loops...")
    println("-------------------")
    println("#1 Standard Julia For Loop:")
    t1 = @elapsed for i = 1:tN 
        for j = 1:N
            a0[j] = a1[j] + a2[j] * a3[j] + 1.0f0
            #a[j] = b[j] + 1.0
        end
    end
    println("Elapsed time: $t1")

    println("#1a Standard Julia For Loop (in Function):")
    function testRun(i0,i1,i2,i3)
        for j = 1:N
            i0[j] = i1[j] + i2[j] * i3[j] + 1.0f0
            #a[j] = b[j] + 1.0
        end
    end
    
    t1a = @elapsed for i = 1:tN
        testRun(a0,a1,a2,a3);
    end
    println("Elapsed time: $t1a")

    println("#2 Standard Julia Vector:")
    t2 = @elapsed for i = 1:tN a0 = a1+a2.*a3 + 1.0f0 end
    #@time a = b + 1.0
    println("Elapsed time: $t2")


    println("#3 Delayed Expression (Julia):")
#    et = @elapsed ad[] = bd+cd.*dd + float32(1.0) # give delayed expression a change to build the function
#    println("  Time to build (if necessary) and run one iteration: $et seconds")
#    et = @elapsed ad[] = bd+cd.*dd + float32(1.0) 
#    println("  Time to only run one iteration: $et seconds")
#    et = @elapsed ad[] = bd+cd.*dd + float32(2.0) 
#    println("  Time to only run one iteration (Add 2 instead of 1): $et seconds")
#    et = @elapsed ad[] = bd + float32(1.0) 
#    println("  Time to only run one iteration (replace second operand of top operator): $et seconds")

    t3 = @elapsed for i = 1:tN d0[] = d1+d2.*d3 + 1.0f0 end
    #@time ad[] = bd + 1.0 
    println("Elapsed time: $t3")

    println("#4 Delayed Expression (CUDA):")
    t4 = @elapsed for i = 1:tN c0[] = c1+c2.*c3 + 1.0f0 end
    #@time ad[] = bd + 1.0f0
    t4rb = @elapsed c0Readback[] = c0; # readback data 
    println("Elapsed time: ",t4," Readback time: ",t4rb)

    errorDej = 0
    errorDec = 0
    errorDej = sum((a0-d0.data).^2) / sum(a0)
    errorDec = sum((a0-c0Readback).^2) / sum(a0)

    println()
    println("error(sum((#1 - DeJulia).^2) / abs(sum(#1)) == $errorDej")

    println("error(sum((#1 - DeCuda).^2) / abs(sum(#1)) == ",errorDec)
    println()

    ########################
    tN = 10
    println("COMPLICATED TEST")
    println("running $tN iterations of $N sized for loops...")
    println("-------------------")
    println("#1 Standard Julia For Loop:")
    t1 = @elapsed for i = 1:tN 
        for j = 1:N
            a0[j] = sin(a1[j] + cos(a2[j] .* exp(a3[j] + a4[j] + a5[j] .* log(a6[j] + a7[j]))))
        end
    end
    println("Elapsed time: $t1")

    println("#1a Standard Julia For Loop (in Function):")
    function testRun2(i0,i1,i2,i3,i4,i5,i6,i7)
        for j = 1:N
            i0[j] = sin(i1[j] + cos(i2[j] .* exp(i3[j] + i4[j] + i5[j] .* log(i6[j] + i7[j]))))
        end
    end
    
    t1a = @elapsed for i = 1:tN
        testRun2(a0,a1,a2,a3,a4,a5,a6,a7);
    end
    println("Elapsed time: $t1a")

    println("#2 Standard Julia Vector:")
    t2 = @elapsed for i = 1:tN a0 = sin(a1 + cos(a2 .* exp(a3 + a4 + a5 .* log(a6 + a7)))) end
    println("Elapsed time: $t2")


    println("#3 Delayed Expression (Julia):")
    t3 = @elapsed for i = 1:tN d0[] = sin(d1 + cos(d2 .* exp(d3 + d4 + d5 .* log(d6 + d7)))) end
    #@time ad[] = bd + 1.0 
    println("Elapsed time: $t3")


    errorDej = 0
    errorDej = sum((a0-d0.data).^2) / sum(a0)

    println()
    println("error(sum((#1 - DeJulia).^2) / abs(sum(#1)) == $errorDej")

    ########################
    println("SUM TEST")
    r1 = 0.0;
    et = @elapsed for i = 1:tN r1 = r1 + sum(a4) end
    println("Julia sum(x): Time $et Result $r1")
    r1 = 0.0;
    et = @elapsed for i = 1:tN r1 = r1 + sum(d4) end
    println("Delay Expression (Julia) sum(x): Time $et Result $r1")
    
    r1 = 0.0;
    et = @elapsed for i = 1:tN r1 = r1 + sum(a4 .* a4) end
    println("Julia sum(x .* x): Time $et Result $r1")
    r1 = 0.0;
    #et = @elapsed for i = 1:tN r1 = r1 + sum(dd - 1.0f0) end
    et = @elapsed for i = 1:tN r1 = r1 + sum(d4 .* d4) end
    println("Delay Expression (Julia) sum(x .* x): Time $et Result $r1")

    return (errorDej,errorDec)
end

demat_test()

