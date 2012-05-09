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

    r1 = 0
    for i = 1:1
        println("-------------------")
        println("#1 Delayed Expression:")
        @time ad[] = bd+cd.*dd + 1.0
        #@time ad[] = bd + 1.0 

        println("#2 Standard Julia Vector:")
        @time a = b+c.*d + 1.0
        #@time a = b + 1.0

        println("#3 Standard Julia For Loop:")
        @time for j = 1:N
            a[j] = b[j] + c[j] * d[j] + 1.0
            #a[j] = b[j] + 1.0
        end

        println()
        println("error(sum((#3 - #1).^2) / abs(sum(#3)) == ",sum((a-ad.data).^2) / sum(a))
    end

    r1
end

#demat_test()

function simple_test()
    #--------------------------------------
    function test1(x::Array{Float64})
       local s = 0
       local i
       for i = 1:1000000
           s += x[i] * i
       end
       s
    end
    #--------------------------------------
    @gensym ns ti
    @eval function test2(x::Array{Float64})
       ($ns) = x;
       local s = 0
       local ($ti)
       for ($ti) = 1:1000000
           s += ($ns)[($ti)] * ($ti)
       end
       s 
    end
    #--------------------------------------
    function extract_x3(x)
      @gensym r

      (r,quote ($r) = ($x) end)
    end

    function test3(x::Array{Float64})
       @gensym ti
       
       (rv,ex) = extract_x3(x)

       @eval function hf()
         local s = 0
         local ($ti)
         $ex
         for ($ti) = 1:1000000
           s += ($rv)[($ti)] * ($ti)
         end
         s
       end

       eval(hf)() #if eval is not here it returns the results for the last call to test3
    end
    #--------------------------------------
    function extract_x4(xi,idx)
      @gensym r src

      (r,quote ($src) = ($x) end,quote ($r) = ($src)[($idx)] end)
    end

    function test4(x::Array{Float64})
       @gensym ti
       
       (rv,ex,bd) = extract_x4(x,ti)

       @eval function hf()
         local s = 0
         local ($ti)
         $ex
         for ($ti) = 1:1000000
           $bd
           s += ($rv) * ($ti)
         end
         s
       end

       eval(hf)() #if eval is not here it returns the results for the last call to test3
    end
    #--------------------------------------

    x = randn(1000000)
    n = 20

    local s1,s2,s3,s4

    t1time = @elapsed for i = 1:n s1 = test1(x) end
    t2time = @elapsed for i = 1:n s2 = test2(x) end 
    t3time = @elapsed for i = 1:n s3 = test3(x) end 
    t4time = @elapsed for i = 1:n s4 = test4(x) end 

    println(" t1time: ",t1time," s: ",s1)
    println(" t2time: ",t2time," s: ",s2)
    println(" t3time: ",t3time," s: ",s3)
    println(" t4time: ",t4time," s: ",s4)
    println()
end

function stest()
   @gensym test1 test2
    
   @eval function ($test1)(a)
      s = 0
      for i = 1:size(a,1)
        s += *(a[i],i)
      end
      s
   end

   @gensym op 
   op = de_op_to_scaler(:.*)
   @eval function ($test2)(a)
      s = 0
      for i = 1:size(a,1)
        s += ($op)(a[i],i)
      end
      s
   end
   
   x = randn(1000000)
   N = 1 
   local s1,s2
   t1 = @elapsed for i = 1:N s1 = eval(test1)(x) end
   t2 = @elapsed for i = 1:N s2 = eval(test2)(x) end

   println("test func 1: time = ",t1," result = ",s1);
   println("test func 2: time = ",t2," result = ",s2);
   
end
