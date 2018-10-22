using BenchmarkTools, LinearAlgebra
import Zygote
import Zygote: @grad
using Test
import AutoGrad, YAAD

Zygote.@grad LinearAlgebra.tr(x) = LinearAlgebra.tr(x), Δ-> (Δ * Matrix(I, size(x)), )

function bench_tr_mul_base(x1, x2)
    z1 = x1 * x2
    z2 = tr(z1)

    grad_z1 = Matrix{eltype(z1)}(I, size(z1))
    grad_z1 * transpose(x2), transpose(x1) * grad_z1
end

function bench_tr_mul_base(x1, x2)
    z1 = x1 * x2
    z2 = tr(z1)

    grads = YAAD.gradient(tr, one(eltype(x1)), z2, z1)
    YAAD.gradient(*, first(grads), z1, x1, x2)
end

function bench_tr_mul_yaad(x1, x2)
    z = tr(x1 * x2)
    YAAD.backward(z)
    x1.grad, x2.grad
end

function bench_tr_mul_autograd(x1, x2)
    z = AutoGrad.@diff tr(x1 * x2)
    AutoGrad.grad(z, x1), AutoGrad.grad(z, x2)
end

function bench_tr_mul_zygote(x1, x2)
    g = Zygote.gradient(()->tr(x1 * x2), Zygote.Params([x1, x2]))
    g[x1], g[x2]
end

xv, yv = rand(30, 30), rand(30, 30)

yaad_x = YAAD.Variable(xv)
yaad_y = YAAD.Variable(yv)

function bench_yaad(x, y)
    for i = 1:1000
        bench_tr_mul_yaad(x, y)
    end
end

function bench_base(x, y)
    for i = 1:1000
        bench_tr_mul_base(x, y)
    end
end

bench_tr_mul_base(xv, yv)

@profiler bench_yaad(yaad_x, yaad_y)
@profiler bench_base(xv, yv)

autograd_x = AutoGrad.Param(xv)
autograd_y = AutoGrad.Param(yv)

println("Check gradient:")

@test bench_tr_mul_yaad(yaad_x, yaad_y) == bench_tr_mul_autograd(autograd_x, autograd_y)

#  == bench_tr_mul_zygote(xv, yv)
println("----------------------------------------------------------------------")
println("YAAD:")
display(@benchmark bench_tr_mul_yaad(yaad_x, yaad_y))

println()
println("----------------------------------------------------------------------")
println("AutoGrad:")
display(@benchmark bench_tr_mul_autograd(autograd_x, autograd_y))

println()
println("----------------------------------------------------------------------")
println("Zygote:")
display(@benchmark bench_tr_mul_zygote(xv, yv))
