using BenchmarkTools, LinearAlgebra
import Zygote
import Zygote: @grad
using Test
import AutoGrad, YAAD

Zygote.@grad LinearAlgebra.tr(x) = LinearAlgebra.tr(x), Δ-> (Δ * Matrix(I, size(x)), )
Zygote.@grad Base.:(*)(lhs::Matrix, rhs::Matrix) = gemm(lhs, rhs), grad -> (grad * transpose(rhs), transpose(lhs) * grad)

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
    Zygote.gradient((x1, x2)->tr(x1 * x2), x1, x2)
end

xv, yv = rand(30, 30), rand(30, 30)

yaad_x = YAAD.Variable(xv)
yaad_y = YAAD.Variable(yv)

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
