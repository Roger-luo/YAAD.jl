using BenchmarkTools, LinearAlgebra
import Zygote
import Zygote: @grad

import AutoGrad, YAAD

Zygote.@grad LinearAlgebra.tr(x) = LinearAlgebra.tr(x), Δ-> (Δ * Matrix(I, size(x)), )

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

x = YAAD.Variable(rand(30, 30))
y = YAAD.Variable(rand(30, 30))

println("----------------------------------------------------------------------")
println("YAAD:")
display(@benchmark bench_tr_mul_yaad(x, y))

x = AutoGrad.Param(rand(30, 30))
y = AutoGrad.Param(rand(30, 30))

println("----------------------------------------------------------------------")
println("AutoGrad:")
display(@benchmark bench_tr_mul_autograd(x, y))

println("----------------------------------------------------------------------")
println("Zygote:")
display(@benchmark bench_tr_mul_zygote($(rand(30, 30)), $(rand(30, 30))))
