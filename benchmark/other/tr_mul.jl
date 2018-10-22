using BenchmarkTools, Test, LinearAlgebra
import Zygote
import Zygote: @grad
import AutoGrad

Zygote.@grad LinearAlgebra.tr(x) = LinearAlgebra.tr(x), Δ-> (Δ * Matrix(I, size(x)), )

# function bench_tr_mul_base(x1, x2)
#     z1 = x1 * x2
#     z2 = tr(z1)
#
#     grad_z1 = Matrix{eltype(z1)}(I, size(z1))
#     grad_z1 * transpose(x2), transpose(x1) * grad_z1
# end
#
# function bench_tr_mul_base(x1, x2)
#     z1 = x1 * x2
#     z2 = tr(z1)
#
#     grads = YAAD.gradient(tr, one(eltype(x1)), z2, z1)
#     YAAD.gradient(*, first(grads), z1, x1, x2)
# end

function bench_tr_mul_autograd(x1, x2)
    z = AutoGrad.@diff tr(x1 * x2)
    AutoGrad.grad(z, x1), AutoGrad.grad(z, x2)
end

function bench_tr_mul_zygote(x1, x2)
    g = Zygote.gradient(()->tr(x1 * x2), Zygote.Params([x1, x2]))
    g[x1], g[x2]
end

xv, yv = rand(30, 30), rand(30, 30)

autograd_x = AutoGrad.Param(xv)
autograd_y = AutoGrad.Param(yv)

#  == bench_tr_mul_zygote(xv, yv)
println("----------------------------------------------------------------------")
println("AutoGrad:")
display(@benchmark bench_tr_mul_autograd(autograd_x, autograd_y))
println()

println()
println("----------------------------------------------------------------------")
println("Zygote:")
display(@benchmark bench_tr_mul_zygote(xv, yv))
