using Test, BenchmarkTools, LinearAlgebra
import YAAD

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

function bench_tr_mul_yaad(x1, x2)
    z = tr(x1 * x2)
    YAAD.backward(z)
    x1.grad, x2.grad
end

xv, yv = rand(30, 30), rand(30, 30)

yaad_x = YAAD.Variable(xv)
yaad_y = YAAD.Variable(yv)

println("----------------------------------------------------------------------")
println("YAAD:")
display(@benchmark bench_tr_mul_yaad(yaad_x, yaad_y))
println()
