using Revise, YAAD, BenchmarkTools

w = Variable(rand(100, 100))
b = Variable(rand(100))
x = rand(100)

z = sin.(w * x + b)

using LinearAlgebra

function bench_linear_base(w, b, x, grad_z)
    z1 = w * x
    z2 = z1 + b

    grad_z * transpose(x), grad_z, transpose(w) * grad_z
end

function bench_linear_yaad(w, b, x, grad_x)
    z = w * x + b
    backward(z, grad_x)
    w.grad, b.grad, x.grad
end

w = Variable(rand(100, 100))
b = Variable(rand(100))
x = Variable(rand(100))
grad_x = rand(100)

bench_linear_base(w.value, b.value, x.value, grad_x) == bench_linear_yaad(w, b, x, grad_x)

z1 = w * x
backward(z1, grad_x)
@which backward(z1, z1.node.f.f, grad_x)

gradient(z1, grad_x)[1] == grad_x * transpose(x.value)
w.grad == grad_x * transpose(x.value)


@benchmark bench_linear_base(w.value, b.value, x.value, grad_x)
@benchmark bench_linear_yaad(w, b, x, grad_x)

@benchmark backward(w, $(rand(100, 100)))
@benchmark backward(b, $(rand(100)))

129 / 1000

25 - 0.129 * 2 - 11

@profiler for i in 1:100000
    bench_linear_yaad(w, b, x, grad_x)
end

flux_w = Flux.param(rand(100, 100))
flux_b = Flux.param(rand(100))
flux_x = rand(100)

Flux.Dense(100, 100, sin)(rand(100))
