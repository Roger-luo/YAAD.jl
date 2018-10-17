using BenchmarkTools, AutoGrad

function bench_tr_mul_yaad(x1, x2)
    z = tr(x1 * x2)
    backward(z)
    x1.grad, x2.grad
end

function bench_tr_mul_autograd(x1, x2)
    z = @diff tr(x1 * x2)
    grad(z, x1), grad(z, x2)
end

x = Variable(rand(30, 30))
y = Variable(rand(30, 30))

@benchmark bench_tr_mul_yaad(x, y)

x = Param(rand(30, 30))
y = Param(rand(30, 30))

@benchmark bench_tr_mul_autograd(x, y)
