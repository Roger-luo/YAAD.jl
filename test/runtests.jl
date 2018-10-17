using Test, YAAD, YAAD.TestUtils

Base.sin(x::AbstractNode) = register(Base.sin, x)
YAAD.gradient(::typeof(sin), grad, output, x) = (grad * cos(x), )

Base.:(*)(lhs::AbstractNode, rhs::AbstractNode) = register(Base.:(*), lhs, rhs)
YAAD.gradient(::typeof(*), grad, output, lhs, rhs) = grad * transpose(rhs), transpose(lhs) * grad

using LinearAlgebra
LinearAlgebra.tr(x::AbstractNode) = register(tr, x)
YAAD.gradient(::typeof(tr), grad, output, x) = (grad * Matrix(I, size(x)), )

x = Variable(rand(30, 30))
y = Variable(rand(30, 30))
z = x * y
z = tr(z)
backward(z)


y = sin(x)
backward(y)

# for fn in [sin, cos]
#     @test gradcheck(x->Broadcast.broadcasted(fn, x), Variable(rand(10)))
# end
x.grad
