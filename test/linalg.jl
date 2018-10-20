using Test, YAAD, YAAD.TestUtils
using LinearAlgebra

x1 = Variable(rand(10, 10))
x2 = Variable(rand(10, 10))
z = tr(x1 * x2)

backward(z)

y = 2 * x1
z = tr(y)

backward(z)
x1.grad
tr(x1)

gradcheck(tr, Variable(rand(10, 10)))
