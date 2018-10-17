using Test, YAAD, YAAD.TestUtils

x = Variable(rand(10, 10))
y = sin.(x)

backward(y, ones(10, 10))

gradcheck(x->Broadcast.broadcasted(sin, x), Variable(rand(10, 10)))
