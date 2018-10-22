using YAAD

a = Variable(rand(2, 2))
b = Variable(rand(2))

y = cat(a, b; dims=2)

y = hcat(a, b)
grads = cat(2*ones(2, 2), ones(2); dims=2)
uncat(2, grads, value(a), value(b))

backward(y, grads)
a.grad
b.grad
a = rand(2, 2)
b = rand(2, 2)
c = cat(a, b; dims=2)
o = uncat(2, c, a, b)

o[1] == a
o[2] == b

y = hcat(Float64[1, 2], Variable(Float64[2, 2]))
backward(y, ones(2, 2))

@which Base.cat_t(1, 1, [1, 2])
Base.cat_t
