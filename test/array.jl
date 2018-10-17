using YAAD

t = Variable(rand(2, 2, 2, 3))

view(t, :, :, :, 1)
