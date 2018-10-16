using Test, YAAD, YAAD.TestUtils

for fn in [sin, cos]
    @test gradcheck(x->Broadcast.broadcasted(fn, x), Variable(rand(10)))
end
