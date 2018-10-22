using Test, YAAD, YAAD.TestUtils

@testset "cat" begin

@test gradcheck((x, y)->hcat(x, y), Variable(rand(2, 2)), Variable(rand(2)))
@test gradcheck((x, y)->cat(x, y; dims=2), Variable(rand(2, 2)), Variable(rand(2)))

end
