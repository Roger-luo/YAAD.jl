using Test, YAAD, YAAD.TestUtils

@testset "index and slice" begin
    @test gradcheck(x->x[:, 2, :, 1], Variable(rand(2, 2, 2, 3)))
    @test gradcheck(x->view(x, :, :, :, 1), Variable(rand(2, 2, 2, 3)))
    @test gradcheck(x->selectdim(x, 2, 1), Variable(rand(2, 2, 2, 3)))
end
