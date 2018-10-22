using Test, YAAD, YAAD.TestUtils

@test "index and slice" begin
    gradcheck(x->x[:, 2, :, 1], Variable(rand(2, 2, 2, 3)))
    gradcheck(x->view(x, :, :, :, 1), Variable(rand(2, 2, 2, 3)))
    gradcheck(x->selectdim(x, 2, 1), Variable(rand(2, 2, 2, 3)))
end
