using Test, YAAD, YAAD.TestUtils
using LinearAlgebra

@testset "linalg.jl" begin

@test gradcheck(tr, Variable(rand(10, 10)))
@test gradcheck(*, Variable(rand(10, 10)), Variable(rand(10, 10)))
@test gradcheck(transpose, Variable(rand(10, 10)))

end
