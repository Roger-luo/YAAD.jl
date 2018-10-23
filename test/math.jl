using Test, YAAD, YAAD.TestUtils
using DiffRules, SpecialFunctions, NaNMath

@testset "broadcast math.jl" begin

# exclude function whose domain is not in [0, 1)
exclusion = Symbol[
    :asec,
    :acosh,
    :acsc,
    :acoth,
    :asecd,
    :acscd,
]

for (mod, name, nargs) in keys(DiffRules.DEFINED_DIFFRULES)
    f_ex_head = Expr(:., mod, QuoteNode(name))

    if nargs == 1
        name in exclusion && continue

        @eval @test gradcheck(x->Broadcast.broadcasted($f_ex_head, x), Variable(rand(10, 10)))
    end
end

@test gradcheck(x->Broadcast.broadcasted(asec, x), Variable(rand(10, 10) .+ 2.0))
@test gradcheck(x->Broadcast.broadcasted(acosh, x), Variable(rand(10, 10) .+ 2.0))
@test gradcheck(x->Broadcast.broadcasted(acsc, x), Variable(rand(10, 10) .+ 2.0))
@test gradcheck(x->Broadcast.broadcasted(acoth, x), Variable(rand(10, 10) .+ 2.0))
@test gradcheck(x->Broadcast.broadcasted(asecd, x), Variable(rand(10, 10) .+ 2.0))
@test gradcheck(x->Broadcast.broadcasted(acscd, x), Variable(rand(10, 10) .+ 2.0))

end # math.jl

gradcheck(x->Broadcast.broadcasted(Base.csch, x), Variable(rand(10, 10)))
