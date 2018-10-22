module YAAD

using DiffRules

abstract type AbstractNode end
abstract type Operator end

# This is copied from this PR
# https://github.com/JuliaLang/julia/pull/28182
include("weakkeyid_dict.jl")

# core
include("comput_graph.jl")
# Base.sin(x::AbstractNode) = register(Base.sin, x)

# operators
include("operators/broadcast.jl")
include("operators/math.jl")
include("operators/linalg.jl")
include("operators/array.jl")


# printings
# include("show.jl")

# test utils
include("test_utils.jl")


end # module
