module YAAD

using DiffRules

abstract type AbstractNode end
abstract type Operator end

# core
include("comput_graph.jl")
# Base.sin(x::AbstractNode) = register(Base.sin, x)

# operators
include("operators/broadcast.jl")
include("operators/math.jl")
include("operators/linalg.jl")
include("operators/array.jl")
include("operators/cat.jl")


# printings
include("show.jl")

# test utils
include("test_utils.jl")


end # module
