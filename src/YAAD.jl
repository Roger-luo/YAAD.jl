module YAAD

using DiffRules

abstract type AbstractNode end
abstract type Operator end

# core
include("comput_graph.jl")

# operators
include("operators/broadcast.jl")
include("operators/math.jl")
include("operators/linalg.jl")
include("operators/array.jl")
include("operators/cat.jl")
include("operators/reduce.jl")
# include("operators/batched.jl")


# printings
include("show.jl")

# utils
include("utils.jl")
# test utils
include("test_utils.jl")


end # module
