module YAAD

using DiffRules

abstract type AbstractNode end
abstract type Operator end

# utils
# include("function_trait.jl")

# core
include("variable.jl")
include("comput_graph.jl")

# operators
include("operators/utils.jl")
include("operators/broadcast.jl")
include("operators/math.jl")
include("operators/linalg.jl")
include("operators/array.jl")


# printings
include("show.jl")

# test utils
include("test_utils.jl")


end # module
