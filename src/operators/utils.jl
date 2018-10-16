const ArrayNode = Union{Variable{<:AbstractArray}, CachedNode{<:Node, <:AbstractArray}}
const MatrixNode = Union{Variable{<:AbstractMatrix}, CachedNode{<:Node, <:AbstractMatrix}}
const NumberNode = Union{Variable{<:Number}, CachedNode{<:Node, <:Number}}
