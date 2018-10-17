Base.summary(io::IO, x::CachedNode) = (print(io, "(tracked) "); summary(io, x.output))
Base.show(io::IO, m::MIME"text/plain", x::CachedNode{<:Node, <:AbstractArray}) = (print(io, "(tracked) "); show(io, m, x.output))
# Base.show(io::IO, x::CachedNode) = show(io, x.output)

# printing
"""
    PrintTrait(node) -> Trait
"""
function PrintTrait end

struct ScalarVariable{T}
    value::T
end

# a mid type to forward variable to array's type tree
struct ArrayVariable{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    value::AT
end

# forward to default
PrintTrait(var::Number) = ScalarVariable(var)
PrintTrait(var::Variable{<:Number}) = ScalarVariable(var.value)
PrintTrait(var::CachedNode{<:AbstractNode, <:Number}) = ScalarVariable(var.output)
PrintTrait(var::Variable{AT}) where {T, N, AT <: AbstractArray{T, N}} = ArrayVariable{T, N, AT}(var.value)
PrintTrait(var::CachedNode{NT, AT}) where {T, N, NT, AT <: AbstractArray{T, N}} = ArrayVariable{T, N, AT}(var.output)

Base.size(x::ArrayVariable) = size(x.value)
Base.getindex(x::ArrayVariable, i...) = getindex(x.value, i...)

Base.show(io::IO, x::Union{Variable, CachedNode, Node}) = show(io, "text/plain", PrintTrait(x))
Base.show(io::IO, m::MIME"text/plain", x::Union{Variable, CachedNode, Node}) = show(io, m, PrintTrait(x))
Base.summary(io::IO, x::Union{Variable, CachedNode, Node}) = summary(io, PrintTrait(x))
Base.show(io::IO, x::ScalarVariable) = print(io, "(Tracked) ",x.value)
Base.summary(io::IO, x::ArrayVariable) = (print(io, "(Tracked) "); summary(io, x.value))
