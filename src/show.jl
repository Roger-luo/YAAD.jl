# Base.summary(io::IO, x::CachedNode) = (print(io, "(Tracked) "); summary(io, x.output))
# Base.show(io::IO, m::MIME"text/plain", x::CachedNode{<:Node, <:AbstractArray}) = (print(io, "(tracked) "); show(io, m, x.output))
# Base.show(io::IO, x::CachedNode) = show(io, x.output)

# printing
"""
    PrintTrait(node) -> Trait
"""
function PrintTrait end

struct ScalarVariable{T}
    value::T
end

# a trait type to forward variable to array's type tree
struct ArrayVariable{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    value::AT
end

struct Other{NT}
    node::NT
end

# forward to default
PrintTrait(var::Number) = ScalarVariable(var)
PrintTrait(var::Value{<:Number}) = ScalarVariable(value(var))
PrintTrait(var::Value{AT}) where {T, N, AT <: AbstractArray{T, N}} = ArrayVariable{T, N, AT}(value(var))
PrintTrait(var::Value{T}) where {T} = Other(var)

Base.size(x::ArrayVariable) = size(x.value)
Base.getindex(x::ArrayVariable, i...) = getindex(x.value, i...)

Base.show(io::IO, x::Value) = show(io, "text/plain", PrintTrait(x))
Base.show(io::IO, m::MIME"text/plain", x::Value) = show(io, m, PrintTrait(x))
Base.summary(io::IO, x::Union{Variable, CachedNode, Node}) = summary(io, PrintTrait(x))
Base.show(io::IO, x::ScalarVariable) = print(io, "(Tracked) ",x.value)
Base.summary(io::IO, x::ArrayVariable) = (print(io, "(Tracked) "); summary(io, x.value))

function Base.show(io::IO, ::MIME"text/plain", x::Other{<:CachedNode})
    println(io, "CachedNode:")
    println(io, "       f: ", operator(x.node))
    println(io, "    args: ", args(x.node))
    println(io, "  kwargs: ", kwargs(x.node))
      print(io, "  output: ", x.node.output)
end

function Base.show(io::IO, ::MIME"text/plain", x::Other{<:Variable})
    println(io, "Variable:")
    println(io, "  value: ", x.node.value)

    if isdefined(x.node, :grad)
        println(io, "   grad: ", x.node.grad)
    end
end
