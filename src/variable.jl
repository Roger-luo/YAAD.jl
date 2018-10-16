export Variable, value

mutable struct Variable{T} <: AbstractNode
    # # flags
    # retain_grad::Bool # retain grad
    # requires_grad::Bool # requires_grad

    value::T
    grad::T

    Variable(x::T) where T = new{T}(x)
    Variable(x::T, grad::T) where T = new{T}(x, grad)
end

value(x::Variable) = x.value
value(x) = x
value_type(x::Variable{T}) where T = T
value_type(::Type{Variable{T}}) where T = T

# 1. Array Variables
# inherit array interface
const ArrayVar{T, N} = Variable{AT} where {T, N, AT <: AbstractArray{T, N}}

Base.size(x::ArrayVar) = size(x.value)
Base.size(x::ArrayVar, d::Int) = size(x.value, d)

# Base.getindex(x::ArrayVar, i...) = getindex(x.value, i...)
# Base.setindex!(x::ArrayVar, v, i...) = setindex!(x.value, v, i...)

Base.IndexStyle(::Type{T}) where {T <:ArrayVar} = IndexStyle(value_type(T))

Base.similar(x::ArrayVar) = Variable(similar(x))
Base.similar(x::ArrayVar, dims::Dims) = Variable(similar(x.value, dims))
Base.similar(x::ArrayVar, element_type::Type{S}, dims::Dims) where S = Variable(similar(x.value, element_type, dims))

Base.axes(x::ArrayVar) = axes(x.value)
Base.eltype(x::Variable) = eltype(x.value)

# printing
# a mid type to forward variable to array's type tree
struct ArrayVariable{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    var::Variable{AT}

    ArrayVariable(var::Variable{AT}) where {T, N, AT <: AbstractArray{T, N}} = new{T, N, AT}(var)
end

Base.size(x::ArrayVariable) = size(value(x.var))
Base.getindex(x::ArrayVariable, i...) = getindex(value(x.var), i...)

Base.summary(io::IO, x::ArrayVariable) = summary(io, x.var)
Base.show(io::IO, m::MIME"text/plain", x::Variable{<:AbstractArray}) = show(io, m, ArrayVariable(x))
Base.show(io::IO, x::Variable) = print(io, value(x))
# broadcasting
# Broadcast.BroadcastStyle(::Type{T}) where {T <: ArrayVar} = Broadcast.BroadcastStyle(value_type(T))
# Broadcast.broadcastable(x::Variable) = Broadcast.broadcastable(x.value)
# Base.similar(bc::Broadcast.Broadcasted{DS}, ::Type{ElType}) where {DS, ElType} = Base.similar()
