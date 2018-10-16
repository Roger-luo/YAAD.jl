struct BroadcastedTrait{FT}
    f::FT
end

(mt::BroadcastedTrait)(args...) = Broadcast.broadcasted(mt.f, args...)

struct ComputGraphStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:AbstractNode}) = ComputGraphStyle()
Broadcast.BroadcastStyle(s::ComputGraphStyle, x::Broadcast.BroadcastStyle) = s

# # this enables method traits broadcast as a constant
# Broadcast.BroadcastStyle(::Type{<:MethodTrait}) = Base.Broadcast.DefaultArrayStyle{0}()
# Base.size(x::MethodTrait) = ()
# Base.getindex(x::MethodTrait, i...) = x
# Broadcast.broadcastable(x::MethodTrait) = x

Broadcast.broadcastable(x::AbstractNode) = x

function Broadcast.broadcasted(::ComputGraphStyle, f, args...)
    mt = BroadcastedTrait(f)
    CachedNode(mt, args)
end

Broadcast.materialize(x::AbstractNode) = CachedNode(Broadcast.materialize, (x, ))
# Broadcast.materialize!(dest, x::AbstractNode) = CachedNode(MethodTrait(Broadcast.materialize!, 2), (dest, x))

gradient(::typeof(Broadcast.materialize), grad, output, x) = (grad, )
# gradient(x::CachedNode, bt::MethodTrait{typeof(Broadcast.materialize), Tuple{Any}}, grad) = (grad, )
# gradient(x::CachedNode, bt::MethodTrait{typeof(Broadcast.materialize!), NTuple{2, Any}}, grad) = (grad, grad)

# function gradient(x::CachedNode, bt::BroadcastedTrait, grad::AbstractArray)
#     gradient.(MethodTrait(bt.f, 1), x.output, grad, map(value, x.node.args)...)
# end

# arraymath.jl
for sym in (:(/), :(\), :*, :+, :-)
    f = Expr(:., :Base, QuoteNode(sym))

    if f != :/
        @eval ($f)(A::Number, B::Variable{<:AbstractArray}) = broadcast($f, A, B)
        @eval ($f)(A::Number, B::CachedNode{<:Node, <:AbstractArray}) = broadcast($f, A, B)
    end
    if f != :\
        @eval ($f)(A::Variable{<:AbstractArray}, B::Number) = broadcast($f, A, B)
        @eval ($f)(A::CachedNode{<:Node, <:AbstractArray}, B::Number) = broadcast($f, A, B)
    end
end

for sym in (:-, :conj, :real, :imag)
    f = Expr(:., :Base, QuoteNode(sym))
    @eval ($f)(A::ArrayNode) = broadcast($f, A)
end
