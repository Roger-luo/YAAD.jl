export AbstractNode, LeafNode
export Variable, Node, CachedNode, forward, gradient, backward, value, args, arg, operator
export register

# export register!

"""
    Operator

Abstract type for operators in the computation graph.
"""
abstract type Operator end

"""
    Trait

This module contains function traits as a subtype of [`Operator`](@ref).
"""
module Trait
import YAAD: Operator

"""
    Method{FT} <: Operator

This trait wraps a callable object in Julia (usually a `Function`).
"""
struct Method{FT} <: Operator
    f::FT
end

"""
    Broadcasted{FT} <: Operator

This trait wraps a callable object that being broadcasted. It will help to
dispatch different gradient methods overloaded for broadcasted operation
comparing to [`Method`](@ref).
"""
struct Broadcasted{FT} <: Operator
    f::FT
end
end # Trait

"""
    AbstractNode

Abstract type for nodes in computation graph.
"""
abstract type AbstractNode end

"""
    LeafNode <: AbstractNode

Abstract type for leaf nodes in a computation graph.
"""
abstract type LeafNode <: AbstractNode end

"""
    Variable{T} <: LeafNode

A kind of leaf node. A general type for variables in a comput-graph.
Similar to PyTorch's Variable, gradient will be accumulated to `var.grad`.
"""
mutable struct Variable{T} <: LeafNode
    value::T
    grad::T

    Variable(val::T) where T = new{T}(val)
    Variable(val::T, grad::T) where T = new{T}(val)
end

"""
    Node{FT, ArgsT} <: AbstractNode

General node in a comput-graph. It stores a callable operator `f` of type `FT`
and its arguments `args` in type `ArgsT` which should be a tuple.
"""
struct Node{FT <: Operator, ArgsT <: Tuple, KwargsT <: NamedTuple} <: AbstractNode
    f::FT
    args::ArgsT
    kwargs::KwargsT
end

# wrap function to Method
Node(f::Function, args, kwargs) = Node(Trait.Method(f), args, kwargs)
Node(op, args) = Node(op, args, NamedTuple())

"""
    CachedNode{NT, OutT} <: AbstractNode

Stores the cache of output with type `OutT` from a node of
type `NT` in comput-graph. CachedNode is mutable, its output
can be updated by [`forward`](@ref).
"""
mutable struct CachedNode{NT <: AbstractNode, OutT} <: AbstractNode
    node::NT
    output::OutT
end

function CachedNode(f, args...; kwargs...)
    node = Node(f, args, kwargs.data)
    output = forward(node)
    CachedNode(node, output)
end

Base.size(x::AbstractNode) = size(value(x))
Base.size(x::AbstractNode, d::Int) = size(value(x), d)
Base.similar(x::AbstractNode) = Variable(similar(value(x)))
Base.similar(x::AbstractNode, dims::Dims) = Variable(similar(value(x), dims))
Base.similar(x::AbstractNode, element_type::Type{S}, dims::Dims) where S = Variable(similar(value(x), element_type, dims))
Base.axes(x::AbstractNode) = axes(value(x))

"""
    arg(node, i) -> ArgumentType

Returns the `i`-th argument of the call in `node`.
"""
function arg end

"""
    args(node) -> Tuple

Returns the arguments of the call in `node`.
"""
function args end

"""
    kwargs(node) -> NamedTuple

Returns the keyword arguements of the call in `node`.
"""
function kwargs end

"""
    operator(node) -> YAAD.Operator

Returns the operator called in this node.
"""
function operator end

arg(x::Node, i::Int) = x.args[i]
args(x::Node) = x.args
kwargs(x::Node) = x.kwargs
operator(x::Node) = x.f

arg(x::CachedNode, i::Int) = x.node.args[i]
args(x::CachedNode) = x.node.args
kwargs(x::CachedNode) = x.node.kwargs
operator(x::CachedNode) = x.node.f

Base.eltype(x::AbstractNode) = eltype(value(x))

"""
    value(node)

Returns the value when forwarding at current node. `value` is different
than [`forward`](@ref) method, `value` only returns what the node contains,
it will throw an error, if this node does not contain anything.
"""
function value end

# forward other values
value(x) = x

function value(x::T) where {T <: AbstractNode}
    error(
        "Expected value in this node $x of type $T ",
        "check if you defined a non-cached node",
        " or overload value function for your node."
    )
end

value(x::Variable) = x.value
value(x::CachedNode) = value(x.output)

"""
    forward(node) -> output

Forward evaluation of the comput-graph. This method will call the operator
in the comput-graph and update the cache.

    forward(f, args...) -> output

For function calls.
"""
function forward end

forward(x) = x
forward(x::NT) where {NT <: AbstractNode} = error("forward method is not implemented for node type: $NT")
forward(x::Colon) = x
forward(node::LeafNode) = value(node)
forward(node::Node) = forward(node.f, map(forward, node.args)...; map(forward, node.kwargs)...)
forward(node::CachedNode) = (node.output = forward(node.node))
forward(op::Operator, args...; kwargs...) = op.f(args...; kwargs...)
forward(op::Trait.Broadcasted, args...) = Broadcast.broadcasted(op.f, args...)

# better error msg
# NOTE: Colon is a Function, this can have ambiguity
# we force to use a trait here.
forward(::Function, args...) =
    error(
        "please wrap your operator as subtype of Operator",
        " directly forward a function may cause ambiguity"
    )

"""
    backward(node) -> nothing

Backward evaluation of the comput-graph.
"""
function backward end

# return nothing for non-node types
backward(x, grad) = nothing
backward(x::AbstractNode) = backward(x::AbstractNode, one(eltype(x)))

function backward(x::Variable, grad)
    if isdefined(x, :grad)
        x.grad += grad
    else
        x.grad = grad
    end
    nothing
end

function backward(x::Variable, grad::SubArray)
    if isdefined(x, :grad)
        x.grad += grad
    else
        x.grad = copyto!(similar(x.value), grad)
    end
    nothing
end

backward(node::CachedNode, grad) = backward(node, node.node.f, grad)
backward(node::CachedNode, op::Operator, grad) = backward(node, op.f, grad)

function backward(node::CachedNode, f, grad)
    backward_type_assert(node, grad)
    # TODO: replace with @assert when there is a compiler option for it
    @boundscheck backward_size_assert(node, grad)

    grad_inputs = gradient(node, grad)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end

"""
    backward_type_assert(node, grad)

throw more readable error msg for backward type check.
"""
function backward_type_assert end

# mute the compiler error msg
backward_type_assert(args...) = true

backward_type_assert(node::CachedNode{<:AbstractNode, T}, grad::T) where T = true
# exclude arrays
backward_type_assert(node::CachedNode{<:AbstractNode, T1}, grad::T2) where
    {T, N, T1 <: AbstractArray{T, N}, T2 <: AbstractArray{T, N}} = true
backward_type_assert(node::CachedNode{<:AbstractNode, T1}, grad::T2) where {T1, T2} =
    error("Gradient is expected to have the same",
          " type with outputs, expected $T1",
          " got $T2")

function backward_size_assert(node::CachedNode, grad)
    size(node.output) == size(grad) ||
        error(
            "gradient should have the same size with output,",
            " expect size $(size(node.output)), got $(size(grad))"
        )
end

"""
    gradient(node, grad)

Returns the gradient.
"""
function gradient end

## CachedNode
# 1. general interface
gradient(x::CachedNode, grad) = gradient(x.node.f, grad, x.output, map(value, x.node.args)...; map(value, x.node.kwargs)...)

# NOTE: operators help to define different grads when the fn is the same
# e.g Broadcasted{typeof(sin)} and `sin`

# 2. forward operator to function type
# this simplifies some operator's definition
gradient(x::Operator, grad, output, args...; kwargs...) =
    gradient(x.f, grad, output, args...; kwargs...)

gradient(fn, grad, output, args...; kwargs...) =
    error(
        "gradient of operator $fn is not defined\n",
        "Possible Fix:\n",
        "define one of the following:\n",
        "1. gradient(::typeof($fn), grad, output, args...; kwargs...)\n",
        "2. gradient(op::Trait.Method{typeof($fn)}, grad, output, args...; kwargs...)\n",
        "3. gradient(op::Trait.Broadcasted{typeof($fn)}, grad, output, args...; kwargs...)\n"
    )

"""
    register(f, args...; kwargs...)

This is just a alias for constructing a `CachedNode`. But notice this function
is used for register a node in `tape` in the global tape version implementation:

https://github.com/Roger-luo/YAAD.jl/tree/tape
"""
register(f, args...; kwargs...) = CachedNode(f, args...; kwargs...)
