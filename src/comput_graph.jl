export Node, CachedNode, forward, gradient, backward

const GLOBAL_TAPE = IdDict{Any, AbstractNode}()

function forward end
function gradient end
function backward end

forward(x) = x
forward(x::Colon) = x # NOTE: Colon is a Function
gradient(x, grad) = nothing
forward(x::Variable) = value(x)
forward(f::Union{Type, Function}, args...) = f(map(forward, args)...)

backward(x::AbstractNode) = backward(x::AbstractNode, one(eltype(x)))
backward(x, grad) = nothing

function backward(x::Variable, grad)
    if isdefined(x, :grad)
        x.grad += grad
    else
        x.grad = grad
    end
    nothing
end

# NOTE: we don't allow keyword functions here
mutable struct Node{FT, ArgsT} <: AbstractNode
    f::FT
    args::ArgsT
    # kwargs::KwargsT
end

# NOTE: we forward to forward method for the operator
# if no forward method is defined for this operator
# then we directly use it as a function (or a callable type)
forward(x::Node) = forward(x.f, x.args...)

mutable struct CachedNode{NT <: Node, OutT} <: AbstractNode
    node::NT
    output::OutT

    CachedNode(x::NT, output::T) where {NT <: Node, T} = new{NT, T}(x, output)
end

value(x::CachedNode) = x.output
args(x::CachedNode, i::Int) = value(x.node.args[i])
Base.eltype(x::CachedNode) = eltype(x.output)

# init without output
function CachedNode(f, args)
    node = Node(f, args)
    output = forward(node)
    CachedNode(node, output)
end

# init with output
CachedNode(f, args, output) = CachedNode(Node(f, args), output)

# make_node(args...) = CachedNode(args...)

# function make_node(args...)
#     node = CachedNode(args...)
#     GLOBAL_TAPE[node.output] = node
#     node
# end

backward(x::CachedNode, grad) = backward(x, x.node, grad)

function backward(x::CachedNode, node, grad)
    backward_assert(x, grad)
    output_grad = gradient(x, grad)
    _backward(x, output_grad)
end

backward_assert(args...) = true
# backward_assert(x::CachedNode{NT, T}, grad::GT) where {NT, T, GT} =
#     error("gradient should have the same type with output, expect $T, got $GT")
backward_assert(x::CachedNode{NT, T}, grad::T) where {NT, T <: AbstractArray} =
    size(x.output) == size(grad) || error("gradient should have same size with output, expect $(size(x.output)), got $(size(grad))")

_backward(x::CachedNode, output_grad::T) where T = error("output of gradient($x, ...) should be a tuple, got $(T)")

function _backward(x::CachedNode, output_grad::Tuple)
    for (each, each_grad) in zip(x.node.args, output_grad)
        backward(each, each_grad)
    end
    nothing
end

# forward cached node x -> x.node
forward(x::CachedNode) = (x.output = forward(x.node))

# 1. use expanded version if there is no gradient function for this node
gradient(x::CachedNode, grad) = gradient(x.node.f, grad, x.output, map(value, x.node.args)...)

# TODO: check if this is possible
# 2. maybe we don't need the output
# gradient(f, grad, output, args...) = gradient(f, grad, args...)

# 3. maybe we just need the grad, e.g +
# gradient(f, grad, args...) = gradient(f, grad)
