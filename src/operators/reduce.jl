Base.length(x::Value) = length(value(x))
Base.eltype(x::Value) = eltype(value(x))

# NOTE: this is a fallback method, it is slow
Base.iterate(x::Value) = iterate_forward(iterate(value(x)), x)
Base.iterate(x::Value, st) = iterate_forward(iterate(value(x), st), x, st)

iterate_forward(out::Nothing, x::Value) = nothing
iterate_forward(out::Nothing, x::Value, st) = nothing

function iterate_forward(out, x::Value, st)
    node = Node(Base.iterate, (x, st))
    v, st = out
    CachedNode(node, v), st
end

function iterate_forward(out, x::Value)
    node = Node(Base.iterate, (x, ))
    v, st = out
    CachedNode(node, v), st
end

function gradient(::typeof(Base.iterate), grad::Number, output, x::AbstractArray)
    out_grad = zero(x)
    out_grad[1] = grad
    (out_grad, )
end

function gradient(::typeof(Base.iterate), grad, output, x::AbstractArray, st)
    out_grad = zero(x)
    out_grad[st] = grad
    (out_grad, )
end

Base.sum(x::Value{<:AbstractArray}; dims=:) = register(Base.sum, x; dims=dims)

gradient(::typeof(Base.sum), grad, output, x::AbstractArray; dims) =
    grad_sum(grad, x, dims)

grad_sum(grad, x, dims::Colon) = (fill!(similar(x), grad), )
