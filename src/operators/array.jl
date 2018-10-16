Base.getindex(x::AbstractNode, inds...) = CachedNode(Base.getindex, (x, inds...))

function gradient(::typeof(Base.getindex), grad, output, x::AbstractArray, inds...)
    grad_output = fill!(similar(x), 0)
    setindex!(grad_output, grad, inds...)
    (grad_output, )
end

gradient(::typeof(Base.getindex), grad, output, x::Number, ind::Int) = (grad, )

Base.selectdim(x::ArrayNode, d, i) = CachedNode(Base.selectdim, (x, d, i))

function gradient(::typeof(Base.selectdim), grad, output, x::AbstractArray, d, i)
    grad_output = fill!(similar(x), 0)
    subgrad = Base.selectdim(grad_output, d, i)
    setindex!(subgrad, grad, :)
    (grad_output, )
end

Base.view(x::ArrayNode, inds...) = CachedNode(Base.view, (x, inds...))

function gradient(::typeof(Base.view), grad, output, x::AbstractArray, inds...)
    grad_output = fill!(similar(x), 0)
    subgrad = view(grad_output, inds...)
    setindex!(subgrad, grad, :)
    (grad_output, )
end

# Base.setindex!(x::AbstractArray, v::ArrayNode, inds...) = CachedNode(Base.setindex!, (x, v, inds...))
#
# function gradient(::typeof(Base.setindex!), grad, output, x::AbstractArray, v::AbstractArray, inds...)
#     getindex(grad, inds...)
# end
