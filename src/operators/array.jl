Base.getindex(x::Value, inds...) = register(Base.getindex, x, inds...)

function gradient(::typeof(Base.getindex), grad, output, x::AbstractArray, inds...)
    grad_output = fill!(similar(x), 0)
    setindex!(grad_output, grad, inds...)
    (grad_output, )
end

gradient(::typeof(Base.getindex), grad, output, x::Number, ind::Int) = (grad, )

Base.selectdim(x::Value, d, i) = register(Base.selectdim, x, d, i)

function gradient(::typeof(Base.selectdim), grad, output, x::AbstractArray, d, i)
    grad_output = fill!(similar(x), 0)
    subgrad = Base.selectdim(grad_output, d, i)
    setindex!(subgrad, grad, :)
    (grad_output, )
end

Base.view(x::Value, inds...) = register(Base.view, x, inds...)

function gradient(::typeof(Base.view), grad, output, x::AbstractArray, inds...)
    grad_output = fill!(similar(x), 0)
    subgrad = view(grad_output, inds...)
    setindex!(subgrad, grad, :)
    (grad_output, )
end

Base.reshape(x::Value, dims...) = register(Base.reshape, x, dims...)
function gradient(::typeof(Base.reshape), grad, output, x, dims...)
    (reshape(grad, size(x)...), )
end
