const NA = Union{Number,AbstractArray}
const NAR = Union{Number,AbstractArray,AbstractNode}

Base.cat(X::NA...; dims) = Base._cat(dims, X...)
# Then define the method that catches at least one Value:
Base.cat(X::NAR...; dims) = register(Base.cat, X...; dims=dims)

gradient(::typeof(Base.cat), grad, output, As...; dims) = uncat(dims, grad, As...)

uncat(dims, cat_output, xs...) = _uncat(Base.dims2cat(dims), cat_output, xs...)

function _uncat(catdims, cat_output, xs...)
    N = ndims(cat_output)
    shape = size(cat_output)

    offsets = zeros(Int, N)
    inds = Vector{UnitRange{Int}}(undef, N)
    concat = copyto!(zeros(Bool, N), catdims)

    rs = Vector{SubArray}(undef, length(xs))
    @inbounds for (k, x) in enumerate(xs)
        for i in 1:N
            if concat[i]
                inds[i] = offsets[i] .+ Base.cat_indices(x, i)
                offsets[i] += Base.cat_size(x, i)
            else
                inds[i] = 1:shape[i]
            end
        end
        I::NTuple{N, UnitRange{Int}} = (inds...,)
        rs[k] = view(cat_output, I...)
    end
    return rs
end
