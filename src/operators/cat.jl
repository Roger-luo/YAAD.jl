const NA = Union{Number,AbstractArray}
const NAR = Union{Number,AbstractArray,Value}

Base.cat(X::NA...; dims) = Base._cat(dims, X...)
# Then define the method that catches at least one Value:
Base.cat(X::NAR...; dims) = register(Base.cat, X...; dims=dims)

Base.vcat(A::Value) = cat(A; dims=Val(1))
Base.vcat(A::Value, B::AbstractArray) = cat(A, B; dims=Val(1))
Base.vcat(A::AbstractArray, B::Value) = cat(A, B; dims=Val(1))
Base.vcat(A::NAR...) = cat(A...; dims=Val(1))
Base.hcat(A::Value) = cat(A; dims=Val(2))
Base.hcat(A::Value, B::AbstractArray) = cat(A, B; dims=Val(2))
Base.hcat(A::AbstractArray, B::Value) = cat(A, B; dims=Val(2))
Base.hcat(A::NAR...) = cat(A...; dims=Val(2))

gradient(::typeof(Base.cat), grad, output, As...; dims) = uncat(dims, grad, As...)

"""
    uncat(dims, cat_output, xs...) -> Vector{SubArray}

The reverse operation of [`Base.cat`],
it will return corresponding [`Base.view`] of the inputs of a `cat`.
"""
uncat(dims, cat_output, xs...) = _uncat(Base.dims2cat(dims), cat_output, xs...)

function _uncat(catdims, cat_output, xs...)
    N = ndims(cat_output)
    shape = size(cat_output)

    offsets = zeros(Int, N)
    inds = Vector{UnitRange{Int}}(undef, N)
    concat = copyto!(zeros(Bool, N), catdims)

    rs = Vector{AbstractArray}(undef, length(xs))
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
        rs[k] = reshape(view(cat_output, I...), size(x))
    end
    return rs
end
