export Batched

"""
Batched operation in Julia.

This module wraps some useful batched operation with a plain
for-loop on CPU. All the functions in this module are defined
with gradients in YAAD.
"""
module Batched
import LinearAlgebra
import LinearAlgebra: BLAS

# last dim is batch dim
function gemm!(tA::Char, tB::Char, alpha::T, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}, beta::T, C::AbstractArray{T, 3}) where T
    nbatch = size(C, 3)
    @inbounds for i = 1:nbatch
        BLAS.gemm!(tA, tB, alpha, view(A, :, :, i), view(B, :, :, i), beta, view(C, :, :, i))
    end
    C
end

gemm!(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where T =
    gemm!('N', 'N', one(T), A, B, one(T), C)

function gemm(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @boundscheck size(A, 3) == size(B, 3) || throw(DimensionMismatch("Batch dimension mismatch, got $(size(A, 3)) and $(size(B, 3))."))
    gemm!(A, B, zeros(T, (size(A, 1), size(B, 2), size(A, 3))))
end

function tr!(A::AbstractArray{T, 3}, out::AbstractArray{T, 1}) where T
    @inbounds for i in eachindex(out)
        out[i] = LinearAlgebra.tr(view(A, :, :, i))
    end
    out
end

tr(A::AbstractArray{T, 3}) where T = tr!(A, Vector{T}(undef, size(A, 3)))

################################################################################
##########################    Transpose Operations   ###########################

export Transpose

"""
    Transpose{B, T, AT <: AbstractArray{T, 3}} <: AbstractArray{T, 3}

Batched transpose. Transpose a batch of matrix.
"""
struct Transpose{B, T, AT <: AbstractArray{T, 3}} <: AbstractArray{T, 3}
    parent::AT

    Transpose(A::AT) where {T, AT <: AbstractArray{T, 3}} = new{size(A, 3), T, AT}(A)
end

Base.size(m::Transpose) = (size(m.parent, 2), size(m.parent, 1), size(m.parent, 3))
Base.axes(m::Transpose) = (axes(m.parent, 2), axes(m.parent, 1), axes(m.parent, 3))
Base.@propagate_inbounds Base.getindex(m::Transpose, i::Int, j::Int, k::Int) = getindex(m.parent, j, i, k)
Base.@propagate_inbounds Base.setindex!(m::Transpose, v, i::Int, j::Int, k::Int) = setindex!(m.parent, v, j, i, k)
Base.IndexStyle(::Type{<:Transpose}) = IndexCartesian()

# FIXME: implement a different generic function in case this becomes a pirate?
Base.transpose(A::AbstractArray{T, 3}) where T = Transpose(A)

gemm!(A::AbstractArray{T, 3}, B::Batched.Transpose, C::AbstractArray{T, 3}) where T =
    gemm!('N', 'T', one(T), A, B.parent, one(T), C)
gemm!(A::Batched.Transpose, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where T =
    gemm!('T', 'N', one(T), A.parent, B, one(T), C)


export ScalarIdentity

"""
    ScalarIdentity{B, K, T} <: AbstractArray{T, 3}

A batch of scalar multiplies a batch of identities, where batch size is
`B`, each identity's size is `K`.
"""
struct ScalarIdentity{B, K, T, VT <: AbstractVector{T}} <: AbstractArray{T, 3}
    scalars::VT
    ScalarIdentity{B, K}(scalars::VT) where {B, K, T, VT <: AbstractVector{T}} = new{B, K, T, VT}(scalars)
end

Base.size(x::ScalarIdentity{B, K, T}) where {B, K, T} = (K, K, B)
Base.@propagate_inbounds Base.getindex(m::ScalarIdentity{B, K, T}, i::Int, j::Int, k::Int) where {B, K, T} =
    i == j ? getindex(m.scalars, k) : zero(T)
Base.IndexStyle(::Type{<:ScalarIdentity}) = IndexCartesian()
Base.transpose(A::ScalarIdentity) = A

function gemm!(A::ScalarIdentity{NBatch, K, T}, B::Transpose{NBatch, T}, C::AbstractArray{T, 3}) where {NBatch, K, T}
    @inbounds for i in 1:NBatch
        C[:, :, i] .+= A.scalars[i] * view(B, :, :, i)
    end
    C
end

function gemm!(A::ScalarIdentity{NBatch, K, T}, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where {NBatch, K, T}
    @inbounds for i in 1:NBatch
        C[:, :, i] .+= A.scalars[i] * view(B, :, :, i)
    end
    C
end

gemm!(A::AbstractArray{T, 3}, B::ScalarIdentity{NBatch, K, T}, C::AbstractArray{T, 3}) where {NBatch, K, T} = gemm!(B, A, C)
gemm!(A::Transpose{NBatch, T}, B::ScalarIdentity{NBatch, K, T}, C::AbstractArray{T, 3}) where {NBatch, K, T} = gemm!(B, A, C)

end # Batched


Batched.gemm(A::AbstractNode, B::AbstractNode) = register(Batched.gemm, A, B)

function gradient(::typeof(Batched.gemm), grad, output, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    Batched.gemm(grad, transpose(B)), Batched.gemm(transpose(A), grad)
end

Batched.tr(A::AbstractNode) = register(Batched.tr, A)

function gradient(::typeof(Batched.tr), grad, output, A::AbstractArray{T, 3}) where T
    (Batched.ScalarIdentity{size(A, 3), size(A, 1)}(grad), )
end
