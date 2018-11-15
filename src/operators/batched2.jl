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
for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin
        function gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3}, beta::($elty), C::AbstractArray{$elty, 3})
            @assert !BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            BLAS.chkstride1(A)
            BLAS.chkstride1(B)
            BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((LinearAlgebra.BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                     Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt},
                     Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end
            C
        end
        function gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            gemm!(transA, transB, alpha, A, B, zero($elty), similar(B, $elty, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), size(B, 3))))
        end
        function gemm(transA::AbstractChar, transB::AbstractChar, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end

gemm(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T = gemm('N', 'N', A, B)

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

gemm(A::AbstractArray{T, 3}, B::Batched.Transpose) where T = gemm('N', 'T', A, B.parent)
gemm(A::Batched.Transpose, B::AbstractArray{T, 3}) where T = gemm('T', 'N', A.parent, B)
gemm(A::Batched.Transpose, B::Batched.Transpose) = gemm('T', 'T', A.parent, B.parent)

export ScaleMatrix

"""
    ScaleMatrix{B, K, T} <: AbstractArray{T, 3}

A batch of scalar multiplies a batch of identities, where batch size is
`B`, each identity's size is `K`.
"""
struct ScaleMatrix{B, K, T, VT <: AbstractVector{T}} <: AbstractArray{T, 3}
    scalars::VT
    ScaleMatrix{B, K}(scalars::VT) where {B, K, T, VT <: AbstractVector{T}} = new{B, K, T, VT}(scalars)
end

Base.size(x::ScaleMatrix{B, K, T}) where {B, K, T} = (K, K, B)
Base.@propagate_inbounds Base.getindex(m::ScaleMatrix{B, K, T}, i::Int, j::Int, k::Int) where {B, K, T} =
    i == j ? getindex(m.scalars, k) : zero(T)
Base.IndexStyle(::Type{<:ScaleMatrix}) = IndexCartesian()
Base.transpose(A::ScaleMatrix) = A

function gemm!(A::ScaleMatrix{NBatch, K, T}, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where {NBatch, K, T}
    @inbounds for k in 1:NBatch, j in 1:size(B, 2), i in 1:size(B, 1)
        C[i, j, k] += A.scalars[k] * B[i, j, k]
    end
    C
end

gemm!(A::AbstractArray{T, 3}, B::ScaleMatrix{NBatch, K, T}, C::AbstractArray{T, 3}) where {NBatch, K, T} = gemm!(B, A, C)
gemm!(A::Transpose{NBatch, T}, B::ScaleMatrix{NBatch, K, T}, C::AbstractArray{T, 3}) where {NBatch, K, T} = gemm!(B, A, C)

end # Batched


Batched.gemm(A::Value, B::Value) = register(Batched.gemm, A, B)

function gradient(::typeof(Batched.gemm), grad, output, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    Batched.gemm(grad, transpose(B)), Batched.gemm(transpose(A), grad)
end

Batched.tr(A::Value) = register(Batched.tr, A)

function gradient(::typeof(Batched.tr), grad, output, A::AbstractArray{T, 3}) where T
    (Batched.ScaleMatrix{size(A, 3), size(A, 1)}(grad), )
end
