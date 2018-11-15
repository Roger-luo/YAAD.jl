module Batched

import LinearAlgebra: BLAS

"""
    tr(::AbstractArray{T, 3}) where T

Batched matrix trace.
"""
tr(A::AbstractArray{T, 3}) where T = tr!(A, similar(A, (size(A, 3), )))

function tr!(A::AbstractArray{T, 3}, B::AbstractVector{T}) where T
    @assert size(A, 1) == size(A, 2) "Expect a square matrix" # checksquare
    @boundscheck size(A, 3) == size(B, 1) || error("Batch size mismatch")

    nbatch = size(A, 3)
    n = size(A, 1)
    @inbounds for k in 1:nbatch
        for i in 1:n
            B[k] += A[i, i, k]
        end
    end
    B
end

function gemm! end
function gemm end

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
                ccall((BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
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


####################

const _BATCHED_MATRIX_LIST = [
        (:BatchedMatrix, 'N'),
        (:(BatchedTranspose{T, N, <:BatchedMatrix} where N), 'T'),
        (:(BatchedAdjoint{T, N, <:BatchedMatrix} where N), 'C')
]

for (TA, transA) in _BATCHED_MATRIX_LIST, (TB, transB) in _BATCHED_MATRIX_LIST
    @eval function batched_mul!(C::BatchedMatrix{T}, A::$TA, B::$TB) where T
        @boundscheck check_batch_dim_size(A, B, C)
        batchA, batchB, batchC = merge_batch_dim(A), merge_batch_dim(B), merge_batch_dim(C)
        batched_gemm!($transA, $transB, one(T), batchA, batchB, one(T), batchC)
        C
    end
end


function batched_mul!(C::AbstractArray{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
end


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
batched_transpose(m::AbstractArray{T, 3}) where T = Transpose(m)

gemm!(A::AbstractArray{T, 3}, B::Batched.Transpose, C::AbstractArray{T, 3}) where T =
    gemm!('N', 'T', one(T), A, B.parent, one(T), C)
gemm!(A::Batched.Transpose, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where T =
    gemm!('T', 'N', one(T), A.parent, B, one(T), C)


struct BatchedScale{B, K, T, VT <: AbstractVector{T}} <: AbstractArray{T, 3}
    scalar::VT
    BatchedScale{B, K}(scalars::VT) where {B, K, T, VT <: AbstractVector{T}} = new{B, K, T, VT}(scalars)
end

Base.size(x::BatchedScale{B, K, T}) where {B, K, T} = (K, K, B)
Base.@propagate_inbounds Base.getindex(m::BatchedScale{B, K, T}, i::Int, j::Int, k::Int) where {B, K, T} =
    i == j ? getindex(m.scalars, k) : zero(T)
Base.IndexStyle(::Type{<:BatchedScale}) = IndexCartesian()
Base.transpose(A::BatchedScale) = A


end
