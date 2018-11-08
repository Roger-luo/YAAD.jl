backward_type_assert(node::CachedNode{<:AbstractNode, T1}, grad::T2) where {T1 <: BatchedUniformScaling, T2} = true
backward_type_assert(node::CachedNode{<:AbstractNode, T1}, grad::T2) where {T1, T2 <: BatchedUniformScaling} = true

function gradient(::typeof(LinearAlgebra.tr), grad::BatchedScalar, output, A::AbstractBatchedMatrix{T}) where T
    (Batched.BatchedUniformScaling(grad), )
end
