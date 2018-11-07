using LinearAlgebra, LuxurySparse

LinearAlgebra.tr(x::Value) = register(LinearAlgebra.tr, x)
gradient(::typeof(tr), grad, output, x) = (grad * IMatrix{size(x, 1), eltype(grad)}(), )

gradient(::typeof(transpose), grad, output, x::AbstractMatrix) = (transpose(grad), )
function gradient(::typeof(*), grad, output, lhs::AbstractVecOrMat, rhs::AbstractVecOrMat)
    grad * transpose(rhs), transpose(lhs) * grad
end
