using LinearAlgebra

LinearAlgebra.tr(x::AbstractNode) = register(LinearAlgebra.tr, x)
gradient(::typeof(tr), grad, output, x) = (grad * Matrix(I, size(x)), )

gradient(::typeof(transpose), grad, output, x::AbstractMatrix) = (transpose(grad), )
function gradient(::typeof(*), grad, output, lhs::AbstractVecOrMat, rhs::AbstractVecOrMat)
    grad * transpose(rhs), transpose(lhs) * grad
end
