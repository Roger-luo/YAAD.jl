export zero_grad!
"""
    zero_grad!(var)

clear gradient storage in the whole comput-graph.
"""
function zero_grad! end

function zero_grad!(x)
    for each in parameters(x)
        zero_grad!(each)
    end
    x
end

function zero_grad!(x::Union{CachedNode, Node})
    for each in args(x)
        zero_grad!(each)
    end

    for each in kwargs(x)
        zero_grad!(each)
    end

    zero_grad!(value(x))
    x
end

function zero_grad!(x::Variable)
    fill!(x.grad, zero(eltype(x.grad)))
    x
end

export register_parameters, parameters

"""
    register_parameters(x::OperatorType) -> iterator

Returns the iterator of all parameters in the instance `x` of `OperatorType`.
Note, here, `OperatorType` does not need to be subtype of [`Operator`](@ref).
"""
function register_parameters end

# TODO: move this to YAAD
register_parameters(op::T) where T = error("$T's parameters are not registered")
register_parameters(op::Function) = ()
register_parameters(op::Operator) = ()
# just an eye candy
parameters(op) = register_parameters(op)
