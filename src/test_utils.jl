module TestUtils

export gradcheck

using YAAD
import YAAD: Value

make_jacobian(input::AbstractArray, num_out::Int) = (zeros(eltype(input), length(input), num_out), )
make_jacobian(input::Variable, num_out::Int) = make_jacobian(value(input), num_out)
make_jacobian(inputs::Tuple, num_out::Int) = Tuple(first(make_jacobian(each, num_out)) for each in inputs)

zero_like(x::T) where {T <: Number} = zero(T)
zero_like(x) = fill!(similar(x), 0)
zero_like(x::Broadcast.Broadcasted) = zero_like(Broadcast.materialize(x))
zero_like(x::Value) = zero_like(value(x))

function get_numerical_jacobian(f, inputs...; target=inputs, eps=1e-3)
    output_size = length(value(f(inputs...)))
    jacobian = make_jacobian(target, output_size)

    x_tensors = [value(t) for t in target]
    for (x_tensor, d_tensor) in zip(x_tensors, jacobian)
        for (d_idx, x_idx) in enumerate(eachindex(x_tensor))
            orig = x_tensor[x_idx]
            x_tensor[x_idx] = orig - eps
            outa = copy(value(f(inputs...)))
            x_tensor[x_idx] = orig + eps
            outb = copy(value(f(inputs...)))
            x_tensor[x_idx] = orig

            r = (outb - outa) / (2 * eps)
            _set_numerical_jacobian_elem!(d_tensor, d_idx, r)
        end
    end
    return jacobian
end

_set_numerical_jacobian_elem!(d_tensor, d_idx, r::AbstractArray) = d_tensor[d_idx, :] = reshape(r, :)
_set_numerical_jacobian_elem!(d_tensor, d_idx, r::Number) = d_tensor[d_idx, 1] = r

function get_analytical_jacobian(inputs, output::Value)
    jacobian = make_jacobian(inputs, length(value(output)))
    jacobian_reentrant = make_jacobian(inputs, length(value(output)))
    grad_output = zero_like(output)
    reentrant = true
    correct_grad_sizes = true

    _analytical_jacobian!(jacobian, jacobian_reentrant, grad_output, output)

    for (jacobian_x, jacobian_reentrant_x) in zip(jacobian, jacobian_reentrant)
        length(jacobian_x) != 0 && maximum(abs.(jacobian_x - jacobian_reentrant_x)) != 0 && (reentrant = false)
    end

    jacobian, reentrant, correct_grad_sizes
end

function _analytical_jacobian!(jacobian::Tuple, jacobian_reentrant::Tuple, grad_output::AbstractArray, output::Value)
    for i in 1:length(grad_output)
        grad_output = fill!(grad_output, 0)
        grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant)
            grads_input = gradient(output, grad_output)
            for (jacobian_x, d_x) in zip(jacobian_c, grads_input)
                jacobian_x[:, i] = reshape(d_x, :)
                # TODO: check gradient size
            end
        end
    end
    jacobian, jacobian_reentrant
end

function _analytical_jacobian!(jacobian::Tuple, jacobian_reentrant::Tuple, grad_output::Number, output::Value)
    grad_output = 1
    for jacobian_c in (jacobian, jacobian_reentrant)
        grads_input = gradient(output, grad_output)
        for (jacobian_x, d_x) in zip(jacobian_c, grads_input)
            jacobian_x[:, 1] = reshape(d_x, :)
            # TODO: check gradient size
        end
    end

    jacobian, jacobian_reentrant
end


_differentiable_outputs(x) = Tuple(o for o in _as_tuple(x) if o isa Value)
_as_tuple(x) = (x, )
_as_tuple(x::Tuple) = x

function gradcheck(f, inputs...; eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=true)
    any_input_requiring_grad = false
    for inp in inputs
        if inp isa Variable
            any_input_requiring_grad = true
        end
    end

    any_input_requiring_grad || error("gradcheck expects at least one input to require gradient")

    output = _differentiable_outputs(f(inputs...))

    # replace this with assert in the future
    function fail_test(msgs...)
        raise_exception && error(msgs...)
        false
    end

    for (i, o) in enumerate(output)
        fn(inputs...) = _as_tuple(f(inputs...))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(inputs, o)
        numerical = get_numerical_jacobian(fn, inputs...; eps=eps)

        correct_grad_sizes || fail_test("Analytical gradient has incorrect size")

        for (j, (a, n)) in enumerate(zip(analytical, numerical))
            if length(a) != 0 || length(n) != 0
                isapprox(a, n; rtol=rtol, atol=atol) ||
                    fail_test("Jacobian mismatch for output $(i) with respect to input $j, \n",
                              "numerical: $n \nanalytical: $a\n")
            end
        end

        # TODO: reentrant is not actually checked
        reentrant || fail_test("Backward is not reentrant, i.e., running backward with same ",
                               "input and grad_output multiple times gives different values, ",
                               "although analytical gradient matches numerical gradient")
    end

    true
end

end # module TestUtils
