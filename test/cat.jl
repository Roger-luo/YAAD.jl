using YAAD

const NA = Union{Number,AbstractArray}
const NAR = Union{Number,AbstractArray,AbstractNode}

Base.cat(X::NA...; dims)=Base._cat(dims, X...)
# Then define the method that catches at least one Value:
Base.cat(X::NAR...; dims)=register(Base.cat, X...; dims=dims)

function YAAD.gradient(::typeof(cat), grad, output, As...; dims)
end

cat([1, 2], Variable([2, 2]); dims=2)
@which Base.cat([1, 2], [1, 2]; dims=1)

@which Base.cat_t(Float64, [1, 2], [1, 2]; dims=1)
