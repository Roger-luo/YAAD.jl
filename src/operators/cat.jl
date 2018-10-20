const NA = Union{Number,AbstractArray}
const NAR = Union{Number,AbstractArray,AbstractNode}

cat(X::NA...; dims)=Base._cat(dims, X...)
# Then define the method that catches at least one Value:
cat(X::NAR...; dims)=register(cat, X...; dims=dims)
