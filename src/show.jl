Base.summary(io::IO, x::CachedNode) = (print(io, "(tracked) "); summary(io, x.output))
Base.show(io::IO, m::MIME"text/plain", x::CachedNode{<:Node, <:AbstractArray}) = (print(io, "(tracked) "); show(io, m, x.output))
# Base.show(io::IO, x::CachedNode) = show(io, x.output)
