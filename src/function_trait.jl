export FunctionTrait, MethodTrait

abstract type FunctionTrait{FT, ArgsType} end

arg_ts(::FunctionTrait{FT, Tuple{}}) where FT = ()
arg_ts(::FunctionTrait{FT, ArgsType <: Tuple}) where {FT, ArgsType} =
    (Base.tuple_type_head(ArgsType), arg_ts(Base.tuple_type_tail(ArgsType))...)

struct MethodTrait{FT, ArgsType} <: FunctionTrait{FT, ArgsType}
    f::FT
end

MethodTrait(f::Function, nargs::Int) = MethodTrait{typeof(f), NTuple{nargs, Any}}(f)
MethodTrait(f::Function, args_t::Type{<:Tuple}) = MethodTrait{typeof(f), args_t}(f)
MethodTrait(f::Function, arg_ts::Type...) = MethodTrait{typeof(f), Tuple{arg_ts...}}(f)

# callable types
MethodTrait(f::Type{T}, arg_ts::Type...) where T = MethodTrait{T, Tuple{arg_ts...}}(f)
MethodTrait(f::Type{T}, nargs::Int) where T = MethodTrait{T, NTuple{nargs, Any}}(f)

(mt::MethodTrait)(args...) = mt.f(args...)

Base.show(io::IO, trait::MethodTrait{FT, NTuple{N, Any}, Tuple{}}) where {FT, N} =
    print(io, trait.f, "(", join(["::Any" for i=1:N], ", "),")")
