using DiffRules, SpecialFunctions, NaNMath

for (mod, name, nargs) in keys(DiffRules.DEFINED_DIFFRULES)
    f_ex_head = Expr(:., mod, QuoteNode(name))

    if nargs == 1
        df_ex = DiffRules.diffrule(mod, name, :x)

        @eval begin
            $(f_ex_head)(x::AbstractNode) = CachedNode($(f_ex_head), (x, ))
            gradient(::typeof($(f_ex_head)), grad, output, x) = (grad * $df_ex, )
            gradient(mt::BroadcastedTrait{typeof($f_ex_head)}, grad, output, x) = (@.(grad * $(df_ex)), )
        end
    elseif nargs == 2
        df_ex = DiffRules.diffrule(mod, name, :x, :y)

        @eval begin

            $(f_ex_head)(x1::AbstractNode, x2) = CachedNode($f_ex_head, (x1, x2))
            $(f_ex_head)(x1, x2::AbstractNode) = CachedNode($f_ex_head, (x1, x2))
            $(f_ex_head)(x1::AbstractNode, x2::AbstractNode) = CachedNode($f_ex_head, (x1, x2))

            gradient(::typeof($f_ex_head), grad, output, x, y) =
                (grad * $(df_ex[1]), grad * $(df_ex[2]))

            gradient(::BroadcastedTrait{typeof($f_ex_head)}, grad, output, x, y) =
                (@.(grad * ($(df_ex[1]))), @.(grad * $(df_ex[2])))
        end
    else
        print(mod, name)
    end
end
