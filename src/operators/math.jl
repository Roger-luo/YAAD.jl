using DiffRules, SpecialFunctions, NaNMath

for (mod, name, nargs) in keys(DiffRules.DEFINED_DIFFRULES)
    f_ex_head = Expr(:., mod, QuoteNode(name))

    if nargs == 1
        df_ex = DiffRules.diffrule(mod, name, :x)

        @eval begin
            $(f_ex_head)(x::Value) = register($(f_ex_head), x)
            gradient(::typeof($(f_ex_head)), grad, output, x) = (grad * $df_ex, )
            gradient(mt::Trait.Broadcasted{typeof($f_ex_head)}, grad, output, x) = (@.(grad * $(df_ex)), )
        end
    elseif nargs == 2
        df_ex = DiffRules.diffrule(mod, name, :x, :y)

        @eval begin

            $(f_ex_head)(x1::Value, x2) = register($f_ex_head, x1, x2)
            $(f_ex_head)(x1, x2::Value) = register($f_ex_head, x1, x2)
            $(f_ex_head)(x1::Value, x2::Value) = register($f_ex_head, x1, x2)

            gradient(::typeof($f_ex_head), grad, output, x, y) =
                (grad * $(df_ex[1]), grad * $(df_ex[2]))
            gradient(::Trait.Broadcasted{typeof($f_ex_head)}, grad, output, x, y) =
                (@.(grad * ($(df_ex[1]))), @.(grad * $(df_ex[2])))
        end
    else
        @info "unknown operator $name"
    end
end
