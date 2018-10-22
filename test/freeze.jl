graph = @freeze x, y begin
    tr(x * y)
end

Node(*, x, y)
struct PlaceHolder end

z1 = Node(*, Variable(PlaceHolder()), Variable(PlaceHolder()))
z2 = Node(tr, z1)

struct StaticNode{FT, ArgsT, KwargsT, OutT}
    f::FT
    args::ArgT
    kwargs::KwargsT

    output::OutT
end
