using Luxor

demo = Movie(1000, 1000, "comput-graph")

backdrop(scene, framenumber) = background("none")

function frame(scene, framenumber)
    radius = 40
    node_color = "#6ca1f7"

    sethue("black")
    p1 = Point(200, -200)
    p2 = Point(-200, 200)
    temps = Point[between(p1, p2, i) for i in LinRange(0, 1, 4)]
    height = abs(temps[3].y - temps[4].y)
    width = abs(temps[3].x - temps[4].x)
    p_A = Point(temps[2].x, temps[4].y)
    p_x = Point(temps[3].x, temps[4].y + height)
    p_b = Point(temps[1].x, temps[4].y + height)
    p_c = Point(p_b.x + 2 * width, p_b.y)
    p_z = Point(temps[2].x + 2 * width, temps[4].y)


    arrow_configs = (arrowheadlength=13, arrowheadangle=pi/8, linewidth=1.5)
    for (prev_p, next_p) in zip(temps[1:end-1], temps[2:end])
        _, a_p1, _ = intersectionlinecircle(prev_p, next_p, prev_p, radius)
        _, _, a_p2 = intersectionlinecircle(prev_p, next_p, next_p, radius)
        arrow(a_p2, a_p1; arrow_configs...)
    end

    _, _, a_p1 = intersectionlinecircle(p_A, temps[3], temps[3], radius)
    _, a_p2, _ = intersectionlinecircle(p_A, temps[3], p_A, radius)
    arrow(a_p2, a_p1; arrow_configs...)

    _, _, a_p1 = intersectionlinecircle(p_x, temps[4], temps[4], radius)
    _, a_p2, _ = intersectionlinecircle(p_x, temps[4], p_x, radius)
    arrow(a_p2, a_p1; arrow_configs...)

    _, _, a_p1 = intersectionlinecircle(p_x, temps[2], temps[2], radius)
    _, a_p2, _ = intersectionlinecircle(p_x, temps[2], p_x, radius)
    arrow(a_p2, a_p1; arrow_configs...)

    _, _, a_p1 = intersectionlinecircle(p_z, temps[1], temps[1], radius)
    _, a_p2, _ = intersectionlinecircle(p_z, temps[1], p_z, radius)
    arrow(a_p2, a_p1; arrow_configs...)

    _, _, a_p1 = intersectionlinecircle(p_c, temps[1], temps[1], radius)
    _, a_p2, _ = intersectionlinecircle(p_c, temps[1], p_c, radius)
    arrow(a_p2, a_p1; arrow_configs...)

    _, a_p1, _ = intersectionlinecircle(p_x, p_z, p_x, radius)
    _, _, a_p2 = intersectionlinecircle(p_x, p_z, p_z, radius)
    arrow(a_p1, a_p2; arrow_configs...)

    _, a_p1, _ = intersectionlinecircle(p_b, p_z, p_b, radius)
    _, _, a_p2 = intersectionlinecircle(p_b, p_z, p_z, radius)
    arrow(a_p1, a_p2; arrow_configs...)


    # line(p_x, temps[4])
    # line(p_A, temps[3])
    strokepath()

    xaxis = Point[]

    expr = String[
        "f(u) = transpose(u)",
        "f(u, v) = uv",
        "f(M, v) = Mv",
        "f(x1, x2, x3) = x1 + x2 + x3"
    ]

    for (p, ex) in zip(temps, reverse(expr))
        sethue(node_color)
        circle(p, radius, :fill)
        fontsize(20)
        sethue("black")
        text(ex, p - (radius + 5, 0), halign=:right, valign=:middle)
    end

    sethue(node_color)
    circle(p_A, radius, :fill)
    circle(p_x, radius, :fill)
    circle(p_b, radius, :fill)
    circle(p_z, radius, :fill)
    circle(p_c, radius, :fill)
    # circle(xaxis[])

    sethue("black")
    text("f(u, v) = uv", p_z - (radius + 5, 0), halign=:right, valign=:middle)

    fontsize(30)
    configs = (halign=:center, valign=:middle)
    text("A", p_A; configs...)
    text("x", p_x; configs...)
    text("b", p_b; configs...)
    text("c", p_c; configs...)
end

function move_frame(scene, framenumber)
    node_color = "#aaf76c"
    radius = 40
    p1 = Point(200, -200)
    p2 = Point(-200, 200)
    temps = Point[between(p1, p2, i) for i in LinRange(0, 1, 4)]
    height = abs(temps[3].y - temps[4].y)
    width = abs(temps[3].x - temps[4].x)
    p_A = Point(temps[2].x, temps[4].y)
    p_x = Point(temps[3].x, temps[4].y + height)
    p_b = Point(temps[1].x, temps[4].y + height)
    p_c = Point(p_b.x + 2 * width, p_b.y)
    p_z = Point(temps[2].x + 2 * width, temps[4].y)

    sethue(node_color)
    setopacity(0.8)
    if framenumber == 1
        circle(p_x, radius, :fill)
        circle(p_A, radius, :fill)
        circle(p_b, radius, :fill)
        circle(p_c, radius, :fill)
    elseif framenumber == 2
        circle(temps[4], radius, :fill)
        circle(temps[3], radius, :fill)
        circle(p_z, radius, :fill)
    elseif framenumber == 3
        circle(temps[2], radius, :fill)
    elseif framenumber == 4
        circle(temps[1], radius, :fill)
    end

    sethue("black")
    fontsize(30)
    configs = (halign=:center, valign=:middle)
    text("A", p_A; configs...)
    text("x", p_x; configs...)
    text("b", p_b; configs...)
    text("c", p_c; configs...)
end

animate(demo, [
    # Scene(demo, backdrop, 0:200),
    Scene(demo, frame, 0:4)
    Scene(demo, move_frame, 0:4)
    ],
    framerate=2,
    creategif=true,
    pathname="test/comput-graph.gif"
)
