Base.@kwdef struct SimpleQuadruped <: AbstractModel
    g::Float64 = 9.81
    mb::Float64 = 5.0
    mf::Float64 = 1.0
    Jx::Float64 = 1.0
    Jy::Float64 = 1.0
    Jz::Float64 = 1.0
    ℓmin::Float64 = 0.0
    ℓmax::Float64 = 1.0
    s1::SVector{3,Float64} = [-1, 1, 0]
    s2::SVector{3,Float64} = [-1, -1, 0]
    s3::SVector{3,Float64} = [1, 1, 0]
    s4::SVector{3,Float64} = [1, -1, 0]
end
RobotDynamics.state_dim(::SimpleQuadruped) = 37
RobotDynamics.control_dim(::SimpleQuadruped) = 12

function skew(v)
    return [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end

function RobotDynamics.dynamics(model::SimpleQuadruped, x, u, t, mode)
    rb = x[1:3]
    ab = x[4:7]
    r1 = x[8:10]
    r2 = x[11:13]
    r3 = x[14:16]
    r4 = x[17:19]
    vb = x[20:22]
    wb = x[23:25]
    v1 = x[26:28]
    v2 = x[29:31]
    v3 = x[32:34]
    v4 = x[35:37]
    
    mb = model.mb
    mf = model.mf
    Jx = model.Jx
    Jy = model.Jy
    Jz = model.Jz
    J = Diagonal([Jx, Jy, Jz])
    g = [0, 0, model.g]
    
    M = Diagonal(vcat(ones(3)*mb, Jx, Jy, Jz, ones(12)*mf))
    V = vcat(-mb*g, zeros(3), -mf*g*(1-mode[1]), -mf*g*(1-mode[2]), -mf*g*(1-mode[3]), -mf*g*(1-mode[4]))
    C = vcat(zeros(3), cross(wb, J*wb), zeros(12))
    B = [-I(3) -I(3) -I(3) -I(3);
         skew(r1) skew(r2) skew(r3) skew(r4);
         Diagonal(vcat(ones(3)*(1-mode[1]), ones(3)*(1-mode[2]), ones(3)*(1-mode[3]), ones(3)*(1-mode[4])))]
    accel = inv(M)*(B*u+V-C)
    adot = 0.5*ab + vcat(0,wb)
    vel = vcat(vb, adot, v1, v2, v3, v4)
    return [vel; accel]
end

function rk4(model::SimpleQuadruped, x, u, t, dt, mode)
    f1 = dynamics(model, x, u, t, mode)
    f2 = dynamics(model, x + 0.5*dt*f1, u, t, mode)
    f3 = dynamics(model, x + 0.5*dt*f2, u, t, mode)
    f4 = dynamics(model, x +     dt*f3, u, t, mode)
    return x + dt*(f1 + 2*f2 + 2*f3 + f4)/6
end

function jumpmap(model::SimpleQuadruped, x, mode2) 
    x1 = x[:]
    for l = 1:4
        x1[26+(l-1)*3:26+(l-1)*3+2] .*= (1-mode2[l])
    end
    return x1
end

function discrete_jacobian(model::SimpleQuadruped, x, u, t, dt, mode)
    xi = SVector{37}(1:37)
    ui = SVector{12}(1:12) .+ 37 
    f(z) = rk4(model, z[xi], z[ui], t, dt, mode)
    jac = ForwardDiff.jacobian(f, [x; u])
    return jac[:,1:37], jac[:,38:end]
end

function jump_jacobian(mode2)
    return (vcat(ones(25), ones(3)*(1-mode2[1]), ones(3)*(1-mode2[2]), ones(3)*(1-mode2[3]), ones(3)*(1-mode2[4])))
end