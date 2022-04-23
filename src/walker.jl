using GeometryBasics
using CoordinateTransformations, Rotations
using RobotDynamics
using Colors
using StaticArrays 
using MeshCat
using Blink
using LinearAlgebra
using TrajOptPlots

Base.@kwdef struct SimpleWalker <: AbstractModel
    g::Float64 = 9.81
    mb::Float64 = 5.0
    mf::Float64 = 1.0
    ℓ_min::Float64 = 0.5
    ℓ_max::Float64 = 1.5
end
RobotDynamics.state_dim(::SimpleWalker) = 12
RobotDynamics.control_dim(::SimpleWalker) = 3

function RobotDynamics.dynamics(model::SimpleWalker, x, u, t, mode::Integer)
     #Foot 1 is in contact
    mb,mf = model.mb, model.mf
    g = model.g
    M = Diagonal([mb mb mf mf mf mf])
    
    rb  = x[1:2]   # position of the body
    rf1 = x[3:4]   # position of foot 1
    rf2 = x[5:6]   # position of foot 2
    v   = x[7:12]  # velocities
    
    ℓ1x = (rb[1]-rf1[1])/norm(rb-rf1)
    ℓ1y = (rb[2]-rf1[2])/norm(rb-rf1)
    ℓ2x = (rb[1]-rf2[1])/norm(rb-rf2)
    ℓ2y = (rb[2]-rf2[2])/norm(rb-rf2)
    
    if mode == 1
        B = [ℓ1x  ℓ2x  ℓ1y-ℓ2y;
            ℓ1y  ℓ2y  ℓ2x-ℓ1x;
            0    0     0;
            0    0     0;
            0  -ℓ2x  ℓ2y;
            0  -ℓ2y -ℓ2x;
        ]
        G = SA[0, -g, 0, 0, 0, -g]
    else
        B = [ℓ1x  ℓ2x  ℓ1y-ℓ2y;
            ℓ1y  ℓ2y  ℓ2x-ℓ1x;
            -ℓ1x   0  -ℓ1y;
            -ℓ1y   0   ℓ1x;
            0    0    0;
            0    0    0;
        ]
        G = SA[0, -g, 0, -g, 0, 0]
    end
    
    v̇ = G + M\(B*u)
    
    ẋ = [v; v̇]   
end

function jumpmap(model::SimpleWalker, x, mode) 
    if mode == 2
        SA[
            x[1], x[2], x[3], x[4], x[5],  x[6],
            x[7], x[8], 0   , 0   , x[11], x[12]
        ]
    else
        SA[
            x[1], x[2], x[3], x[4],  x[5], x[6],
            x[7], x[8], x[9], x[10], 0   , 0
        ]
    end
end

function rk4(model::SimpleWalker, x, u, t, dt, mode)
    f1 = dynamics(model, x, u, t, mode)
    f2 = dynamics(model, x + 0.5*dt*f1, u, t, mode)
    f3 = dynamics(model, x + 0.5*dt*f2, u, t, mode)
    f4 = dynamics(model, x +     dt*f3, u, t, mode)
    return x + dt*(f1 + 2*f2 + 2*f3 + f4)/6
end

function set_mesh!(vis, model::SimpleWalker)
    l,w,h = 0.4,0.4,0.6
    body = HyperRectangle(Vec(-l/2,-w/2,0), Vec(l,w,h)) 
    body = Sphere(Point3f0(0,0,0), 7l/16)
    setobject!(vis["robot"]["torso"]["body"], body, MeshPhongMaterial(color=colorant"gray"))
    axle = Cylinder(Point3f0(0,0,0), Point3f0(0,w/2,0), 0.03f0)
    setobject!(vis["robot"]["torso"]["Laxle"], axle, MeshPhongMaterial(color=colorant"black"))
    setobject!(vis["robot"]["torso"]["Raxle"], axle, MeshPhongMaterial(color=colorant"black"))
    settransform!(vis["robot"]["torso"]["Laxle"], Translation(0,+l/4,0))
    settransform!(vis["robot"]["torso"]["Raxle"], Translation(0,-3l/4,0))

    foot = HyperSphere(Point3f0(0,0,0f0), 0.05f0)
    Lfoot = setobject!(vis["robot"]["Lfoot"]["geom"], foot, MeshPhongMaterial(color=colorant"firebrick"))
    setobject!(vis["robot"]["Rfoot"]["geom"], foot, MeshPhongMaterial(color=colorant"firebrick"))
    settransform!(vis["robot"]["Lfoot"]["geom"], Translation(0,+l/2,0))
    settransform!(vis["robot"]["Rfoot"]["geom"], Translation(0,-l/2,0))

    Lleg = Cylinder(Point3f0(0,+l/2,0), Point3f0(0,+l/2,1), 0.03f0)
    Rleg = Cylinder(Point3f0(0,-l/2,0), Point3f0(0,-l/2,1), 0.03f0)
    setobject!(vis["robot"]["torso"]["Lleg"]["geom"], Lleg, MeshPhongMaterial(color=colorant=colorant"green"))
    setobject!(vis["robot"]["torso"]["Rleg"]["geom"], Rleg, MeshPhongMaterial(color=colorant=colorant"green"))
    # settransform!(vis["robot"]["torso"]["Lleg"]["geom"], Translation(0,+l/2,0))
    # settransform!(vis["robot"]["torso"]["Rleg"]["geom"], Translation(0,-l/2,0))

    return Lfoot
end
function TrajOptPlots.visualize!(vis, model::SimpleWalker, x::StaticVector)
    xb,yb = x[1],x[2]
    xl,yl = x[3],x[4]
    xr,yr = x[5],x[6]
    settransform!(vis["robot"]["torso"], Translation(xb,0,yb))
    settransform!(vis["robot"]["Lfoot"], Translation(xl,0,yl))
    settransform!(vis["robot"]["Rfoot"], Translation(xr,0,yr))

    Llen = norm(SA[xl-xb, yl-yb])
    Rlen = norm(SA[xr-xb, yr-yb])
    θl = atan(xl-xb, yl-yb)
    θr = atan(xr-xb, yr-yb)
    settransform!(vis["robot"]["torso"]["Lleg"], LinearMap(RotY(θl)))
    settransform!(vis["robot"]["torso"]["Rleg"], LinearMap(RotY(θr)))

    settransform!(vis["robot"]["torso"]["Lleg"]["geom"], LinearMap(Diagonal(SA[1,1,Llen])))
    settransform!(vis["robot"]["torso"]["Rleg"]["geom"], LinearMap(Diagonal(SA[1,1,Rlen])))
end

function stance1_dynamics(model::SimpleWalker, x,u)
    #Foot 1 is in contact
    mb,mf = model.mb, model.mf
    g = model.g

    M = Diagonal(SA[mb mb mf mf mf mf])
    
    # rb  = x[1:2]   # position of the body
    # rf1 = x[3:4]   # position of foot 1
    # rf2 = x[5:6]   # position of foot 2
    # v   = x[7:12]  # velocities
    rb = SA[x[1], x[2]]
    rf1 = SA[x[3], x[4]]
    rf2 = SA[x[5], x[6]]
    v = SA[x[7], x[8], x[9], x[10], x[11], x[12]]
    
    
    ℓ1x = (rb[1]-rf1[1])/norm(rb-rf1)
    ℓ1y = (rb[2]-rf1[2])/norm(rb-rf1)
    ℓ2x = (rb[1]-rf2[1])/norm(rb-rf2)
    ℓ2y = (rb[2]-rf2[2])/norm(rb-rf2)
    
    B = SA[ℓ1x  ℓ2x  ℓ1y-ℓ2y;
         ℓ1y  ℓ2y  ℓ2x-ℓ1x;
          0    0     0;
          0    0     0;
          0  -ℓ2x  ℓ2y;
          0  -ℓ2y -ℓ2x]
    
    v̇ = SA[0; -g; 0; 0; 0; -g] + M\(B*u)
    
    ẋ = [v; v̇]
end

function stance2_dynamics(model::SimpleWalker, x,u)
    #Foot 2 is in contact
    mb,mf = model.mb, model.mf
    g = model.g
    M = Diagonal(SA[mb mb mf mf mf mf])
    
    # rb  = x[1:2]   # position of the body
    # rf1 = x[3:4]   # position of foot 1
    # rf2 = x[5:6]   # position of foot 2
    # v   = x[7:12]  # velocities
    rb = SA[x[1], x[2]]
    rf1 = SA[x[3], x[4]]
    rf2 = SA[x[5], x[6]]
    v = SA[x[7], x[8], x[9], x[10], x[11], x[12]]
    
    ℓ1x = (rb[1]-rf1[1])/norm(rb-rf1)
    ℓ1y = (rb[2]-rf1[2])/norm(rb-rf1)
    ℓ2x = (rb[1]-rf2[1])/norm(rb-rf2)
    ℓ2y = (rb[2]-rf2[2])/norm(rb-rf2)
    
    B = SA[ℓ1x  ℓ2x  ℓ1y-ℓ2y;
         ℓ1y  ℓ2y  ℓ2x-ℓ1x;
        -ℓ1x   0  -ℓ1y;
        -ℓ1y   0   ℓ1x;
          0    0    0;
          0    0    0]
    
    v̇ = SA[0; -g; 0; -g; 0; 0] + M\(B*u)
    
    ẋ = [v; v̇]
end

function stance1_dynamics_rk4(model, x,u, h)
    #RK4 integration with zero-order hold on u
    f1 = stance1_dynamics(model, x, u)
    f2 = stance1_dynamics(model, x + 0.5*h*f1, u)
    f3 = stance1_dynamics(model, x + 0.5*h*f2, u)
    f4 = stance1_dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function stance2_dynamics_rk4(model, x,u, h)
    #RK4 integration with zero-order hold on u
    f1 = stance2_dynamics(model, x, u)
    f2 = stance2_dynamics(model, x + 0.5*h*f1, u)
    f3 = stance2_dynamics(model, x + 0.5*h*f2, u)
    f4 = stance2_dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function stance1_jacobian(model::SimpleWalker, x, u, dt)
    xi = SVector{12}(1:12)
    ui = SVector{3}(1:3) .+ 12 
    f(z) = stance1_dynamics_rk4(model, z[xi], z[ui], dt)
    ForwardDiff.jacobian(f, [x; u])
end

function stance2_jacobian(model::SimpleWalker, x, u, dt)
    xi = SVector{12}(1:12)
    ui = SVector{3}(1:3) .+ 12 
    f(z) = stance2_dynamics_rk4(model, z[xi], z[ui], dt)
    ForwardDiff.jacobian(f, [x; u])
end

function jump1_map(x)
    #Foot 1 experiences inelastic collision
    xn = [x[1:8]; 0.0; 0.0; x[11:12]]
    return xn
end

function jump2_map(x)
    #Foot 2 experiences inelastic collision
    xn = [x[1:10]; 0.0; 0.0]
    return xn
end

jump1_jacobian() = 
    Diagonal(SA[1,1, 1,1, 1,1, 1,1, 0,0, 1,1])

jump2_jacobian() = 
    Diagonal(SA[1,1, 1,1, 1,1, 1,1, 1,1, 0,0])
