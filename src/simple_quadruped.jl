Base.@kwdef struct SimpleQuadruped <: AbstractModel
    g::Float64 = 9.81                      # gravitational acceleration (m/s^2)
    mb::Float64 = 5.0                      # mass of the body (kg)
    mf::Float64 = 1.0                      # mass of an individual foot (kg)
    Jx::Float64 = 1.0                      # body moment of inertia about x-axis (kg m^2)
    Jy::Float64 = 1.0                      # body moment of inertia about y-axis (kg m^2)
    Jz::Float64 = 1.0                      # body moment of inertia about z-axis (kg m^2)
    ℓmin::Float64 = 0.0                    # minimum leg length (m)
    ℓmax::Float64 = 1.0                    # maximum leg length (m)
    s1::SVector{3,Float64} = [-1, 1, 0]    # front left shoulder relative to CoM (m)
    s2::SVector{3,Float64} = [-1, -1, 0]   # rear left shoulder relative to CoM (m)
    s3::SVector{3,Float64} = [1, 1, 0]     # front right shoulder relative to CoM (m)
    s4::SVector{3,Float64} = [1, -1, 0]    # rear right shoulder relative to CoM (m)
end
RobotDynamics.state_dim(::SimpleQuadruped) = 37
RobotDynamics.control_dim(::SimpleQuadruped) = 12

function simplifyQuadruped(full_model)
    mf = RigidBodyDynamics.spatial_inertia(findbody(full_model.mech, "RR_foot")).mass
    mb = RigidBodyDynamics.spatial_inertia(findbody(full_model.mech, "trunk")).mass + 
         RigidBodyDynamics.spatial_inertia(findbody(full_model.mech, "RR_hip")).mass*4 +
         RigidBodyDynamics.spatial_inertia(findbody(full_model.mech, "RR_thigh")).mass*4 +
         RigidBodyDynamics.spatial_inertia(findbody(full_model.mech, "RR_calf")).mass*4
    J = RigidBodyDynamics.spatial_inertia(findbody(full_model.mech, "trunk")).moment
    state = MechanismState(full_model.mech)
    zero!(state)
    leg = translation(relative_transform(state, 
          default_frame(findbody(full_model.mech, "RR_hip")),
          default_frame(findbody(full_model.mech, "RR_foot"))))
    s1 = translation(relative_transform(state, 
         default_frame(findbody(full_model.mech, "trunk")),
         default_frame(findbody(full_model.mech, "RR_hip"))))
    s1 = vcat(-s1[2], s1[1], s1[3])
    s2 = s1.*[1,-1,1]
    s3 = s1.*[-1,1,1]
    s4 = s1.*[-1,-1,1]

    model = SimpleQuadruped(mf=mf, mb=mb, Jx=J[1,1], Jy=J[2,2], Jz=J[3,3], ℓmax=leg[3], s1=s1, s2=s2, s3=s3, s4=s4)
end

function skew(v)
    return [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end

function RobotDynamics.dynamics(model::SimpleQuadruped, x, u, t, mode)
    rb = x[1:3]   # body position (m)
    ab = x[4:7]   # body orientation (quaternion)
    r1 = x[8:10]  # foot 1 position (m)
    r2 = x[11:13] # foot 2 position (m)
    r3 = x[14:16] # foot 3 position (m)
    r4 = x[17:19] # foot 4 position (m)
    vb = x[20:22] # body velocity (m/s)
    wb = x[23:25] # body angular velocity in body frame (rad/s)
    v1 = x[26:28] # foot 1 velocity (m/s)
    v2 = x[29:31] # foot 2 velocity (m/s)
    v3 = x[32:34] # foot 3 velocity (m/s)
    v4 = x[35:37] # foot 4 velocity (m/s)
    
    mb = model.mb
    mf = model.mf
    Jx = model.Jx
    Jy = model.Jy
    Jz = model.Jz
    J = Diagonal([Jx, Jy, Jz])
    g = [0, 0, -model.g]
    
    M = Diagonal(vcat(ones(3)*mb, Jx, Jy, Jz, ones(12)*mf))
    V = vcat(mb*g, zeros(3), mf*g*(1-mode[1]), mf*g*(1-mode[2]), mf*g*(1-mode[3]), mf*g*(1-mode[4]))
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