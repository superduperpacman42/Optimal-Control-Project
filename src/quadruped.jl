using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using Random
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff
using RobotDynamics
using Printf
import RigidBodyDynamics.RigidBody

const URDFPATH = joinpath(@__DIR__, "..","a1","urdf","a1.urdf")
const STATEMAP = (
    foot_joint_x = (1,1),
    foot_joint_y = (2,2),
    foot_joint_z = (3,3),
    FR_hip_joint = (4,7),
    FL_hip_joint = (5,8),
    RR_hip_joint = (6,6),
    RL_hip_joint = (7,9),
    FR_thigh_joint = (8,10),
    FL_thigh_joint = (9,11),
    RR_thigh_joint = (10,5),
    RL_thigh_joint = (11,12),
    FR_calf_joint = (12,13),
    FL_calf_joint = (13,14),
    RR_calf_joint = (14,4),
    RL_calf_joint = (15,15)
)
"""
    attach_foot!(mech, [foot; revolute])

Attach one of the feet of the quadruped to the ground using revolute joints. This version
only adds 2 revolute joints if `revolute=true`, disallowing rotation in z at the foot. 

The order of the joints in the mechanism will be modified after this function.
"""
function attach_foot!(mech::Mechanism{T}, foot="RR"; revolute::Bool=true) where T
    # Get the relevant bodies from the model
    foot = findbody(mech, foot * "_foot")
    trunk = findbody(mech, "trunk")
    world = findbody(mech, "world")

    # Get the location of the foot
    state = MechanismState(mech)
    trunk_to_foot = translation(relative_transform(state, 
        default_frame(trunk), default_frame(foot)))
    foot_location = SA[trunk_to_foot[1], trunk_to_foot[2], 0]  # set the z height to 0

    # Build the new foot joint
    if !revolute 
        foot_joint = Joint("foot_joint", QuaternionSpherical{T}())
        world_to_joint = Transform3D(
            frame_before(foot_joint),
            default_frame(world),
            -foot_location        
        )

        # Attach to model
        attach!(mech,
            world,
            foot,
            foot_joint,
            joint_pose = world_to_joint,
        )
        remove_joint!(mech, findjoint(mech, "base_to_world"))
    else
        # Create dummy bodies 
        dummy1 = RigidBody{T}("dummy1")
        dummy2 = RigidBody{T}("dummy2")
        ax1 = SA[1,0,0]
        ax2 = SA[0,2,0]
        for body ∈ (dummy1, dummy2)
            inertia = SpatialInertia(default_frame(body),
                moment = ax1*ax1'*1e-3,
                com    = SA[0,0,0],
                mass   = 1e-3
            )
            spatial_inertia!(body, inertia)
        end

        # X-Joint
        foot_joint_x = Joint("foot_joint_x", Revolute{T}(SA[-1,0,0]))
        foot_joint_y = Joint("foot_joint_y", Revolute{T}(SA[0,-1,0]))
        foot_joint_z = Joint("foot_joint_z", Revolute{T}(SA[0,0,-1]))
        # foot_joint_z = Joint("foot_joint_z", Fixed{T}())
        world_to_joint = Transform3D(
            frame_before(foot_joint_y),
            default_frame(world),
            -foot_location        
        )
        attach!(mech,
            world,
            dummy1,
            foot_joint_y,
            joint_pose = world_to_joint
        )

        # Y-Joint
        dummy_to_dummy = Transform3D(
            frame_before(foot_joint_x),
            default_frame(dummy1),
            SA[0,0,0]
        )
        attach!(mech,
            dummy1,
            foot,
            foot_joint_x,
            joint_pose = dummy_to_dummy 
        )

        remove_joint!(mech, findjoint(mech, "base_to_world"))
    end
end

"""
    build_quadruped()

Read the A1 urdf and attach the foot to the floor. Returns a `Mechanism` type.
"""
function build_quadruped()
    a1 = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false) 
    attach_foot!(a1)
    return a1
end

"""
    UnitreeA1 <: AbstractModel

A model of the UnitreeA1 quadruped, with one foot attached to the floor. 
Uses the `RobotDynamics` interface to and uses `RigidBodyDynamics` to evaluate 
the continuous-time dynamics.
"""
struct UnitreeA1{C} <: AbstractModel
    mech::Mechanism{Float64}
    statecache::C
    dyncache::DynamicsResultCache{Float64}
    xdot::Vector{Float64}
    function UnitreeA1(mech::Mechanism)
        N = num_positions(mech) + num_velocities(mech)
        statecache = StateCache(mech)
        rescache = DynamicsResultCache(mech)
        xdot = zeros(N)
        new{typeof(statecache)}(mech, statecache, rescache, xdot)
    end
end
function UnitreeA1()
    UnitreeA1(build_quadruped())
end

RobotDynamics.state_dim(model::UnitreeA1) = 28 
RobotDynamics.control_dim(model::UnitreeA1) = 12 
function get_partition(model::UnitreeA1)
    n,m = state_dim(model), control_dim(model)
    return 1:n, n .+ (1:m), n+m .+ (1:n)
end

function RobotDynamics.dynamics(model::UnitreeA1, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1,T2} 
    T = promote_type(T1,T2)
    state = model.statecache[T]
    res = model.dyncache[T]

    copyto!(state, x)
    τ = [zeros(2); u]
    dynamics!(res, state, τ)
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

function jacobian(model::UnitreeA1, x, u)
    z = StaticKnotPoint(SVector{28}(x),SVector{12}(u),0.1,0.0)
    ∇f = RobotDynamics.DynamicsJacobian(model)
    jacobian!(∇f, model, z)
    return ∇f
end

function discrete_jacobian(::Type{Q}, model::UnitreeA1, x, u, dt) where Q <: QuadratureRule 
    z = StaticKnotPoint(SVector{28}(x),SVector{12}(u),dt,0.0)
    ∇f = RobotDynamics.DynamicsJacobian(model)
    discrete_jacobian!(Q, ∇f, model, z)
    return ∇f
end

# Set initial guess
"""
    initial_state(model)

Get the default initial state for the model
"""
function initial_state(model::UnitreeA1)
    state = model.statecache[Float64]
    a1 = model.mech
    zero!(state)
    leg = ("FR","FL","RR","RL")
    for i = 1:4
        s = isodd(i) ? 1 : -1
        f = i < 3 ? 1 : -1
        set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), deg2rad(-20s))
        set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), deg2rad(-30f))
        set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), deg2rad(10f))
    end
    set_configuration!(state, findjoint(a1, "foot_joint_y"), deg2rad(30))

    return [configuration(state); velocity(state)]
end

function initialize_visualizer(a1::UnitreeA1)
    vis = Visualizer()
    curdir = pwd()
    delete!(vis)
    cd(joinpath(@__DIR__,"..","a1","urdf"))
    mvis = MechanismVisualizer(a1.mech, URDFVisuals(URDFPATH), vis)
    cd(curdir)
    return mvis
end

function visualize!(mvis, model::UnitreeA1, tf::Real, X)
    fps = Int(round((length(X)-1)/tf))
    anim = MeshCat.Animation(fps)
    for (k,x) in enumerate(X)
        atframe(anim, k) do
            set_configuration!(mvis, x[1:14])
        end
    end
    setanimation!(mvis, anim)
end

############################################################################################
#                                 EQUILIBRIUM SOLVER (HW1)
############################################################################################
"""
    kkt_conditions(model, x, u, λ, A, B)

Evaluate the KKT conditions for the optimization problem in Part(a). The `model` provides access to the dynamics and the initial state (see above).
The KKT conditions are evaluated with states `x`, controls `u` and Lagrange multipliers `λ`. The `A` and `B` matrices are the continuous-time dynamics Jacobians.
"""
function kkt_conditions(model::UnitreeA1, x, u, λ, A, B; α=1e-3)
    # Get initial state from the model (if you need it)
    x_guess = initial_state(model)
    
    # TODO: Fill out these lines
    ∇ₓL = x - x_guess + A'λ
    ∇ᵤL = α*u + B'λ
    c = dynamics(model,x,u)
    
    # Return the concatenated vector
    return [∇ₓL; ∇ᵤL; c]
end

"""
    kkt_jacobian(model, x, u, λ, A, B, [ρ])

Form the Jacobian of the KKT conditions. Uses a Gauss-Newton approximation to avoid 2nd order derivatives of the dynamics.
Evaluated at states `x`, controls `u`, Lagrange multipliers `λ`, provided the dynamics Jacobians `A` and `B`.

The optional parameter `ρ` can be used to add regularization to the KKT system (recommended). 
This should be applied along the diagonal of the KKT system, and should be positive for primal variables and negative for dual variables.
For example: `Hreg = H + Diagonal([ones(n+m); -ones(n)])*ρ`.
"""
function kkt_jacobian(model::UnitreeA1, x, u, λ, A, B, ρ=1e-5; α=1e-3)
    # HINT: You may find these ranges to be helpful
    parts = get_partition(model)
    ix = parts[1]     # state variables
    iu = parts[2]     # control variables
    ic = parts[3]     # constraints
    n = length(ix)    # number of states
    m = length(iu)    # number of controls
    
    # TODO: Create the KKT matrix
    H = zeros(2n+m, 2n+m)
    H[ix,ix] .= I(n)*(1 + ρ)
    H[ix,ic] .= A'
    H[iu,iu] .= I(m)*(1e-3 + ρ)
    H[iu,ic] .= B'
    H[ic,ix] .= A
    H[ic,iu] .= B
    H[ic,ic] .= -I(n)*ρ   # add regularization
    
    return Symmetric(H,:L)
end

"""
    newton_solve(model, x_guess, [mvis; verbose])

Use Newton's method to find an equilibrium point for the quadruped, starting at `x_guess`. 
Should return the optimal states, controls, and Lagrange multipliers, along with the vector
of 2-norm residuals for each iteration.

# Optional Arguments
* `mvis`: Pass in a MechanismVisualizer to visualize the iterations in a MeshCat window
* `verbose`: flag to print out solver iterations
* `max_iters`: maximum number of solver iterations
* `ρ`: regularization
"""
function newton_solve(model::UnitreeA1, x_guess, mvis=nothing; verbose=true, max_iters=50, ρ=1e-3, tol=1e-6, R=1e-3)
    u = zeros(12)
    x = copy(x_guess) 
    λ = zero(x_guess)
    ix,iu,ix2 = get_partition(model)

    # TODO: Finish the function
    # SOLUTION
    res_hist = Float64[]
    for i in 1:max_iters
        # Evaluate the Dynamics Jacobian
        ∇f = jacobian(model, x, u)
        A = ∇f[:,ix]
        B = ∇f[:,iu]
        
        # Form the KKT System
        r = kkt_conditions(model, x, u, λ, A, B, α=R)
        H = kkt_jacobian(model, x, u, λ, A, B, ρ, α=R)

        # Calculate delta x and update 
        δx = -H \ r
        x = x + δx[ix]
        u = u + δx[iu]
        λ = λ + δx[ix2]
        
        # Display the intermediate results
        if !isnothing(mvis)
            set_configuration!(mvis, x[1:15])
            sleep(0.1)
        end

        # TODO: Check convergence
        verbose && @printf("Iter %2d: residual = %0.2e\n", i, norm(r))
        push!(res_hist, norm(r))
        if norm(r) < tol 
            break
        end
    end
    return x,u,λ, res_hist
end
