l = 0.2
w = 0.1
h = 0.05

function visualize!(vis, Xs, h, terrain)
    set_mesh!(vis, Xs[1], terrain)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / h)))
    r_foot = 0.05
    for t = 1:length(Xs)
        x = Xs[t]
        p_body = x[1:3]
        q = x[4:7]
        angs = q2angleaxis(q)
        p_f1 = x[8:10]
        p_f2 = x[11:13]
        p_f3 = x[14:16]
        p_f4 = x[17:19]
        
        z_shift = [0.0, 0.0, r_foot]
        b_shift = 1/2 * [-l, -w, 0.0] 
        # body_shift = [0.0, 0.0, r_foot]  
        
        MeshCat.atframe(anim, t) do 
            H = compose(Translation(p_body + b_shift), LinearMap(AngleAxis(angs[1], angs[2], angs[3], angs[4])))
            settransform!(vis["body"], H)
            settransform!(vis["f1"], Translation(p_f1 + z_shift))
            settransform!(vis["f2"], Translation(p_f2 + z_shift))
            settransform!(vis["f3"], Translation(p_f3 + z_shift))
            settransform!(vis["f4"], Translation(p_f4 + z_shift))
        end
        
    end
    MeshCat.setanimation!(vis, anim)
end

function q2angleaxis(q)
    aa = AngleAxis(UnitQuaternion(q))
    return [aa.theta; aa.axis_x; aa.axis_y; aa.axis_z]
end

function set_mesh!(vis, initState, terrain)
    r_foot = 0.05
    bodyFL = Vec(0.0, 0.0, 0.0)
    bodyBR = Vec(l, w, h)
    init_loc = 1/2 * [l, w, h] 
    
    footSphere = Sphere(Point3f0(0), convert(Float32, r_foot))
    bodyRect = HyperRectangle(bodyFL, bodyBR)
    setvisible!(vis["/Background"], true)
    setobject!(vis["f1"], footSphere, MeshPhongMaterial(color = RGBA(1.0, 1.0, 0.0, 1.0)))
    setobject!(vis["f2"], footSphere, MeshPhongMaterial(color = RGBA(1.0, 1.0, 0.0, 1.0)))
    setobject!(vis["f3"], footSphere, MeshPhongMaterial(color = RGBA(1.0, 1.0, 0.0, 1.0)))
    setobject!(vis["f4"], footSphere, MeshPhongMaterial(color = RGBA(1.0, 1.0, 0.0, 1.0)))
    setobject!(vis["body"], bodyRect, MeshPhongMaterial(color = RGBA(0.0, 1.0, 1.0, 1.0)))
    z_shift = [0.0, 0.0, r_foot/2]
    # body_shift = initState[1:3]
    
    settransform!(vis["f1"], Translation(initState[8:10] + z_shift))
    settransform!(vis["f2"], Translation(initState[11:13] + z_shift))
    settransform!(vis["f3"], Translation(initState[14:16] + z_shift))
    settransform!(vis["f4"], Translation(initState[17:19] + z_shift))
    settransform!(vis["body"], Translation(initState[1:3] - init_loc))
    
    x = 0.:.001:1.
    y = -.5:.001:.5
    verts = zeros(Point3f0, length(x)*length(y))
    for i = 1:length(x)
        for j = 1:length(y)
            verts[(i-1)*length(x)+j] = Point3f0(x[i], y[j], getHeight(terrain, x[i], y[j]))
        end
    end
    colors = [RGB((p.*[1,1,0] .+ [0, 0.5, 0])...) for p in verts]
    setobject!(vis, PointCloud(verts, colors))
end

function unpack_params()
    
end