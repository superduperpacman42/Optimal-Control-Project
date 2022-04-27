function reference_trajectory(model::SimpleQuadruped, times, modes;
        xinit = 0.0,
        xterm = 10.0
    )
    
    height = 0.5*model.ℓmax # just guessed on this
    
    # Some useful variables
    n,m = size(model)
    tf = times[end]
    N = length(times)
    Δx = xterm - xinit
    
    mb,g = model.mb, model.g
    body_width = 2*abs(model.s1[2]) # from FL shoulder y pos wrt COM
    body_length = 2*abs(model.s1[1]) # from FL shoulder x pos wrt COM
    
    # initialization
    xref = zeros(n,N)
    uref = zeros(m,N)
    
    # linearly interpolate x-pos
    # constant y-pos
    # velocities by taking difference
    
    xs = range(xinit,xterm,length=N)
    
    dt = tf/Δx
    
    # do this smarter in the future
    # to make a less boring trajectory
    
    for k = 1:N-1
          
        xref[1,k] = xs[k] # body x
        xref[2,k] = 0 # body y
        xref[3,k] = height # body z
        
        # no body rotation in reference
        xref[4,k] = 1 # no rotation
        
        # foot index: (1,2,3,4) = (FL,BL,FR,BR)
        
        # foot x pos
        xref[8,k] = xs[k] + (body_length/2)
        xref[11,k] = xs[k] - (body_length/2)
        xref[14,k] = xs[k] + (body_length/2)
        xref[17,k] = xs[k] - (body_length/2)
        
        # foot y pos
        xref[9,k] = body_width / 2
        xref[12,k] = body_width / 2
        xref[15,k] = -body_width / 2
        xref[18,k] = -body_width / 2
        
        # foot z pos
        # keep at 0
        if k > 1
            xref[20,k] = (xs[k+1] - xs[k])/dt # body x vel
            xref[26,k] = (xs[k+1] - xs[k])/dt # foot1 x vel
            xref[29,k] = (xs[k+1] - xs[k])/dt # foot2 x vel
            xref[32,k] = (xs[k+1] - xs[k])/dt # foot3 x vel
            xref[35,k] = (xs[k+1] - xs[k])/dt # foot4 x vel
        end
    end
    
    # end state
    # set terminal positions, all velocities should 0 out
    
    # body pos
    xref[1,N] = xterm
    xref[2,N] = 0
    xref[3,N] = height
    
    xref[4,N] = 1 # no rotation
    xref[5,N] = 0 # 5:7 already set to 0, so this is unnecessary
    xref[6,N] = 0
    xref[7,N] = 0
    
    # foot x pos
    xref[8,N] = xterm + (body_length/2)
    xref[11,N] = xterm - (body_length/2)
    xref[14,N] = xterm + (body_length/2)
    xref[17,N] = xterm - (body_length/2)

    # foot y pos
    xref[9,N] = body_width / 2
    xref[12,N] = body_width / 2
    xref[15,N] = -body_width / 2
    xref[18,N] = -body_width / 2
    
    
    # reference trajectory
    uref .= kron(ones(N)', -[0;0;0.5*mb*g; 0;0;0.5*mb*g; 0;0;0.5*mb*g; 0;0;0.5*mb*g]) # 1/2mg for each foot, ignoring foot mass
    for k = 1:N-1
        uref[[3,6,9,12],k] .*= modes[:,k]
    end
    
    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(xref)]
    Uref = [SVector{m}(u) for u in eachcol(uref)]
    return Xref, Uref
end