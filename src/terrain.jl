abstract type Terrain end

struct FlatTerrain <: Terrain
end

struct StepTerrain <: Terrain
    N::Int64         # Number of steps
    dx::Float64      # Height of each step
    dz::Float64      # Width of each step
    r::Float64       # Sharpness of corners
end

struct RampTerrain <: Terrain
    mx::Float64      # Slope in x direction
    my::Float64      # Slope in y direction
end

function getHeight(terrain::FlatTerrain, x, y)
    return 0
end

function getSlope(terrain::FlatTerrain, x, y)
    return 0, 0
end

function getHeight(terrain::RampTerrain, x, y)
    return terrain.mx*x + terrain.my*y
end

function getSlope(terrain::RampTerrain, x, y)
    return terrain.mx, terrain.my
end

function getHeight(terrain::StepTerrain, x, y)
    z = 0
    for i = 1:terrain.N
        z += terrain.dz*(1+tanh((x/terrain.dx - i)*terrain.r))
    end
    return z
end

function getSlope(terrain::StepTerrain, x, y)
    gx = 0
    for i = 1:terrain.N
        gx += terrain.dz*(dtanh((x/terrain.dx - i)*terrain.r))*terrain.r/terrain.dx
    end    
    return gx, 0
end

function plotTerrain(terrain::Terrain)
    x = 0.:.01:1.
    y = -.5:.01:.5
    z = zeros(length(x), length(y))
    for i = 1:length(x)
        for j = 1:length(y)
            z[i, j] = getHeight(terrain, x[i], y[i])
        end
    end
    surface(y, x, z, aspect_ratio = :equal)
end

function dtanh(x)
    return 1-tanh(x)^2
end