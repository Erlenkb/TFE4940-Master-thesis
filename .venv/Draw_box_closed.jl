

function draw_box_closed(Start_pos::Array{Float64,1}, wall::Int64, d_box::Float64, b_box::Float64, h_box::Float64, N_QRD::Int64, b_QRD::Float64, d_max_QRD::Float64, d_vegger::Float64, d_bakplate::Float64, d_skilleplate::Float64, d_absorbent_QRD::Float64)


    if wall == 1
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]],
        [Start_pos[2], Start_pos[2] - b_box],
        [Start_pos[3], Start_pos[3] + h_box])
        # w2
        w2 = ([Start_pos[1], Start_pos[1]] + d_box / 2, 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w3
        w3 = ([Start_pos[1] + d_box / 2, Start_pos[1] + d_box / 2], 
        [Start_pos[2], Start_pos[2]] - d_vegger, 
        [Start_pos[3], Start_pos[3] + h_box])
        # w4
        w4 = ([Start_pos[1] + d_box / 2, Start_pos[1] + d_bakplate], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w5
        w5 = ([Start_pos[1] + d_bakplate, Start_pos[1] + d_bakplate], 
        [Start_pos[2] - d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w6
        w6 = ([Start_pos[1] + d_bakplate, Start_pos[1] + d_box / 2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w7
        w7 = ([Start_pos[1] + d_box / 2, Start_pos[1] + d_box / 2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2]] - b_box, 
        [Start_pos[3], Start_pos[3] + h_box])
        # w8
        w8 = ([Start_pos[1] + d_box / 2, Start_pos[1]], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        #############################################

        ################## QRD Box ##################
        # w9
        w9 = ([Start_pos[1] + d_box / 2, Start_pos[1] + d_box], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w10
        w10 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2], Start_pos[2]] - d_vegger, 
        [Start_pos[3], Start_pos[3] + h_box])
        # w11
        w11 = ([Start_pos[1] + d_box, Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box / 2], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w12
        w12 = ([Start_pos[1] + d_absorbent_QRD + d_box / 2, Start_pos[1] + d_box / 2], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w13
        w13 = ([Start_pos[1] + d_absorbent_QRD + d_box / 2, Start_pos[1] + d_absorbent_QRD + d_box / 2], 
        [Start_pos[2] - d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w14
        w14 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box / 2 , Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box / 2], 
        [Start_pos[2] - d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w15
        w15 = ([Start_pos[1] + d_absorbent_QRD + d_box / 2, Start_pos[1] + d_box / 2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w16
        w16 = ([Start_pos[1] + d_box / 2, Start_pos[1] + d_box / 2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w17
        w2 = ([Start_pos[1] + d_box / 2, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w18
        w2 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w19
        w2 = ([Start_pos[1] + d_box, Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w20
        #################################################

    elseif wall == 2
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]],
        [Start_pos[2], Start_pos[2] + b_box],
        [Start_pos[3], Start_pos[3] + h_box])
        # w2
        w2 = ([Start_pos[1], Start_pos[1] - d_box/2], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w3
        w3 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2], Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w4
        w4 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_bakplate], 
        [Start_pos[2] + d_vegger, Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w5
        w5 = ([Start_pos[1] - d_bakplate, Start_pos[1] - d_bakplate], 
        [Start_pos[2] + d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w6
        w6 = ([Start_pos[1] - d_bakplate, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w7
        w7 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w8
        w8 = ([Start_pos[1] - d_box/2, Start_pos[1]], 
        [Start_pos[2] + b_box, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        #############################################

        ################## QRD Box ##################
        # w9
        w9 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w10
        w10 = ([Start_pos[1] - d_box, Start_pos[1] - d_box], 
        [Start_pos[2], Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w11
        w11 = ([Start_pos[1] - d_box, Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w12
        w12 = ([Start_pos[1] - d_absorbent_QRD - d_box/2, Start_pos[1] - d_box / 2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w13
        w13 = ([Start_pos[1] - d_absorbent_QRD - d_box/2, Start_pos[1] - d_absorbent_QRD - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w14
        w14 = ([Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2, Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w15
        w15 = ([Start_pos[1] - d_absorbent_QRD - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w16
        w16 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w17
        w17 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box], 
        [Start_pos[2] + b_box, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w18
        w18 = ([Start_pos[1] - d_box, Start_pos[1] - d_box], 
        [Start_pos[2] + b_box, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w19
        w19 = ([Start_pos[1] - d_box, Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w20
        #################################################

    elseif wall == 3
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]] + b_box,
        [Start_pos[2], Start_pos[2]],
        [Start_pos[3], Start_pos[3] + h_box])
        # w2
        w2 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2], Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w3
        w3 = ([Start_pos[1], Start_pos[1]] + d_vegger, 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w4
        w4 = ([Start_pos[1] + d_vegger, Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w5
        w5 = ([Start_pos[1] + d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_bakplate, Start_pos[2] + d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w6
        w6 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_bakplate, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w7
        w7 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w8
        w8 = ([Start_pos[1] + b_box, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w9
        w9 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w10
        w10 = ([Start_pos[1], Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_box, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w11
        w11 = ([Start_pos[1] + d_vegger, Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w12
        w12 = ([Start_pos[1] + d_vegger, Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w13
        w13 = ([Start_pos[1] + d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_box/2, Start_pos[2] + d_absorbent_QRD + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w14
        w14 = ([Start_pos[1] + d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w15
        w15 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w16
        w16 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w17
        w17 = ([Start_pos[1] + b_box, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w18
        w18 = ([Start_pos[1] + b_box, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_box, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w19
        w19 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_box, Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w20
        ################################################

    elseif wall == 4
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]] - b_box,
        [Start_pos[2], Start_pos[2]],
        [Start_pos[3], Start_pos[3] + h_box])
        # w2
        w2 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2], Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w3
        w3 = ([Start_pos[1], Start_pos[1]] - d_vegger, 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w4
        w4 = ([Start_pos[1] - d_vegger, Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w5
        w5 = ([Start_pos[1] - d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_bakplate, Start_pos[2] - d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w6
        w6 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_bakplate, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w7
        w7 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w8
        w8 = ([Start_pos[1] - b_box, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w9
        w9 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w10
        w10 = ([Start_pos[1], Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_box, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w11
        w11 = ([Start_pos[1] - d_vegger, Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w12
        w12 = ([Start_pos[1] - d_vegger, Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w13
        w13 = ([Start_pos[1] - d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_box/2, Start_pos[2] - d_absorbent_QRD - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w14
        w14 = ([Start_pos[1] - d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2, Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w15
        w15 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w16
        w16 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w17
        w17 = ([Start_pos[1] - b_box, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w18
        w18 = ([Start_pos[1] - b_box, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_box, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w19
        w19 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_box, Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        # w20
        ################################################
        
    end
    return [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19]
end

function heatmap3d(data::AbstractArray{T,3}, d_step::T) where T<:Real
    x = collect(1:size(data,1))
    y = collect(1:size(data,2))
    z = collect(1:size(data,3))
    values = [data[i,j,k] == 0 ? NaN : data[i,j,k]/(maximum(data)) for i in x, j in y, k in z]
    heatmap!(x, y, values, c=:blues, colorbar=false, seriescolor=:blues, markerstrokecolor=:blues, markersize=d_step*4, z = z, transpose=true)
end

function plot_3d_heatmap(data::AbstractArray{T, 3}, d_step::Real) where T <: Real
    x = LinRange(1, size(data, 1), size(data, 1))
    y = LinRange(1, size(data, 2), size(data, 2))
    z = LinRange(1, size(data, 3), size(data, 3))
    cmap = cgrad([:black, :white], [0, 1])
    alpha = 0.1 .+ 0.9 .* (data .!= 0)
    heatmap(x, y, z, data, c = cmap, alpha = alpha, colorbar = false, aspect_ratio = :equal,
            xticks = (1:size(data, 1),), yticks = (1:size(data, 2),), zticks = (1:size(data, 3),),
            xlabel = "x", ylabel = "y", zlabel = "z",
            camera=(30,-45), background_color = :white, dpi = 300)
end

