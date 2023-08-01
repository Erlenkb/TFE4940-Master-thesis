

function draw_box_closed(Start_pos::Array{Float64,1}, wall::Int64, d_box::Float64, b_box::Float64, h_box::Float64, N_QRD::Int64, b_QRD::Float64, d_max_QRD::Float64, d_vegger::Float64, d_bakplate::Float64, d_skilleplate::Float64, d_absorbent_QRD::Float64)

    prog_draw_box = Progress(21)


    if wall == 1
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]],
        [Start_pos[2] - b_box, Start_pos[2]],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w2
        w2 = ([Start_pos[1], Start_pos[1] + d_box/2], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w3
        w3 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box/2], 
        [Start_pos[2] - d_vegger, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w4
        w4 = ([Start_pos[1] + d_bakplate , Start_pos[1] + d_box/2], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w5
        w5 = ([Start_pos[1] + d_bakplate, Start_pos[1] + d_bakplate], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w6
        w6 = ([Start_pos[1] + d_bakplate, Start_pos[1] + d_box/2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w7
        w7 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box/2], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w8
        w8 = ([Start_pos[1], Start_pos[1] + d_box/2], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        #############################################

        ################## QRD Box ##################
        # w9
        w9 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w10
        w10 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - d_vegger, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w11
        w11 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w12
        w12 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box/2 + d_absorbent_QRD], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w13
        w13 = ([Start_pos[1] + d_absorbent_QRD + d_box/2, Start_pos[1] + d_absorbent_QRD + d_box/2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w14
        w14 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2 , Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w15
        w15 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box/2 + d_absorbent_QRD], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w16
        w16 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box/2], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w17
        w17 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w18
        w18 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w19
        w19 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w20
        w20 = ([Start_pos[1], Start_pos[1]+d_box],
        [Start_pos[2]-b_box, Start_pos[2]],
        [Start_pos[3]+h_box, Start_pos[3]+h_box])
        next!(prog_draw_box)
        
        # w21
        w21 = ([Start_pos[1], Start_pos[1]+d_box],
        [Start_pos[2]- b_box, Start_pos[2]],
        [Start_pos[3], Start_pos[3]])
        
        #################################################

    elseif wall == 2
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]],
        [Start_pos[2], Start_pos[2] + b_box],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w2
        w2 = ([Start_pos[1], Start_pos[1] - d_box/2], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w3
        w3 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2], Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w4
        w4 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_bakplate], 
        [Start_pos[2] + d_vegger, Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w5
        w5 = ([Start_pos[1] - d_bakplate, Start_pos[1] - d_bakplate], 
        [Start_pos[2] + d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w6
        w6 = ([Start_pos[1] - d_bakplate, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w7
        w7 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w8
        w8 = ([Start_pos[1] - d_box/2, Start_pos[1]], 
        [Start_pos[2] + b_box, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        #############################################

        ################## QRD Box ##################
        # w9
        w9 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w10
        w10 = ([Start_pos[1] - d_box, Start_pos[1] - d_box], 
        [Start_pos[2], Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w11
        w11 = ([Start_pos[1] - d_box, Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w12
        w12 = ([Start_pos[1] - d_absorbent_QRD - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w13
        w13 = ([Start_pos[1] - d_absorbent_QRD - d_box/2, Start_pos[1] - d_absorbent_QRD - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w14
        w14 = ([Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2, Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[2] + d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w15
        w15 = ([Start_pos[1] - d_absorbent_QRD - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w16
        w16 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w17
        w17 = ([Start_pos[1] - d_box/2, Start_pos[1] - d_box], 
        [Start_pos[2] + b_box, Start_pos[2] + b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w18
        w18 = ([Start_pos[1] - d_box, Start_pos[1] - d_box], 
        [Start_pos[2] + b_box, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w19
        w19 = ([Start_pos[1] - d_box, Start_pos[1] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[2] + b_box - d_vegger, Start_pos[2] + b_box - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w20
        w20 = ([Start_pos[1], Start_pos[1]-d_box],
        [Start_pos[2], Start_pos[2]+b_box],
        [Start_pos[3]+h_box, Start_pos[3]+h_box])
        next!(prog_draw_box)
        # w21
        w21 = ([Start_pos[1], Start_pos[1]-d_box],
        [Start_pos[2], Start_pos[2]+b_box],
        [Start_pos[3], Start_pos[3]])
        #################################################

    elseif wall == 3
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]] + b_box,
        [Start_pos[2], Start_pos[2]],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w2
        w2 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2], Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w3
        w3 = ([Start_pos[1], Start_pos[1]] + d_vegger, 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w4
        w4 = ([Start_pos[1] + d_vegger, Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w5
        w5 = ([Start_pos[1] + d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_bakplate, Start_pos[2] + d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w6
        w6 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_bakplate, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w7
        w7 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w8
        w8 = ([Start_pos[1] + b_box, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w9
        w9 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w10
        w10 = ([Start_pos[1], Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_box, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w11
        w11 = ([Start_pos[1] + d_vegger, Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w12
        w12 = ([Start_pos[1] + d_vegger, Start_pos[1] + d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w13
        w13 = ([Start_pos[1] + d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_box/2, Start_pos[2] + d_absorbent_QRD + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w14
        w14 = ([Start_pos[1] + d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w15
        w15 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_absorbent_QRD + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w16
        w16 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w17
        w17 = ([Start_pos[1] + b_box, Start_pos[1] + b_box], 
        [Start_pos[2] + d_box/2, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w18
        w18 = ([Start_pos[1] + b_box, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_box, Start_pos[2] + d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w19
        w19 = ([Start_pos[1] + b_box - d_vegger, Start_pos[1] + b_box - d_vegger], 
        [Start_pos[2] + d_box, Start_pos[2] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w20
        w20 = ([Start_pos[1], Start_pos[1]+b_box],
        [Start_pos[2], Start_pos[2]+d_box],
        [Start_pos[3]+h_box, Start_pos[3]+h_box])
        next!(prog_draw_box)
        # w21
        w21 = ([Start_pos[1], Start_pos[1]+b_box],
        [Start_pos[2], Start_pos[2]+d_box],
        [Start_pos[3], Start_pos[3]])
        ################################################

    elseif wall == 4
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]] - b_box,
        [Start_pos[2], Start_pos[2]],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w2
        w2 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2], Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w3
        w3 = ([Start_pos[1], Start_pos[1]] - d_vegger, 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w4
        w4 = ([Start_pos[1] - d_vegger, Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w5
        w5 = ([Start_pos[1] - d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_bakplate, Start_pos[2] - d_bakplate], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w6
        w6 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_bakplate, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w7
        w7 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w8
        w8 = ([Start_pos[1] - b_box, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w9
        w9 = ([Start_pos[1], Start_pos[1]], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w10
        w10 = ([Start_pos[1], Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_box, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w11
        w11 = ([Start_pos[1] - d_vegger, Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w12
        w12 = ([Start_pos[1] - d_vegger, Start_pos[1] - d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w13
        w13 = ([Start_pos[1] - d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_box/2, Start_pos[2] - d_absorbent_QRD - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w14
        w14 = ([Start_pos[1] - d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2, Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w15
        w15 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_absorbent_QRD - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w16
        w16 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w17
        w17 = ([Start_pos[1] - b_box, Start_pos[1] - b_box], 
        [Start_pos[2] - d_box/2, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w18
        w18 = ([Start_pos[1] - b_box, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_box, Start_pos[2] - d_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w19
        w19 = ([Start_pos[1] - b_box + d_vegger, Start_pos[1] - b_box + d_vegger], 
        [Start_pos[2] - d_box, Start_pos[2] - d_absorbent_QRD - d_skilleplate - d_box/2], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        # w20
        w20 = ([Start_pos[1], Start_pos[1]-b_box],
        [Start_pos[2], Start_pos[2]-d_box],
        [Start_pos[3]+h_box, Start_pos[3]+h_box])
        next!(prog_draw_box)
        # w21
        w21 = ([Start_pos[1], Start_pos[1]-b_box],
        [Start_pos[2], Start_pos[2]-d_box],
        [Start_pos[3], Start_pos[3]])
        ################################################
    elseif wall == 5
        ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]],
        [Start_pos[2] - b_box, Start_pos[2]],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w2
        w2 = ([Start_pos[1], Start_pos[1] + d_box], 
        [Start_pos[2], Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w3
        w3 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - d_vegger, Start_pos[2]], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w7
        w4 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w8
        w5 = ([Start_pos[1], Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        #############################################

        ################## QRD Box ##################
       
       
        # w11
        w6 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w14
        w7 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2 , Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w19
        w8 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger], 
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        


        #XXXXXXXXXXX QRD Del
        # W22
        w21 = ([Start_pos[1]+ d_box - 0.048, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- b_QRD - d_vegger, Start_pos[2]-d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        

        # w23
        w11 = ([Start_pos[1]+ d_box/2, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- b_QRD - d_vegger, Start_pos[2] - b_QRD - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w24
        w12 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1]+d_box - 0.024],
        [Start_pos[2]- b_QRD*3 - d_vegger, Start_pos[2] - b_QRD*2 - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w25
        w13 = ([Start_pos[1]+ d_box/2, Start_pos[1]+d_box - 0.024],
        [Start_pos[2]- b_QRD*2 - d_vegger, Start_pos[2] - b_QRD*2 - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w26
        w14 = ([Start_pos[1]+ d_box, Start_pos[1] + d_box],
        [Start_pos[2]- 4*b_QRD - d_vegger, Start_pos[2] - 3*b_QRD - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w27
        w15 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1] + d_box],
        [Start_pos[2]- 3*b_QRD - d_vegger, Start_pos[2] - 3*b_QRD - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w27
        w16 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1] + d_box],
        [Start_pos[2]- 4*b_QRD - d_vegger, Start_pos[2] - 4*b_QRD - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w28
        w17 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1]+d_box - 0.024],
        [Start_pos[2]- b_QRD*5 - d_vegger, Start_pos[2] - b_QRD*4 - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w29
        w18 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box - 0.024],
        [Start_pos[2]- b_QRD*5 - d_vegger, Start_pos[2] - b_QRD*5 - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w30
        w19 = ([Start_pos[1]+ d_box - 0.048, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- 7*b_QRD - d_vegger, Start_pos[2]- 6*b_QRD - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        
        # w13

        w20 = ([Start_pos[1]+ d_box/2, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- 6*b_QRD - d_vegger, Start_pos[2] - 6*b_QRD - d_vegger],
        [Start_pos[3], Start_pos[3] + h_box])
        next!(prog_draw_box)
        

        # w20
        w9 = ([Start_pos[1], Start_pos[1]+d_box],
        [Start_pos[2]-b_box, Start_pos[2]],
        [Start_pos[3]+h_box, Start_pos[3]+h_box])
        next!(prog_draw_box)
        
        # w21
        w10 = ([Start_pos[1], Start_pos[1]+d_box],
        [Start_pos[2]- b_box, Start_pos[2]],
        [Start_pos[3], Start_pos[3]])
        
        #################################################




        
    end
    #return [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10]
    return [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21]
end







function draw_box_closed_2D(Start_pos::Array{Float64,1}, d_box::Float64, b_box::Float64, N_QRD::Int64, b_QRD::Float64, d_max_QRD::Float64, d_vegger::Float64, d_bakplate::Float64, d_skilleplate::Float64, d_absorbent_QRD::Float64)


    prog_draw_box = Progress(21)
    ################## Lower Box ################
        # w1
        w1 = ([Start_pos[1], Start_pos[1]],
        [Start_pos[2] - b_box, Start_pos[2]])
        next!(prog_draw_box)
        
        # w2
        w2 = ([Start_pos[1], Start_pos[1] + d_box], 
        [Start_pos[2], Start_pos[2]])
        next!(prog_draw_box)
        
        # w3
        w3 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - d_vegger, Start_pos[2]])
        next!(prog_draw_box)
        
        # w7
        w4 = ([Start_pos[1] + d_box, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box + d_vegger])
        next!(prog_draw_box)
        
        # w8
        w5 = ([Start_pos[1], Start_pos[1] + d_box], 
        [Start_pos[2] - b_box, Start_pos[2] - b_box])
        next!(prog_draw_box)
        
        #############################################

        ################## QRD Box ##################
       
       
        # w11
        w6 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - d_vegger, Start_pos[2] - d_vegger])
        next!(prog_draw_box)
        
        # w14
        w7 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2 , Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - d_vegger])
        next!(prog_draw_box)
        
        # w19
        w8 = ([Start_pos[1] + d_absorbent_QRD + d_skilleplate + d_box/2, Start_pos[1] + d_box], 
        [Start_pos[2] - b_box + d_vegger, Start_pos[2] - b_box + d_vegger])
        next!(prog_draw_box)
        


        #XXXXXXXXXXX QRD Del
        # W22
        w21 = ([Start_pos[1]+ d_box - 0.048, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- b_QRD - d_vegger, Start_pos[2]-d_vegger])
        next!(prog_draw_box)
        

        # w23
        w11 = ([Start_pos[1]+ d_box/2, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- b_QRD - d_vegger, Start_pos[2] - b_QRD - d_vegger])
        next!(prog_draw_box)
        
        # w24
        w12 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1]+d_box - 0.024],
        [Start_pos[2]- b_QRD*3 - d_vegger, Start_pos[2] - b_QRD*2 - d_vegger])
        next!(prog_draw_box)
        
        # w25
        w13 = ([Start_pos[1]+ d_box/2, Start_pos[1]+d_box - 0.024],
        [Start_pos[2]- b_QRD*2 - d_vegger, Start_pos[2] - b_QRD*2 - d_vegger])
        next!(prog_draw_box)
        
        # w26
        w14 = ([Start_pos[1]+ d_box, Start_pos[1] + d_box],
        [Start_pos[2]- 4*b_QRD - d_vegger, Start_pos[2] - 3*b_QRD - d_vegger])
        next!(prog_draw_box)
        
        # w27
        w15 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1] + d_box],
        [Start_pos[2]- 3*b_QRD - d_vegger, Start_pos[2] - 3*b_QRD - d_vegger])
        next!(prog_draw_box)
        
        # w27
        w16 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1] + d_box],
        [Start_pos[2]- 4*b_QRD - d_vegger, Start_pos[2] - 4*b_QRD - d_vegger])
        next!(prog_draw_box)
        
        # w28
        w17 = ([Start_pos[1]+ d_box - 0.024, Start_pos[1]+d_box - 0.024],
        [Start_pos[2]- b_QRD*5 - d_vegger, Start_pos[2] - b_QRD*4 - d_vegger])
        next!(prog_draw_box)
        
        # w29
        w18 = ([Start_pos[1] + d_box/2, Start_pos[1] + d_box - 0.024],
        [Start_pos[2]- b_QRD*5 - d_vegger, Start_pos[2] - b_QRD*5 - d_vegger])
        next!(prog_draw_box)
        
        # w30
        w19 = ([Start_pos[1]+ d_box - 0.048, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- 7*b_QRD - d_vegger, Start_pos[2]- 6*b_QRD - d_vegger])
        next!(prog_draw_box)
        
        # w13

        w20 = ([Start_pos[1]+ d_box/2, Start_pos[1]+d_box - 0.048],
        [Start_pos[2]- 6*b_QRD - d_vegger, Start_pos[2] - 6*b_QRD - d_vegger])
        next!(prog_draw_box)
        

        # w20
        w9 = ([Start_pos[1], Start_pos[1]+d_box],
        [Start_pos[2]-b_box, Start_pos[2]])
        next!(prog_draw_box)
        
        # w21
        w10 = ([Start_pos[1], Start_pos[1]+d_box],
        [Start_pos[2]- b_box, Start_pos[2]])
        
        ################################################
        
    return [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21]
end
