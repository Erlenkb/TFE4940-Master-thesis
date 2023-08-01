



"""
    _heatmap_gif_single_heatmap(p, N, time, Δd, title)

computes and plot the heatmap for the given pressure array. save the animation as a gif. Takes in the ´N´ number of samples to be treadet, same as size(p,3),  a time vector, step value Δd and a title string to be used for the plot
"""




function animate_heatmap_slices(data, filename)
    N = size(data, 3)
    stepsize = 1

    anim = @gif for i in 1:stepsize:N
        heatmap(data[:, :, i], c=:grays, clim=((-40,-6)), title="Slice $i")
    end

    # Save the animation as a file
    gif(anim, filename, fps=15)
end






















function heatmap_gif2(p, p2, p3, N, time, D)
    # Convert tuples to arrays

    # Set up heatmap ticks
    nz = size(p[1],3)
    nx = size(p[1], 2)
    ny = size(p[1], 1)


    println("nx: ",nx,"\tny: ",ny, "\tnz: ",nz)
    step_x = nx ÷ 4
    step_y = ny ÷ 4
    xtick_values = collect(1:step_x:(nx*D))
    ytick_values = collect(1:step_y:(ny*D))

    # Set up plot layout
    layout = @layout [a b c; d{0.1h}]
    
    # Set up heatmap color limits
    lim_pos = 10*log10((maximum(p[3]) / 9)^2)
    lim_neg = 10*log10(0.000005)
    
    # Create initial plot with three heatmaps
    plot1 = plot(
        heatmap(p[1], xticks=xtick_values, yticks=ytick_values, title="x", c=cgrad(:viridis, rev=true), clims=(lim_neg,lim_pos)),
        heatmap(p2[1], xticks=xtick_values, yticks=ytick_values, title="y", c=cgrad(:viridis, rev=true), clims=(lim_neg,lim_pos)),
        heatmap(p3[1], xticks=xtick_values, yticks=ytick_values, title="z", c=cgrad(:viridis, rev=true), clims=(lim_neg,lim_pos)),
        layout=layout,
        size=(900, 600),
    )

    # Set up loading bar plot
    plot2 = plot(
        0, [0], xlims=(minimum(time), maximum(time)), ylims=(0,1),yticks=[],ylabel="Progress", xlabel="Time [s]"
    )
    plot2[:yaxis][:tickfont] = font(6)
    plot2[:yaxis][:ticksize] = 0
    plot2[:margin] = 5Plots.px
    plot2[:foreground_color] = :black
    plot2[:background_color] = :white
    plot2[:size] = (1, 0.1)

    # Combine plots
    plot1 = plot(plot1, plot2)

    # Create animation
    anim = @animate for i=2:div(N, 10):N
        plot1[1][1][:z] = p[i]
        plot1[1][2][:z] = p2[i]
        plot1[1][3][:z] = p3[i]
        t = time[i]
        x = [t,t]
        y = [0,1]
        plot!(plot2, x, y, color="blue", legend=false)
        plot1
    end

    # Create gif
    gif(anim, "heatmap.gif", fps=10)
end









function _heatmap_gif1(p1, p2, p3, N, time, Δd)
    p = make_subplots(
        rows=2, cols=3,
        specs=[Spec() Spec() Spec(); Spec(colspan=2) missing missing],
        subplot_titles=["X plane" "Y plane" "Z plane"; "Time [s]" missing missing]
    )

    println("Initializing Gif creation")
    
    # Get the size of the pressure data arrays and the aspect ratio
    nx, ny = size(p1[1])
    aspect_ratio = ny / nx
    
    # Define the tick values for the x and y axes
    step_x = nx / 4
    step_y = ny / 4
    xtick_values = collect(1:step_x:(nx*Δd))
    ytick_values = collect(1:step_y:(ny*Δd))
    
    # Set the plot limits and colormap
    lim_pos = 10*log10((A/9)^2)
    lim_neg = 10*log10(0.000005)
    cmap = :viridis

    plot2 = plot(
        0, [0], xlims=(minimum(time), maximum(time)), ylims=(0,1),yticks=[],ylabel="Progress", xlabel="Time [s]"
    )
    """
    plot2[:yaxis][:tickfont] = font(6)
    plot2[:yaxis][:ticksize] = 0
    plot2[:margin] = 5Plots.px
    plot2[:foreground_color] = :black
    plot2[:background_color] = :white
    plot2[:size] = (1, 0.1)
    """
    anim = @animate for i=2:7:N
        add_trace!(p, heatmap(p1[1], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values, yticks=ytick_values), row=1, col=1)
        add_trace!(p, heatmap(p2[1], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values, yticks=ytick_values), row=1, col=2)
        add_trace!(p, heatmap(p3[1], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values, yticks=ytick_values), row=1, col=3)
        
        t = time[i]
        x = [t,t]
        y = [0,1]

        add_trace!(p, plot(plot2, x, y, color="blue", legend=false), row=2, col=1)
        relayout!(p, showlegend=false, title_text="Specs with Subplot Title")
        p
    end
    
    # Save the animation as a gif
    println("Creating Gif")
    gif(anim, "heatmaps.gif", fps=10)
    println("Gif created")


end







function _heatmap_gif(p, N, time, Δd)

    println("Initializing Gif creation")
    
    # Get the size of the pressure data arrays and the aspect ratio
    nx, ny = size(p[1])
    aspect_ratio = ny / nx
    
    # Define the tick values for the x and y axes
    step_x = nx / 4
    step_y = ny / 4
    xtick_values = collect(1:step_x:(nx*Δd))
    ytick_values = collect(1:step_y:(ny*Δd))
    
    # Set the plot limits and colormap
    lim_pos = 10*log10((2*A)^2)
    lim_neg = 10*log10(0.000005)
    cmap = :viridis
    
    # Define the layout of the subplots
    layout = @layout [a b c; d{0.08h} e{0.08h} f{0.08h}]
    
    # Create the subplots with the heatmaps and titles
    plot1 = plot(
        heatmap(p[1], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values, yticks=ytick_values),
        PlotyJS.subplot(1,1,2)
    )
    
    # Customize the second row of the subplots with the time graph
    plot2 = plot(
        0, [0], xlims=(minimum(time), maximum(time)), ylims=(0,1),yticks=[],ylabel="Progress", xlabel="Time [s]",
        PlotlyJS.subplot(2,1,2:3), size=(1200, 50)
    )

    plot2[:yaxis][:tickfont] = font(6)
    plot2[:yaxis][:ticksize] = 0
    plot2[:margin] = 5Plots.px
    plot2[:foreground_color] = :black
    plot2[:background_color] = :white
    plot2[:size] = (1, 0.1)
    
    # Create the animation
    anim = @animate for i=2:7:N
        heatmap!(p[i], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values,yticks=ytick_values)
        heatmap!(p2[i], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values,yticks=ytick_values)
        heatmap!(p3[i], color=cmap, clims=(lim_neg,lim_pos), xticks=xtick_values,yticks=ytick_values)
        t = time[i]
        x = [t,t]
        y = [0,1]
        plot!(plot2, x, y, color="blue", legend=false)
        plot1
    end
    
    # Save the animation as a gif
    println("Creating Gif")
    gif(anim, "heatmaps.gif", fps=25)
    println("Gif created")
end

