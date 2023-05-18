using PlotlyJS

p = make_subplots(
    rows=2, cols=2,
    specs=[Spec() Spec(); Spec(colspan=2) missing],
    subplot_titles=["First Subplot" "Second Subplot"; "Third Subplot" missing]
)

add_trace!(p, PlotlyJS.scatter(x=[1, 2], y=[1, 2]), row=1, col=1)
add_trace!(p, PlotlyJS.scatter(x=[1, 2], y=[1, 2]), row=1, col=2)
add_trace!(p, PlotlyJS.scatter(x=[1, 2, 3], y=[2, 1, 2]), row=2, col=1)

relayout!(p, showlegend=false, title_text="Specs with Subplot Title")
p