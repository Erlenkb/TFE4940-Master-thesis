# TFE4940-Master-thesis
 The following repository will contain articles and relevant code for the master thesis done by Erlend K. Berg

All files are located in the ´.venv´ folder.
The files used for the TLM operations are: 

* TLM matrix.jl -- The main file used for the TLM.
* TLM creation.jl -- Contain functions used for creating the TLM grid and apply the size of the shoebox shaped room.
* TLM calculation.jl -- Functions used for the general propagation rules
* TLM operations.jl -- Functions used to handle certain aspects of the TLM matrix
* Acoustical operations.jl -- Functions used for handling the wave signals and acoustical parameters
* Draw_box_closed.jl -- create all the necessary walls for the box of interest in the closed stage.
* Plot_heatmap.jl -- Used for creating animated gifs for the propagation
* Place objects into TLM.jl -- Contain function that update the labeled TLM with a plane/wall











19.02.2023
Added Julia code creating a TLM TLM matrix labeling each element with respect of its positioning relative to fluid, surfaces, edges and corners.

19.05.2023
Added structure to the folder and split the julia code into several files.