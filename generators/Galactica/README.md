# Galactica for Generation
## Files and Folders
- `archive and experiments` contains notebooks in which we experimented with Galactica. 
- `output` contains output generations. files inside this folder are named as such `galactica_${generation_start_index}_${generation_end_index}.csv` 
- `Galactica_0_500.ipynb` is an example notebook in which 500 Galactica papers are generated. For the notebook to work properly, `galactica.csv` must exist with the `title` which is used to generate the other sections. 
- `combine_output.ipynb` concatinates all the generated outputs and cleans them.