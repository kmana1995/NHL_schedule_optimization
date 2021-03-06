# NHL_schedule_optimization
A linear program to optimize any NHL teams travel schedule schedule for distance traveled.<br><br>

## Results
The model has shown appreciable results over the actual travel schedule for a majority of NHL teams. Consider the below example using the Vancouver Canucks, where the optimized schedule resulted in a <b> 15% reduction in travel time, or 7,109 miles</b>. <br>
![alt text](https://github.com/kmana1995/NHL_schedule_optimization/blob/master/vancouver_canucks_opt_results.png?raw=true)<br><br>
## How to use the program
This project easily allows the analyst to optimize the 2019-2020 schedule for any NHL team. Just by following these steps:<br><br>

<b> Initialize the class with a string passed indicating the full name of a team</b> optimizer = NHLScheduleOptimizer('Arizona Coyotes')<br><br>

<b> Run the optimizer by running </b> optimizer.optimize_schedule(), <b> the output will be located in the current directory or root path by default </b><br><br>

An example of the run can be found at the bottom of the main .py file, under the if __name__ == '__main__' statement.

## Read the paper

A paper discussing the method can be found in the repository or at https://github.com/kmana1995/NHL_schedule_optimization/blob/master/NHL_Schedule_Optimization.pdf.
