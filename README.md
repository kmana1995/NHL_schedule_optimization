# NHL_schedule_optimization
A linear program to optimize any NHL teams travel schedule schedule for distance traveled.<br><br>


## How to use the program
This project easily allows the analyst to optimize the 2019-2020 schedule for any NHL team. Just by following these steps:<br><br>

<b> Initialize the class with a string passed indicating the full name of a team</b> optimizer = NHLScheduleOptimizer('Arizona Coyotes')<br><br>

<b> Run the optimizer by running </b> optimizer.optimize_schedule(), <b> the output will be located in the current directory or root path by default </b><br><br>

An example of the run can be found at the bottom of the main .py file, under the if __name__ == '__main__' statement.

A paper discussing the method is titled NHL_Schedule_Optimization.pdf
