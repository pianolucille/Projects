# Projects

The purpose of this project was simply to experiment with different optimization methods (hill climbing, iterated local search, and a genetic algorithm). The optimizer takes the colors (in the colors.txt file), which are lists of three-dimensional coordinates, and performs three different optimizations. In this case, the optimal solution is when the total distance between all the colors is the least. Since each color is associated with a vector, distance is a simple calculation. Each run of the embedded functions essentially sorts the colors into rainbow order. For instance, given three colors red, blue, and green, the optimizer would find that the distance between blue and red is greater than blue and green. So, green and blue would be placed next to each other and so on.

To run the project, first make sure that the .py file is in the same directory as the color.txt file. Also, matplotpib should be installed on your device. Run the .py file via python in a terminal (or in whichever interface you prefer to run python in). It will prompt you accordingly.
