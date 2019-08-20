usage: python3 gen_synthetic_data.py [operation] [number of times to do operation] [number of fake graphs to generate] 

Outputs: 

In OUTPUT_DIR = FakeGenPlots/ we draw the the original graph and save it as FakeGenPlots/[original graph id]_original.png.
We then draw each of the generated fake graphs and save it as FakeGenPlots/[original graph id]_[fake graph number][operation done on fake graph]_fake.png.
We finally draw the MCS for each fake graph ans save it as FakeGenPlots/[original graph id]_[fake graph number][operation done on fake graph]_mcs.png

Then for each original graph we print to console the original graph's nodes and edges and each corresponding fake graph's nodes and edges along with the MCS and the mapping dictionary.

Finally, we print the total number of fake graphs generated.  


In the top directory there is a Data folder that stores the initial dataset, a save folder that will store the train.klepto and test.klepto parts of the dataset and a Test folder that contains the source code. In the Test folder there is also a FakeGenPlots folder that will save the png images of the original graph, the fake graph and the MCS. 
