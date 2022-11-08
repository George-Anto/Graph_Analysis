import networkx as nx
import matplotlib.pyplot as plt
import pandas
import numpy as np
from matplotlib.ticker import PercentFormatter

#Create a total dataframe from the stackoverflow file
#Read only part of the file
total_df = pandas.read_csv("sx-stackoverflow.txt", sep = " ", header = None, nrows = 100000)
total_df.columns = ["source", "target", "timestamp"]

#print(total_df.head())
#print(total_df.loc[0:4, :])
print(total_df)

print("---------------------------------")

#Get the minimum timestamp of the total dataframe, it is the timestamp of the first row
min_time = total_df.iloc[0, 2]
#Get the maximum timestamp of the total dataframe, it is the timestamp of the last row
max_time = total_df.iloc[-1, 2]
#Calculate the difference between those two
timespan = int(max_time - min_time)

#Get the N from the user
#Create a loop to ensure tha the user will give a valid input (pisitive integer number)
while True:
    N = input("\n Give the N (it must be a positive integer): ")
    try:
        #If the input is not an integer, the code in the except block will be executed
        #and the user will be prompted to enter a new value 
        #If the number is positive but not integer the except block will be executed too
        N = int(N)
        #If the number is negative or zero, the user will have to enter a new value too
        if N <= 0:  # if not a positive int print message and ask for input again
            print("Sorry, input must be a positive integer, try again.")
            continue
        break
    except ValueError:
        print("That's not an integer, try again.") 

#Calculate the time that belongs to each subgraph
time_for_each_graph = timespan / N

print("Min: ", min_time, " | Max: ", max_time)
print("Time Span: ", timespan)
print("Time for each graph: ", time_for_each_graph)

#Arrays for the number of nodes and edges for each time period
number_of_nodes_in_each_time_period = [];
number_of_edges_in_each_time_period = [];

#       --- 1 ---
#We loop the total_df for N times, to create all the subgraphs the user requested
for i in range(1, N + 1):
    print("\n---------- Graph ", i, "----------\n")
    #If we are not in the last subgraph, we create the subgraph with an open set from the right [first_edge, last_edge),
    #so the last element (edge) of the subgraph does not overlapse with the first of the next subgraph
    if i != N:
        current_df = total_df.loc[(total_df['timestamp'] >= time_for_each_graph * (i - 1) + min_time) & (total_df['timestamp'] < time_for_each_graph * i + min_time)]
        #If we are on the last subgraph we include an '=' to the comparison, so the last element will be definitely included to the subgraph 
    else:
        current_df = total_df.loc[(total_df['timestamp'] >= time_for_each_graph * (i - 1) + min_time) & (total_df['timestamp'] <= time_for_each_graph * i + min_time)]
    print(current_df)

    #Create the current subgraph using the current dataframe
    current_G = nx.from_pandas_edgelist(current_df, source = 'source', target = 'target', create_using = nx.Graph(), edge_attr = "timestamp")
    print("\n", current_G)
    
    #Test for the histograms of the degrees of each subgraph 
    #plt.hist([v for k,v in nx.degree(current_G)])
    #plt.show()
    
    #       --- 2 ---
    print("\n-------- Adjacency Matrix ", i, "--------\n")
    current_ad_matrix = nx.to_pandas_adjacency(current_G)
    print(current_ad_matrix)
    
    # --- 3 ---
    #Push into the corresponding array the number of nodes and adges of the current subgraph
    number_of_nodes_in_each_time_period.append(current_G.number_of_nodes());
    number_of_edges_in_each_time_period.append(current_G.number_of_edges());
    
    # --- 4 ---
    #Plot the histograms only for the 1st subgraph
    if i == 1:
        #Styling for the plots
        centrality_font = {'family':'serif', 'size':15}
        
        #Degree Centrality Histogram
        current_degree_cen = nx.degree_centrality(current_G)
        plt.hist(list(current_degree_cen.values()), weights=np.ones(len(current_degree_cen.values())) / len(current_degree_cen.values()), color = "lightcoral", ec="red", lw=2)
        plt.xlabel("Normalized Degree Centrality", fontdict = centrality_font)
        plt.ylabel("Percentage (%) of Nodes", fontdict = centrality_font)
        plt.title("Relative Degree Centrality Distribution of Subgraph " + str(i), fontdict = centrality_font)
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        
        #Closeness Centrality Histogram
        current_closeness_cen = nx.closeness_centrality(current_G)
        plt.hist(list(current_closeness_cen.values()), color = "skyblue", weights=np.ones(len(current_closeness_cen.values())) / len(current_closeness_cen.values()), ec="blue", lw=2)
        plt.xlabel("Normalized Closeness Centrality", fontdict = centrality_font)
        plt.ylabel("Percentage (%) of Nodes", fontdict = centrality_font)
        plt.title("Relative Closeness Centrality Distribution of Subgraph " + str(i), fontdict = centrality_font)
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        
        #Betweenness Centrality Histogram
        current_betweenness_cen = nx.betweenness_centrality(current_G)
        plt.hist(list(current_betweenness_cen.values()), color = "khaki", weights=np.ones(len(current_betweenness_cen.values())) / len(current_betweenness_cen.values()), ec="gold", lw=2)
        plt.xlabel("Normalized Betweenness Centrality", fontdict = centrality_font)
        plt.ylabel("Percentage (%) of Nodes", fontdict = centrality_font)
        plt.title("Relative Betweenness Centrality Distribution of Subgraph " + str(i), fontdict = centrality_font)
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        
        #Eigenvector Centrality Histogram
        current_eigenvector_cen = nx.eigenvector_centrality(current_G)
        plt.hist(list(current_eigenvector_cen.values()), weights=np.ones(len(current_eigenvector_cen.values())) / len(current_eigenvector_cen.values()), color = "greenyellow", ec="forestgreen", lw=2)
        plt.xlabel("Normalized Eigenvector Centrality", fontdict = centrality_font)
        plt.ylabel("Percentage (%) of Nodes", fontdict = centrality_font)
        plt.title("Relative Eigenvector Centrality Distribution of Subgraph " + str(i), fontdict = centrality_font)
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        
        #Katz Centrality Histogram
        current_katz_cen = nx.katz_centrality_numpy(current_G)
        plt.hist(list(current_katz_cen.values()), weights=np.ones(len(current_katz_cen.values())) / len(current_katz_cen.values()), color = "plum", ec="purple", lw=2)
        plt.xlabel("Normalized Katz Centrality", fontdict = centrality_font)
        plt.ylabel("Percentage (%) of Nodes", fontdict = centrality_font)
        plt.title("Relative Katz Centrality Distribution of Subgraph " + str(i), fontdict = centrality_font)
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()
        
    
print("Nodes: ", number_of_nodes_in_each_time_period)
print("Edges: ", number_of_edges_in_each_time_period) 

#Styling for the plots
nodes_font = {'family':'serif','color':'darkred','size':15}
edges_font = {'family':'serif','color':'midnightblue','size':15} 

#Create a plot to show the nodes each subgraph contains and the evolution of their number
plt.plot(number_of_nodes_in_each_time_period, color='red', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt.title("Nodes Evolution for the total " + str(N) + " time periods", fontdict = nodes_font)
plt.xlabel("Time Period (N = " + str(N) + ")", fontdict = nodes_font)
plt.ylabel("Total Number of Nodes", fontdict = nodes_font)
plt.grid(True)
plt.show()   

#Create a plot to show the edges each subgraph contains and the evolution of their number
plt.plot(number_of_edges_in_each_time_period, color='blue', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt.title("Edges Evolution for the total " + str(N) + " time periods", fontdict = edges_font)
plt.xlabel("Time Period (N = " + str(N) + ")", fontdict = edges_font)
plt.ylabel("Total Number of Edges", fontdict = edges_font)
plt.grid(True)
plt.show() 