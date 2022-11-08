import networkx as nx
import matplotlib.pyplot as plt
import pandas

#Function that writes a dataframe to a file
#It can also delete the "source" and "target" columns from
#the dataframe before saving it to the file if the user chooses so
def write_dataframe_to_file(df, file_name = "data", remove_nodes_and_edges_columns = False):
    if remove_nodes_and_edges_columns:
        df.drop(['source', 'target'], axis = 1, inplace = True)
    df.to_csv("data/" + file_name, sep = " ", index = False, na_rep = 0)

#Helper function that calls the function that creates the dataframes files
#in a specific way, to store both the metrics and the existing edges of a graph 
def create_dataframes_and_files(metrics_list, metrics_data_file = "metrics_data.csv", existing_edges_file = "existing_edges.csv", remove_nodes_and_edges_columns = False):
    folder_name = "10000_rows_N10"
    # folder_name = "5000_rows_N10"
    # folder_name = "1000_rows_N5"
    
    metrics_df = metrics_list[5]
    existing_edges_df = metrics_list[6]
    write_dataframe_to_file(metrics_df, folder_name + "/" + metrics_data_file, remove_nodes_and_edges_columns)
    write_dataframe_to_file(existing_edges_df, folder_name + "/" + existing_edges_file, remove_nodes_and_edges_columns)    
    
#Function that creates the temp graph that is described below
#It also creates a dataframe that contains the information about what edges exist
#in the graph and what edges are missing  
def create_temp_graph_and_existing_edges(G):
    #Create a temp graph with all posible edges to pass as a parameter to the ebunch of the following functions 
    #in order to calculate all the possible pair of nodes 
    #The temp graph is also usefull to create a dataframe in another function 
    #that will store the metrics for all the possible adges of the current graph
    temp_G = G.copy()
    nx.set_edge_attributes(temp_G, 1, "does edge exist")
    
    temp_G.add_edges_from(nx.non_edges(G))
    
    #Initialize a list
    does_edge_exist_list = []
    
    for u, v, attr in temp_G.edges(data = True):
        #If this attribute is '1', the edge exist
        if attr.get('does edge exist') == 1:
            does_edge_exist_list.append(1)
            continue
        #If not, it doesn't 
        does_edge_exist_list.append(0)
    
    #Return the graph with all the edges and the list with the existing edges
    return [temp_G, does_edge_exist_list]

#Calculate the graph distance and the common neighbors from all the nodes 
#to all the other nodes  
def graph_dist_and_common_neighbors_calc(G):

    #Create the list of nodes
    nodes = list(G.nodes)
    #Create 2 empty dictionaries 
    graph_distance = {}
    common_neighbors = {}

    #For each pair of nodes
    for first_node in nodes:
        for second_node in nodes:
            
            #Find the common neighbors of the curent 2 nodes
            neighbors = []
            temp_neighbors = nx.common_neighbors(G, first_node, second_node)

            for temp in temp_neighbors:
                neighbors.append(temp)

            #Append the length og the current list as value to the dictionary
            #The key is the pait of nodes
            common_neighbors[first_node, second_node] = len(neighbors)

            #Find the graph distance of the curent 2 nodes
            try:
                distance = nx.shortest_path_length(G, first_node, second_node)
                #We do the same as the common neighbors but we store the distance
                #as a negative number, according to our assignment's requests 
                graph_distance[first_node, second_node] = -distance
            except:
                #If the distance of the 2 nodes can not be calculated,
                #the nodes are not connected to each other
                #so we assign the number of the total nodes to that dictionary position
                graph_distance[first_node, second_node] = - len(nodes)

    #Return the calculated dictionaries 
    return [graph_distance, common_neighbors]

#A long function that calculates the metrics that are requested for the 2nd part of our assignment
#It can also print the metrics to the console, helpful for small graphs and for debugging
def all_metrics_calc(G, graph_name, i, debug_metrics = False):
    
    #Call the function that calculates the graph distance for each node in the current graph
    #and also for the common neighbors
    graph_dist_and_common_neighbors = graph_dist_and_common_neighbors_calc(G)
    graph_distance = graph_dist_and_common_neighbors[0]
    
    #Calculate the common neighbors array for the Graph 
    # 1st Approach
    # common_neighbors = nx.to_scipy_sparse_array(G)
    # common_neighbors = (common_neighbors ** 2).todok()
    
    # 2nd Approach
    common_neighbors = graph_dist_and_common_neighbors[1]
    
    #Call the method to create the temp graph and get the list that contains the existing nodes
    temp_G_and_existing_edges_list = create_temp_graph_and_existing_edges(G)
    temp_G = temp_G_and_existing_edges_list[0]
    existing_edges_list = temp_G_and_existing_edges_list[1]
    
    #Create a dataframe from that list and the temp graph
    existing_edges_df = pandas.DataFrame(temp_G.edges, columns=['source', 'target'])
    existing_edges_df["does edge exist"] = existing_edges_list
    
    #Calculate the Jaccard Coefficient list 
    jaccard_coef = list(nx.jaccard_coefficient(G, ebunch = temp_G.edges))
    
    #Calculate the Adamic - Adar list 
    adamic_adar = list(nx.adamic_adar_index(G, ebunch = temp_G.edges))
    
    #Calculate the Preferential Attachment list 
    pref_attachment = list(nx.preferential_attachment(G, ebunch = temp_G.edges))
    
    #Create a dataframe for the metrics, for all the possible edges 
    metrics_df = pandas.DataFrame(temp_G.edges, columns=['source', 'target'])
    
    #Initialize the lists that will hold only the useful part of the data 
    #calculated by the corresponding metrics functions 
    graph_distance_list = []
    common_neighbors_list = []
    jaccard_coef_list = []
    adamic_adar_list = []
    pref_attachment_list = []
        
    #Create a for loop for each pair of nodes in the temp graph
    for i in range(len(jaccard_coef)):
        
        #Get the source and target data for each iteration (pair of nodes)
        source = jaccard_coef[i][0]
        target = jaccard_coef[i][1]
        
        #Get the 3 metrics for each pair of nodes
        jaccard_coef_list.append(jaccard_coef[i][2])
        adamic_adar_list.append(adamic_adar[i][2])
        pref_attachment_list.append(pref_attachment[i][2])
        
        #For our custmom calculations, the data are not in the same order as in 
        #the networkx functions (e.g. nx.jaccard_coefficient), so we must loop
        #through our data and compare the adge from that functions and the adge
        #of the current iteration to match them and then store the corrent data
        for x, y in graph_distance.items():
            if x == (source, target):
                graph_distance_list.append(y)
                break
        
        #The same and here     
        for x, y in common_neighbors.items():
            if x == (source, target):
                common_neighbors_list.append(y)
                break            
        
    #Create new columns to the metrics dataframe with the metrics lists         
    metrics_df["graph distance"] = graph_distance_list
    metrics_df["common neighbors"] = common_neighbors_list
    metrics_df["jaccard coef"] = jaccard_coef_list
    metrics_df["adamic - adar"] = adamic_adar_list
    metrics_df["pref attachment"] = pref_attachment_list
    
    #Print the above results if this parameter is true
    if(debug_metrics):
        print("\n---------- Round", i, "-", graph_name, "----------")
        print("\n---------- Graph Distance for the", graph_name, "----------")
        print(graph_distance)
        print("\n----------------------------")
        
        print("\n---------- Common Neighbors for the", graph_name, "----------")
        print(common_neighbors)
        print("\n----------------------------")
        
        print("\n---------- Jaccard Coefficient -", graph_name, "Edges ----------")
        print(jaccard_coef)
        print("\nLength: ", len(jaccard_coef))
        print("\n----------------------------")
        
        print("\n---------- Adamic Adar -", graph_name, "Edges ----------")
        print(adamic_adar)
        print("\nLength: ", len(adamic_adar))
        print("\n----------------------------")
        
        print("\n---------- Preferential Attachment -", graph_name, "Edges ----------")
        print(pref_attachment)   
        print("\nLength: ", len(pref_attachment))
        print("\n----------------------------")
        
        print("\n---------- Metrics DataFrame -", graph_name, "----------")
        print(metrics_df)
        
    #Return the calculated metrics and the dataframes of the metrics and the existing edges    
    return [graph_distance, common_neighbors, jaccard_coef, adamic_adar, pref_attachment, metrics_df, existing_edges_df]      

#Create a total dataframe from the stackoverflow file
#Read only part of the file
total_df = pandas.read_csv("sx-stackoverflow.txt", sep = " ", header = None, nrows= 10000)
total_df.columns = ["source", "target", "timestamp"]

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

#Initialize the lists 
number_of_nodes_in_each_paired_time_period = []
number_of_edges_in_first_half_of_paired_time_period = []
number_of_edges_in_second_half_of_paired_time_period = []

for i in range(1, N):
    print("\n---------- Round ", i, "----------\n")
    
    #Create the first dataframe for each round
    first_df = total_df.loc[(total_df['timestamp'] >= time_for_each_graph * (i - 1) + min_time) & (total_df['timestamp'] < time_for_each_graph * i + min_time)]
    print(first_df)
    
    #Create the second dataframe for each round
    #If we are not in the last subgraph, we create the subgraph with an open set from the right [first_edge, last_edge),
    #so the last element (edge) of the subgraph does not overlapse with the first of the next subgraph
    if i != N - 1:
        second_df = total_df.loc[(total_df['timestamp'] >= time_for_each_graph * i + min_time) & (total_df['timestamp'] < time_for_each_graph * (i +1) + min_time)]
        #If we are on the last subgraph we include an '=' to the comparison, so the last element will be definitely included to the subgraph 
    else:
        second_df = total_df.loc[(total_df['timestamp'] >= time_for_each_graph * i + min_time) & (total_df['timestamp'] <= time_for_each_graph * (i +1) + min_time)]
    print(second_df)

    #Create the first subgraph using the first dataframe
    first_G = nx.from_pandas_edgelist(first_df, source = 'source', target = 'target', create_using = nx.Graph(), edge_attr = "timestamp")
    print("\n---------- First Graph ----------")
    print("\n", first_G)
    #Create the second subgraph using the second dataframe
    second_G = nx.from_pandas_edgelist(second_df, source = 'source', target = 'target', create_using = nx.Graph(), edge_attr = "timestamp")
    print("\n---------- Second Graph ----------")
    print("\n", second_G)
    
    #Create a graph that has only the nodes that are present in both the subgraphs of this round
    #and the edges that are present in the first subgraph of the round
    print("\n---------- Intersected Graph with First (1st) Period Edges ----------")  
    intersected_nodes_first_period_edges_G = first_G.copy()
    intersected_nodes_first_period_edges_G.remove_nodes_from(n for n in first_G if n not in second_G)
    #Remove self loops of each Node
    intersected_nodes_first_period_edges_G.remove_edges_from(nx.selfloop_edges(intersected_nodes_first_period_edges_G))
    print("\n", intersected_nodes_first_period_edges_G)
    print("\n----------------------------------")
    
    #Create a graph that has only the nodes that are present in both the subgraphs of this round
    #and the edges that are present in the second subgraph of the round
    print("\n---------- Intersected Graph with Second (2nd) Period Edges ----------")  
    intersected_nodes_second_period_edges_G = second_G.copy()
    intersected_nodes_second_period_edges_G.remove_nodes_from(n for n in second_G if n not in first_G)
    #Remove self loops of each Node
    intersected_nodes_second_period_edges_G.remove_edges_from(nx.selfloop_edges(intersected_nodes_second_period_edges_G))
    print("\n", intersected_nodes_second_period_edges_G)
    print("\n----------------------------------")
    
    # print("\n---------- Intersected Nodes and Edges of each Subgraph ----------")
    # print("\n Nodes: ", intersected_nodes_first_period_edges_G.nodes)
    # print("\n Edges 1: ", intersected_nodes_first_period_edges_G.edges)
    # print("\n Edges 2: ", intersected_nodes_second_period_edges_G.edges)
    # print("\n--------------------------------")
    
    #Create a Union Graph that contains the intersected nodes and the union of the edges of the 2 graphs
    union_G = nx.compose(intersected_nodes_first_period_edges_G, intersected_nodes_second_period_edges_G)
    
    # print("\n---------- Union Graph Data ----------")
    # print("\n", union_G)
    # print("\n Nodes: ", union_G.nodes)
    # print("\n Edges: ", union_G.edges)
    # print("\n--------------------------------")
    
    #Push into the corresponding array the number of nodes and adges of the their subgraph
    number_of_nodes_in_each_paired_time_period.append(intersected_nodes_first_period_edges_G.number_of_nodes())
    number_of_edges_in_first_half_of_paired_time_period.append(intersected_nodes_first_period_edges_G.number_of_edges())
    number_of_edges_in_second_half_of_paired_time_period.append(intersected_nodes_second_period_edges_G.number_of_edges())
    
    #Only for the last pair of subgraphs (in the last time period)
    if i == N - 1: 
        #Call the function that calculates the metrics for those 3 graphs that we have created
        metrics_list_1 = all_metrics_calc(intersected_nodes_first_period_edges_G, "1st Subgraph", i, False)
        metrics_list_2 = all_metrics_calc(intersected_nodes_second_period_edges_G, "2nd Subgraph", i, False)
        metrics_list_union = all_metrics_calc(union_G, "Union Graph", i, False)
        
        #Create dataframes and files for the metrics and the existing edges of each of the 3 graphs
        create_dataframes_and_files(metrics_list_union, "union_graph_data.csv", "union_graph_existing_edges.csv", True)
        create_dataframes_and_files(metrics_list_1, "1st_subgraph_data.csv", "1st_subgraph_existing_edges.csv", True)
        create_dataframes_and_files(metrics_list_2, "2nd_subgraph_data.csv", "2nd_subgraph_existing_edges.csv", True)
    
print("\nNodes in intersected Graphs: ", number_of_nodes_in_each_paired_time_period)
print("\nEdges in intersected Graphs (1st subgraph) : ", number_of_edges_in_first_half_of_paired_time_period)
print("\nEdges in intersected Graphs (2nd subgraph) : ", number_of_edges_in_second_half_of_paired_time_period)

#Styling for the plots
nodes_font = {'family':'serif','color':'darkred','size':15}
edges_font = {'family':'serif','color':'midnightblue','size':15} 

#Create a plot to show the nodes each pair of subgraphs contains and the evolution of their number
plt.plot(number_of_nodes_in_each_paired_time_period, color='red', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt.title("V∗ [tj−1, tj+1] \nNodes Evolution for the total " + str(N - 1) + " paired time periods", fontdict = nodes_font)
plt.xlabel("Paired Time Period (N -1 = " + str(N -1) + ")", fontdict = nodes_font)
plt.ylabel("Total Number of Nodes", fontdict = nodes_font)
plt.grid(True)
plt.show() 

#Create a plot to show the edges of the first subgraph of each pair of subgraphs and the evolution of their number
plt.plot(number_of_edges_in_first_half_of_paired_time_period, color='blue', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt.title("E∗ [tj−1, tj] \nEdges Evolution of the first (1st) subgraph for the total " + str(N - 1) + " paired time periods", fontdict = edges_font)
plt.xlabel("Paired Time Period (N -1 = " + str(N -1) + ")", fontdict = edges_font)
plt.ylabel("Total Number of Edges", fontdict = edges_font)
plt.grid(True)
plt.show()

#Create a plot to show the edges of the second subgraph of each pair of subgraphs and the evolution of their number
plt.plot(number_of_edges_in_second_half_of_paired_time_period, color='blue', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt.title("E∗ [tj , tj+1] \nEdges Evolution of the second (2nd) subgraph for the total " + str(N - 1) + " paired time periods", fontdict = edges_font)
plt.xlabel("Paired Time Period (N -1 = " + str(N -1) + ")", fontdict = edges_font)
plt.ylabel("Total Number of Edges", fontdict = edges_font)
plt.grid(True)
plt.show()