#This the python code for CST Method in Finine Element Analysis that I have used during my University's course project:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data_EL =  pd.read_csv(r"D:\university of tehran\Term 7\Finite Element Method\Project\DATA_Elements&Nodes.csv")

# Assigning Question 2 Data to their corresponding matrices
Node_No = int(Data_EL["Number_Joints"].max()) #number of joints #and we use the max method to extract only a number from the data frame above. 
Max_Freedom = int(Data_EL["Maximum_Degrees_Freedom"].max()) #In this question there are only horizontal and vertical displacements
Element_No = int(Data_EL["Element"].max())
AreaElement_No = int(Data_EL["Area Element"].max())
Constraints = Data_EL["Rx"].values
Constraint_No = int(np.count_nonzero(Constraints[:5]))
Force_No = int(np.count_nonzero(~np.isnan(np.array(Data_EL["F"].values[:5]))))

t_Width = Data_EL["t"].max()
v_Poisson = Data_EL["v"].max()
E_Modulus = Data_EL['E'].max()

K_StiffnessDimension= int(Max_Freedom  * Node_No)
K_matrix = np.zeros((K_StiffnessDimension, K_StiffnessDimension), dtype=float)
Q2_Data = Data_EL.values #In order to proceed with the available data, we convert pandas' DataFrame to NumPy Array
print(K_matrix)
print(Constraint_No)
print(Force_No)

# Calculating the required AREAS between the previously assigned elements 
# To calculate the triangle areas we define a function herein

def calculate_triangle_area(vertices):
    x1, y1, x2, y2, x3, y3 = vertices
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area

# Assuming data is a 2D NumPy array with the vertex coordinates
Q2_Data = np.array(Data_EL)

el_node_indices = []

Area = []

for i in range(AreaElement_No):
    el_node_indices = Q2_Data[i, 15:18].astype(int) - 1
    print(el_node_indices + 1)
    
# Convert vertices to a NumPy array before using flatten
    vertices = np.array([Q2_Data[el_node_indices[0], 3], Q2_Data[el_node_indices[0], 4],
                         Q2_Data[el_node_indices[1], 3], Q2_Data[el_node_indices[1], 4],
                         Q2_Data[el_node_indices[2], 3], Q2_Data[el_node_indices[2], 4]]).flatten()
    
    area = calculate_triangle_area(vertices)
    
    Area.append(area)
    print(Area)

# Computing B matrix for each area element
def calculate_B(x, y):
    # Compute components of the B matrix
    # We note that the 0, 1, and 2 represent i, j, and m
    B_matrix = np.array([[y[1] - y[2], 0, y[2] - y[0], 0, y[0] - y[1], 0],[0, x[2] - x[1], 0, x[0] - x[2], 0, x[1] - x[0]],[x[2] - x[1], y[1] - y[2], x[0] - x[2], y[2] - y[0], x[1] - x[0], y[0] - y[1]]]) / (2 * Area[i])
        
    return B_matrix

B_total = []
B = []
for i in range(AreaElement_No):
    el_node_indices = Q2_Data[i, 15:18].astype(int) - 1
    print("Area Nodes:",el_node_indices + 1)
    
    # Extract node coordinates for the current element
    x_coords = Q2_Data[el_node_indices, 3]
    y_coords = Q2_Data[el_node_indices, 4]
    
    # Calculate B matrix for the current element
    B_el_areas = calculate_B(x_coords, y_coords)
    print("B",str(i+1),'=', B_el_areas)
    
    # Append B matrix for the current element to the list
    B_total.append(B_el_areas)

    print("B total:", B_total)


# Computing the D matrix and assigning the plane stress related matrix:
def calculate_D(E, v):
    # Initialize D matrix
    D_matrix = np.zeros((3, 3), dtype=float)
    
    # Assign values to D matrix
    D_matrix[0, 0] = 1
    D_matrix[1, 1] = 1
    D_matrix[0, 1] = v
    D_matrix[1, 0] = v
    D_matrix[2, 2] = (1 - v) / 2
    
    # Scale D matrix by material properties
    D_matrix = (E / (1 - v**2)) * D_matrix
    
    return D_matrix

# Example usage
E_amount = E_Modulus  # Young's modulus
v_amount = v_Poisson   # Poisson's ratio

# Calculate D matrix for the given material properties
D_matrix_calculated = calculate_D(E_amount, v_amount)
print("Calculated D Matrix:")
print(D_matrix_calculated)

# Assigning K matrix for each area element (CST):
K_el_T = []

for i in range(AreaElement_No):
    # Extract B matrix for the current element
    B_element = B_total[i]
    
    # Transpose of B matrix
    B_transpose = B_element.T
    
    # Calculate element stiffness matrix
    K_element = t_Width * Area[i] * np.dot(np.dot(B_transpose, D_matrix_calculated), B_element)
    
    # Append K_element to the list K_K_el_T
    K_el_T.append(K_element)
    
    print('K_ELEMENT',str(i + 1), K_element)
    print('K matrix for 4 areas in one array:', K_el_T)


# Assembling the K matrix

# We use the previously calculated K_el_T which contains all four 6x6 elemental matrices in an array form to assemble the total K matrix

# Assuming we have defined K_el_T, Q2_Data, and AreaElement_No

# Initialize K_total with zeros
K_total = np.zeros((2 * Node_No, 2 * Node_No))

for element in range(AreaElement_No):
    k = K_el_T[element]
    el_node_indices = Q2_Data[element, 15:18].astype(int) - 1
    # Connection table 
    for a in range(3):
        for b in range(3):
            global_a = 2 * el_node_indices[a] # each node of CST has 2 degrees of freedom
            global_b = 2 * el_node_indices[b]

            # Add the contribution of k to the corresponding submatrix in K_total
            K_total[global_a:global_a+2, global_b:global_b+2] += k[a*2:a*2+2, b*2:b*2+2]

print(K_total)


# Applying the F=Kd equation to find the nodal displacements and internal stresses

# Assuming we have defined K_total, F, Q2_Data, Constraint_No, and Force_No

# Copy original matrices to avoid modifying the original ones
original_shape = K_total.shape
K_total_modified = K_total.copy()
F = np.zeros((K_StiffnessDimension,1 ), dtype=float)

Constraint_nodes = np.array([np.ravel(np.column_stack((Q2_Data[:5, 5], Q2_Data[:5, 6]))).reshape(-1, 1)])
# for i in range(Node_No)]

print(Constraint_nodes)

global_x_fixed = []
global_y_fixed = []
for node in range(Node_No * 2):
    if np.array(Constraint_nodes[0][node, 0]) == 1:
        global_x_fixed.append(node)
        global_y_fixed.append(node)
    
    # Set the corresponding values in the force vector to 0
    F[global_y_fixed] = 0.0
    F[global_x_fixed] = 0.0

K_total_modified = np.delete(np.delete(K_total_modified, global_x_fixed, axis=0), global_y_fixed, axis=1)


# Convert the array elements to numeric types before applying np.cos and np.sin
cos_values = np.vectorize(np.cos)(np.array(Data_EL['External_Force_Angle'][:Force_No])*np.pi/180)
sin_values = np.vectorize(np.sin)(np.array(Data_EL['External_Force_Angle'][:Force_No])*np.pi/180)
Q2_Data_values = np.array(Data_EL['F'].values[:Force_No])
# Flatten the array and reshape it to a single column to have external forces like F array
applied_forces = np.array([Q2_Data_values * cos_values, Q2_Data_values * sin_values]).flatten(order='F').reshape(-1, 1) 

# Flatten the array and reshape it to a single column

# Interleave elements
Constraint_forces = np.array(np.ravel(np.column_stack((Q2_Data[:5, 5], Q2_Data[:5, 6]))).reshape(-1, 1))

for node in range(Force_No * 2):
    F[node] = applied_forces[node] + Constraint_forces[node]
    if F[node] == 1:
        F[node] = 0
    else:
        F[node]

        
F_modified = np.delete(F, global_x_fixed, axis=0)
print(F_modified)
U_modified = (np.linalg.solve(K_total_modified, F_modified))
# Add the deleted rows back to U_modified with zeros
zero_rows = np.zeros((Force_No * 2, 1))
zero_rows[~np.isin(np.arange(len(zero_rows)), global_x_fixed)] = U_modified
  
U_Final = zero_rows

print("Nodal displacements:")
print(U_Final)

#Computing the internal stresses for the area elements (CSTs)

for i in range(AreaElement_No):
    el_node_indices = Q2_Data[i, 15:18].astype(int) - 1
    # Calculate indices for selecting elements from U_Final

    element_node = []
    for j in range(len(el_node_indices)):
        node_pair = [2 * el_node_indices[j], 2 * el_node_indices[j] + 1 ]
        element_node.append(node_pair)
        
    selected_indices = np.array(element_node)

    # Select corresponding elements from U_Final
    U_element = U_Final[selected_indices].reshape(-1,1)

    Sigma = np.dot(np.dot(D_matrix_calculated, B_total[i]), U_element)

    print("Sigma",str(i+1),'=', Sigma)
