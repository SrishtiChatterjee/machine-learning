import numpy as np
import matplotlib.pyplot as plt
#create the training set
x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

#print the training set
print(f"x_train={x_train}") #the f helps print the variable within double quotes
#print(x_train) works as well
print("y_train={y_train}")

#number of training examples:
print(f"x_train_shape: {x_train.shape}")

l=x_train.shape[0]
print(f"No of training examples: {l}")

n=len(x_train)
print(f"No of training examples: {n}")

#ith training example:
i=0
x_i=x_train[i]
y_i=y_train[i]
print(f"(x^({i+1}),y^({i+1})) = ({x_i},{y_i})")

#plotting the data:
plt.scatter(x_train,y_train, marker='o',c='r') #plot the data points #marker=shape of point c=colour default is blue
plt.title("Housing prices") #set title
plt.ylabel("Price in 1000s dollars") #set y-axis label
plt.xlabel("Size in 1000s sqft") #set x-axis label
plt.show()

#define parameters:
m=100
b=100

#function to get equation of regression lline
def compute_model_op(x,m,b):
  w=x.shape[0]
  f=np.zeros(w) #fill array with zeros
  for i in range(w):
    f[i] = (m * x[i]) + b        
  return f

tmp_f = compute_model_op(x_train, m, b,)
#plot model prediction:
plt.plot(x_train,tmp_f, c='b', label='Our Prediction')
#plot data points:
plt.scatter(x_train,y_train,marker="x", c="g", label='Actual Values')
plt.title("Housing Prices")
plt.ylabel("Price in 1000s dollars") #set y-axis label
plt.xlabel("Size in 1000s sqft") #set x-axis label
plt.legend()
plt.show()
