import urllib.request
import numpy as np
urllib.request.urlretrieve(
    'https://gist.github.com/BirajCoder/a4ffcb76fd6fb221d76ac2ee2b8584e9/raw/4054f90adfd361b7aa4255e99c2e874664094cea/climate.csv', 
    'climate.txt')

climate_data = np.genfromtxt('climate.txt', delimiter=',', skip_header=1)

print(climate_data.shape)

weights = np.array([0.3,0.2,0.5])

yields = climate_data @ weights 

print(yields.shape)

climate_resuts = np.concatenate((climate_data,yields.reshape(10000,1)),axis=1)
# axis= 1: concatenate via column

print(climate_resuts)

np.savetxt('climate_results.txt',
           climate_resuts,
           fmt='%.2f',
           delimiter=',',
           header='temperature,rainfall,humiditymyield_apples',
           comments='')

arr2 = np.array([[1, 2, 3, 4], 
                 [5, 6, 7, 8], 
                 [9, 1, 2, 3]])

arr3 = np.array([[11, 12, 13, 14], 
                 [15, 16, 17, 18], 
                 [19, 11, 12, 13]])

#Adding a scalar
arr2 += 3 
print(arr2)


arr3 = np.array([
    [[11, 12, 13, 14], 
     [13, 14, 15, 19]], 
    
    [[15, 16, 17, 21], 
     [63, 92, 36, 18]], 
    
    [[98, 32, 81, 23],      
     [17, 18, 19.5, 43]]])

# Subarray using ranges
print(arr3[1, 0:1, 2])