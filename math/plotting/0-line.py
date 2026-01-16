#!/usr/bin/env python3
import numpy as np                                                                                                                                           
import matplotlib.pyplot as plt                                                                                                                              
                                                                                                                                                             
x = np.arange(0, 11)                                                                                                                                         
y = np.arange(0, 11) ** 3                                                                                                                                    

plt.figure(figsize=(10, 6))                                                                                                                                                   
plt.plot(x, y, color='red') 
plt.show() 
