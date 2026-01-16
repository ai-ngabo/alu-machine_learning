#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3
x0 = np.arange(0, 11)

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure(figsize=(10, 8))
plt.subplot(3, 2, 1)
plt.plot(x0, y0, color='red')

plt.subplot(3, 2, 2)
plt.scatter(x1, y1, color='magenta')
plt.xlabel('Weight (lbs)')
plt.ylabel('Height (in)')
plt.title("Men's Height vs Weight")

plt.subplot(3, 2, 3)
plt.plot(x2, y2)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.yscale('log')

plt.subplot(3, 2, 4)
plt.plot(x3, y31, color = 'red', linestyle='--', label='C-14')
plt.plot(x3, y32, color = 'green', label='Ra-226')
plt.legend()
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.xlim(0, 20000)

plt.subplot(3, 2, (5, 6))
plt.hist(student_grades, bins = np.arange(0, 101, 10), edgecolor='black')
plt.xticks(np.arange(0, 101, 10))
plt.ylim(0, 30)
plt.xlabel('Grades')
plt.ylabel('Number of Students')

plt.tight_layout()
plt.show()
