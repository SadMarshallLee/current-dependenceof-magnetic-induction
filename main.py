import numpy
import numpy as np
import matplotlib.pyplot as plt

B = np.loadtxt('B.txt')
J = np.loadtxt('I.txt')

B_T = np.array([])

for i in range(len(B)):
    B_T = np.append(B_T, B[i] / 10)

coef_b = np.array([])
coef_k = np.array([])

for i in range(len(B_T) - 1):
    M1 = numpy.array([[J[i], 1], [J[i + 1], 1]])
    V1 = numpy.array([B_T[i], B_T[i + 1]])
    coef = numpy.linalg.solve(M1, V1)
    coef_b = np.append(coef_b, coef[1])
    coef_k = np.append(coef_k, coef[0])

k = np.average(coef_k)
b = np.average(coef_b)

B_add = B[len(B) - 1]
J_add = J[len(J) - 1]

new_J = np.array([])
new_B = np.array([])
while B_add <= 1500.0:
    J_add = J_add + 0.5
    new_J = np.append(new_J, J_add)
    B_add = k * J_add + b
    new_B = np.append(new_B, B_add)

J_max = new_J.max()
B_max = new_B.max()

fin_B = np.savetxt('final_B.txt', np.concatenate((B, new_B)))
fin_J = np.savetxt('final_I.txt', np.concatenate((J, new_J)))

plt.text(100, 1700, 'Bmax =' + str(B_max))
plt.text(100, 1600, 'Imax = ' + str(J_max))

plt.plot(J, B_T, 'b')
plt.plot(new_J, new_B, 'b')

plt.xlabel('I, A')
plt.ylabel('B, mT')

plt.locator_params(axis='y', nbins=20)
plt.locator_params(axis='x', nbins=20)
plt.grid()

plt.savefig('B(I).png')
plt.show()
