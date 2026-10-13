import numpy as np
import matplotlib.pyplot as plt

n=10

#declarando e calculando vetor x
vetorx = np.zeros(n)

i=0
while i<n:
    vetorx[i]=10*i
    i=i+1

#declarando e calculando vetor fx
vetorfx = [1489.4,658.6,326.87,178.01,105.10,66.464,44.585,31.35,23.006,17.467]

#Determinando o polinomio interpolador de newtown
#declarando e calculando matrix f
matrizf = np.zeros((n,n))
i=0
while(i<n):
    matrizf[i,0]=vetorfx[i]
    i=i+1
j=1
while (j<n):
    i=0
    while(i<n-j):
        matrizf[i,j]=(matrizf[i+1,j-1]-matrizf[i,j-1])/(vetorx[i+j]-vetorx[i])
        i=i+1
    j=j+1

def newton(xx):
    xt=1
    nx=matrizf[0,0]
    j=0
    while (j<n-1):
        xt=xt*(xx-vetorx[j])
        nx=nx+matrizf[0,j+1]*xt
        j=j+1
    return nx

#gerar os dados para o grafico usando neton
x_grafico = np.linspace(min(vetorx), max(vetorx), 100)
y_grafico = np.vectorize(newton)(x_grafico)

#gerar o grafico
plt.figure(figsize=(8, 5))
plt.plot(
    x_grafico, y_grafico, label="Polinomio de Newton", color="blue", linewidth=2
)
plt.plot(
    vetorx, vetorfx, "ro", label="Pontos Conhecidos"
)
plt.title("Interpolação de Newton")
plt.xlabel("T/°C")
plt.ylabel("v/mm2s-1")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

plt.show()