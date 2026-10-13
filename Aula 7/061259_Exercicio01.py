import numpy as np
import matplotlib.pyplot as plt

n=20

#declarando e calculando vetor x
vetorx = np.zeros(n)

i=0
while i<n:
    vetorx[i]=i/(n-1)
    i=i+1

#Resolvendo analiticamente

def funcao(x):
    #usando taylor até o 3 valor
    y=1-2/np.sqrt(np.pi)*(x-(x**3)/3+(x**5)/10)
    return y

#declarando e calculando vetor fx
vetorfx = np.zeros(n)

i=0
while i<n:
    vetorfx[i]=funcao(vetorx[i])
    i=i+1

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
plt.xlabel("y/sqt(4vt)")
plt.ylabel("vx/v0")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

plt.show()