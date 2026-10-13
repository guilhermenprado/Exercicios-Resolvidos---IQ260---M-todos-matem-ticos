import numpy as np
import matplotlib.pyplot as plt

n=8

#declarando e calculando vetor x
vetorlx = np.array([0.0, 10.8, 16.03, 22.91, 32.56, 36.76, 39.88, 43.68])

#declarando e calculando vetor fx
vetorlfx = np.array([0.0, 0.299, 0.576, 1.036, 1.781, 2.432, 2.846, 3.304])

#Determinando o polinomio interpolador de lagrange

def lagrange(xx):
    l=1
    lx=0
    i=0
    while(i<n):
        l=1
        j=0
        while(j<i):
            l=l*(xx-vetorlx[j])/(vetorlx[i]-vetorlx[j])
            j=j+1
        j=i+1
        while(j<n):
            l=l*(xx-vetorlx[j])/(vetorlx[i]-vetorlx[j])
            j=j+1
        lx=lx+l*vetorlfx[i]
        i=i+1
    return lx

#gerar os dados para o grafico usando lagrange
x_grafico = np.linspace(min(vetorlx), max(vetorlx), 100)
y_grafico = np.vectorize(lagrange)(x_grafico)

#Determinando a tecnica de spliles quadratica

#Declarando matriz A
matrizA=np.array([
[1, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 1, 1]
], dtype=float)

#Declarando vetor b
vetorb=np.array([
    0,
    2*(vetorlfx[1]-vetorlfx[0])/(vetorlx[1]-vetorlx[0]),
    2*(vetorlfx[2]-vetorlfx[1])/(vetorlx[2]-vetorlx[1]),
    2*(vetorlfx[3]-vetorlfx[2])/(vetorlx[3]-vetorlx[2]),
    2*(vetorlfx[4]-vetorlfx[3])/(vetorlx[4]-vetorlx[3]),
    2*(vetorlfx[5]-vetorlfx[4])/(vetorlx[5]-vetorlx[4]),
    2*(vetorlfx[6]-vetorlfx[5])/(vetorlx[6]-vetorlx[5]),
    2*(vetorlfx[7]-vetorlfx[6])/(vetorlx[7]-vetorlx[6])
], dtype=float)

#Declarando vetor x
n=8
vetorx=np.zeros(n, dtype=float)

#Aplicando metodo de gaauss com pivotamento parcial

#declarando variaveis
h=0
k=0
i=0
j=0
imax=0
f=0


#executando metodo Eliminacao de Gauss com pivotamento parcial
while h<n and k<n:
    imax=h+np.argmax(np.abs(matrizA[h:,k]))
    if matrizA[imax,k]==0:
        k=k+1
    else:
        matrizA[[h,imax]]=matrizA[[imax,h]]
        vetorb[[h,imax]]=vetorb[[imax,h]]
        i=h+1
        while i<n:
            f=matrizA[i,k]/matrizA[h,k]
            matrizA[i,k]=0
            j=k+1
            while j<n:
                matrizA[i,j]=matrizA[i,j]-f*matrizA[h,j]
                j=j+1
            vetorb[i]=vetorb[i]-vetorb[h]*f
            i=i+1
        h=h+1
        k=k+1

#Executando a substituicao reversa
vetorx[n-1]=vetorb[n-1]/matrizA[n-1,n-1]
i=n-2
while i>=0:
    s=0
    j=i+1
    while j<n:
        s=s+matrizA[i,j]*vetorx[j]
        j=j+1
    vetorx[i]=(vetorb[i]-s)/matrizA[i,i]
    i=i-1

#Criando a matriz de coeficientes para o polinomio de splines quadratica
matrizcoef=np.zeros((n,3), dtype=float)   
i=1
while i<n:
    matrizcoef[i,0]=vetorlfx[i-1]
    matrizcoef[i,1]=vetorx[i-1]
    matrizcoef[i,2]=(vetorx[i]-vetorx[i-1])/(2*(vetorlx[i]-vetorlx[i-1]))
    i=i+1

#gerar os dados para o grafico usando splines quadratica
i=0
y_grafico2 = np.zeros(len(x_grafico))
while i<n-1:
    indices = np.where((x_grafico >= vetorlx[i]) & (x_grafico <= vetorlx[i+1]))[0]
    y_grafico2[indices] = (matrizcoef[i+1,0] + 
                           matrizcoef[i+1,1]*(x_grafico[indices]-vetorlx[i]) + 
                           matrizcoef[i+1,2]*(x_grafico[indices]-vetorlx[i])**2)
    i=i+1

#gerar o grafico
plt.figure(figsize=(8, 5))
plt.plot(
    x_grafico, y_grafico, label="Polinomio de Lagrange", color="blue", linewidth=2
)
plt.plot(
    x_grafico, y_grafico2, label="Splines Quadráticas", color="red", linewidth=2
)
plt.plot(
    vetorlx, vetorlfx, "ro", label="Pontos Conhecidos"
)
plt.title("Interpolação de Lagrange e Splines Quadráticas")
plt.xlabel("q/ls-1")
plt.ylabel("DP/KPa")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

plt.show()