import numpy as np


#Declarando matriz A
matrizA=np.array([
[3,2,-1],
[1,-1,0],
[0,2,5]
], dtype=float)

#Declarando vetor b
vetorb=np.array([
    0,
    1,
    2
], dtype=float)

#Declarando vetor x novo
vetorxNovo=np.array([
    1,
    1,
    1
], dtype=float)

#Declarando vetor x velho

vetorxVelho=vetorxNovo.copy()

#Aplicando metodo iterativo de gauss seidel

#declarando variaveis
n=3
i=0
s=0
j=0
tolerancia=1e-2
erro=1
aux=0

#executa o metodo
while (erro>=tolerancia):
    i=0
    while (i<n):
        s=0
        j=0
        while (j<=i-1):
            s=s+matrizA[i,j]*vetorxNovo[j]
            j=j+1
        j=i+1
        while (j<n):
            s=s+matrizA[i,j]*vetorxVelho[j]
            j=j+1
        vetorxNovo[i]=(vetorb[i]-s)/matrizA[i,i]
        print(i,vetorxNovo,erro)
        i=i+1
    print(vetorxNovo,vetorxVelho)
    aux=(vetorxNovo-vetorxVelho)**2
    print(aux)
    erro=np.sqrt(np.sum(aux))
    print(i,vetorxNovo,erro)
    vetorxVelho=vetorxNovo.copy()

print(vetorxNovo)

"""
#Declarando matriz A
matrizA=np.array([
[1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.12, 0, 0, 0, 0, 0, -1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.12, 0, 0, 0, 0, 0, -1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.12, 0, 0, 0, 0, 0, -1],
[0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0 ,0],
[0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1,],
[0, 0, 0, 0, (1 - 0.6) ,0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, - 28/70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, - 2/70, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

#Declarando vetor b
vetorb=np.array([
    0,
    0,
    0,
    155,
    465,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
], dtype=float)

#Declarando vetor x novo
vetorxNovo=np.array([
    100,
    100,
    10,
    100,
    100,
    10,
    100,
    100,
    10,
    100,
    100,
    10,
    100,
    100,
    10
], dtype=float)

#Declarando vetor x velho

vetorxVelho=vetorxNovo

#Aplicando metodo iterativo de gauss seidel com pivotamento parcial

#declarando variaveis
n=15
i=0
s=0
j=0
tolerancia=1e-2
erro=1


#Cria a matriz sem 0 na diagonal
h=0
k=0
imax=0
while h<n and k<n:
    imax=h+np.argmax(np.abs(matrizA[h:,k]))
    if matrizA[imax,k]==0:
        k=k+1
    else:
        matrizA[[h,imax]]=matrizA[[imax,h]]
        vetorb[[h,imax]]=vetorb[[imax,h]]
    k=k+1
    h=h+1
"""