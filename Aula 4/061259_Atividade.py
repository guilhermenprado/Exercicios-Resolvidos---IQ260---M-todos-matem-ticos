import numpy as np

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

#Declarando vetor x
n=15
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

print(matrizA)
print(vetorb)

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

print(vetorx)





