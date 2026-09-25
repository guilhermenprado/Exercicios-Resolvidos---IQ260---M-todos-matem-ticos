import numpy as np

#Declarando matriz
matrizA=np.array([
[3, 2, -1],
[1, -1, 0],
[0, 2, 5]

], dtype=float)

#Declarando vetor
vetorb=np.array(
[0,
    1,
    2,
], dtype=float)

#Declarando vetor x
n=3
vetorx=np.zeros(n, dtype=float)

#Aplicando metodo de gaauss com pivotamento parcial

#declarando variaveis
h=0
k=0
i=0
j=0
imax=0
f=0
s=0

#executando metodo Decomposicao LU pelo metodo de Doolittle com pivotamento parcial
while h<n and k<n:
    imax=h+np.argmax(np.abs(matrizA[h:,k]))
    if matrizA[imax,k]==0:
        k=k+1
    else:
        matrizA[[h,imax]]=matrizA[[imax,h]]
        vetorb[[h,imax]]=vetorb[[imax,h]]
        i=h+1
        while i<n:
            #print(matrizA[i,k],matrizA[k,k])
            matrizA[i,k]=matrizA[i,k]/matrizA[k,k]
           # print(matrizA[i,k])
            j=k+1
            while j<n:
               # print(matrizA[i,j],matrizA[i,k],matrizA[k,j])
                matrizA[i,j]=matrizA[i,j]-matrizA[i,k]*matrizA[k,j]
                #print(matrizA[i,j],i,j)
                j=j+1
            i=i+1
        h=h+1
        k=k+1

print(matrizA)
print(vetorb)

#Executando a substituicao progressiva
vetorx[0]=vetorb[0]
i=1
while i<n:
    s=0
    j=0
    while j<i:
        print(s,matrizA[i,j],vetorx[j],i,j)
        s=s+matrizA[i,j]*vetorx[j]
        j=j+1
    print(vetorx[i],vetorb[i],s)
    vetorx[i]=(vetorb[i]-s)
    print(vetorx[i])
    i=i+1

print(vetorx)

#Executando a substituicao reversa
vetorx[n-1]=vetorx[n-1]/matrizA[n-1,n-1]
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