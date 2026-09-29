import numpy as np

#definindo vetor x
vetorx=np.array([0.002,0.002,0.002,0.002])

#Definindo parametros do processo
q1=1
q2=1
q3=2
CAE=0.1
CBE=0.3
V=100
kf=2e9*np.exp(-48700/(8.314*298.15))

#definindo funcoes
def f1(CA,CB,CC,CD):return q1*CAE-q3*CA-kf*CA*CB*V
def f2(CA,CB,CC,CD):return q2*CBE-q3*CB-kf*CA*CB*V
def f3(CA,CB,CC,CD):return -q3*CC+kf*CA*CB*V
def f4(CA,CB,CC,CD):return -q3*CD+kf*CA*CB*V

#definindo vetor f(x)
vetorfx=np.array([f1,f2,f3,f4], dtype=object)


#Definindo funcoes da matriz jacobiana
def a11(CA,CB):return -q3-kf*CB*V
def a12(CA,CB):return -kf*CA*V
def a13(CA,CB):return 0
def a14(CA,CB):return 0
def a21(CA,CB):return -kf*CB*V
def a22(CA,CB):return -q3-kf*CA*V
def a23(CA,CB):return 0
def a24(CA,CB):return 0
def a31(CA,CB):return kf*CB*V
def a32(CA,CB):return kf*CA*V
def a33(CA,CB):return -q3
def a34(CA,CB):return 0
def a41(CA,CB):return kf*CB*V
def a42(CA,CB):return kf*CA*V
def a43(CA,CB):return 0
def a44(CA,CB):return -q3

#definindo matriz J
matrizJ=np.array([
    [a11,a12,a13,a14],
    [a21,a22,a23,a24],
    [a31,a32,a33,a34],
    [a41,a42,a43,a44]
], dtype=object)


#Utilizando metodo de broyden

CA=vetorx[0]
CB=vetorx[1]
CC=vetorx[2]
CD=vetorx[3]

print(CA,CB,CC,CD)

resultados= np.vectorize(lambda f: f(CA,CB,CC,CD))(vetorfx)
resultadomatriz= np.vectorize(lambda f: f(CA, CB))(matrizJ)

print(resultados)


#Executando a inversao da matriz

#Declarando matrizA = MatrizJ -->utiliza a matriz como numeros para o calculo
matrizJ_num= np.vectorize(lambda f: f(CA, CB))(matrizJ).astype(float)
print(matrizJ_num)

#Declarando matriz identidade =vetor b
matrizI=np.array([
[1,0,0,0],
[0,1,0,0],
[0,0,1,0],
    [0,0,0,1],
], dtype=float)

#Declarando matrizJinv
matrizJinv=np.array([
[0,0,0,0],
[0,0,0,0],
[0,0,0,0],
    [0,0,0,0],
], dtype=float)

#declarando variaveis
n=4
h=0
k=0
i=0
j=0
imax=0
f=0
matrizJ_num2=matrizJ_num.copy()

#executando metodo Decomposicao LU pelo metodo de Doolittle com pivotamento parcial
while h<n and k<n:
    imax=h+np.argmax(np.abs(matrizJ_num2[h:,k]))
    if matrizJ_num2[imax,k]==0:
        k=k+1
    else:
        matrizJ_num2[[h,imax]]=matrizJ_num2[[imax,h]]
        matrizI[[h,imax]]=matrizI[[imax,h]]
        i=h+1
        while i<n:
            matrizJ_num2[i,k]=matrizJ_num2[i,k]/matrizJ_num2[k,k]
            j=k+1
            while j<n:
                matrizJ_num2[i,j]=matrizJ_num2[i,j]-matrizJ_num2[i,k]*matrizJ_num2[k,j]
                j=j+1
            i=i+1
        h=h+1
        k=k+1

print(matrizJ_num2)

#Executando a substituicao progressiva
c=0
while c<4:
    matrizJinv[0,c]=matrizI[0,c]
    i=1
    while i<n:
        s=0
        j=0
        while j<i:
            s=s+matrizJ_num2[i,j]*matrizJinv[j,c]
            j=j+1
        matrizJinv[i,c]=(matrizI[i,c]-s)
        i=i+1

    #Executando a substituicao reversa
    matrizJinv[n-1,c]=matrizJinv[n-1,c]/matrizJ_num2[n-1,n-1]
    i=n-2
    while i>=0:
        s=0
        j=i+1
        while j<n:
            s=s+matrizJ_num2[i,j]*matrizJinv[j,c]
            j=j+1
        matrizJinv[i,c]=(matrizJinv[i,c]-s)/matrizJ_num2[i,i]
        i=i-1

    c=c+1
print(matrizJinv)


#Calcular xi+1
vetorxnovo=vetorx-np.dot(matrizJinv,resultados)

#calcular Erro
aux2=(vetorxnovo-vetorx)**2
erro=np.sqrt(np.sum(aux2))

#realizando o calculo de J-1 numericamente
#declarando erro
erro=1
tolerancia=1e-5

#realizando as iteracoes do metodo
while (erro>=tolerancia):

    #calculando deltax
    deltax=vetorxnovo-vetorx

    #calculadno delta fx
    CA=vetorxnovo[0]
    CB=vetorxnovo[1]
    CC=vetorxnovo[2]
    CD=vetorxnovo[3]

    resultadosnovo= np.vectorize(lambda f: f(CA,CB,CC,CD))(vetorfx)

    deltafx=resultadosnovo-resultados
    resultados=resultadosnovo.copy()

    #calculando vetorX transposta
    deltax_col = deltax.reshape(-1, 1)
    deltax_lin = deltax.reshape(1, -1) 
    deltafx_col = deltafx.reshape(-1, 1)

    #calculando J-1 nova
    sup1 = deltax_col - np.dot(matrizJinv, deltafx_col)
    sup2 = np.dot(deltax_lin, matrizJinv)                       
    sup3 = np.dot(sup2,deltafx_col)[0, 0]

    matrizJinv = matrizJinv + np.dot(sup1, sup2) / sup3

    vetorx=vetorxnovo.copy()
    #Calcular xi+1
    vetorxnovo=vetorxnovo-np.dot(matrizJinv,resultadosnovo)

    #calcular Erro
    aux2=(vetorxnovo-vetorx)**2
    erro=np.sqrt(np.sum(aux2))

    print(vetorxnovo)
    print(erro)
