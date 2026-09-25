import numpy as np

#definindo vetor x
vetorx=np.array([0.0002,0.0002,0.0002,0.0002])

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

#declarando erro
erro=1
tolerancia=1e-5

#Utilizando metodo de newtown multidimensional
while (erro>=tolerancia):

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

    #executando metodo Decomposicao LU pelo metodo de Doolittle com pivotamento parcial
    while h<n and k<n:
        imax=h+np.argmax(np.abs(matrizJ_num[h:,k]))
        if matrizJ_num[imax,k]==0:
            k=k+1
        else:
            matrizJ_num[[h,imax]]=matrizJ_num[[imax,h]]
            matrizI[[h,imax]]=matrizI[[imax,h]]
            i=h+1
            while i<n:
                matrizJ_num[i,k]=matrizJ_num[i,k]/matrizJ_num[k,k]
                j=k+1
                while j<n:
                    matrizJ_num[i,j]=matrizJ_num[i,j]-matrizJ_num[i,k]*matrizJ_num[k,j]
                    j=j+1
                i=i+1
            h=h+1
            k=k+1

    print(matrizJ_num)

    #Executando a substituicao progressiva
    c=0
    while c<4:
        matrizJinv[0,c]=matrizI[0,c]
        i=1
        while i<n:
            s=0
            j=0
            while j<i:
                s=s+matrizJ_num[i,j]*matrizJinv[j,c]
                j=j+1
            matrizJinv[i,c]=(matrizI[i,c]-s)
            i=i+1

        #Executando a substituicao reversa
        matrizJinv[n-1,c]=matrizJinv[n-1,c]/matrizJ_num[n-1,n-1]
        i=n-2
        while i>=0:
            s=0
            j=i+1
            while j<n:
                s=s+matrizJ_num[i,j]*matrizJinv[j,c]
                j=j+1
            matrizJinv[i,c]=(matrizJinv[i,c]-s)/matrizJ_num[i,i]
            i=i-1

        c=c+1
    print(matrizJinv)

    #Calcular xi+1
    vetorxnovo=vetorx-np.dot(matrizJinv,resultados)

    #calcular Erro
    aux2=(vetorxnovo-vetorx)**2
    erro=np.sqrt(np.sum(aux2))

    vetorx=vetorxnovo.copy()

    print(vetorx)
    print(erro)