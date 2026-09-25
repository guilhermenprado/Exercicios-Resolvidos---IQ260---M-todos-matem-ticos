import numpy as np

#definindo vetor x
vetorx=np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2])


#definindo vetor dados
dados=np.array([
    [0.25,0.20,0.15,0.17,0.2,0.2,0.1],
    [500,500,150,700,650,180,100]
               ])

#Definindo parametros do processo
rho = 1000
mu = 1e-3
g=9.81
rugosidade=0.000045

#definindo perda de carga
def K(q,d,l):
    #definindo fator de atrito
    Re=4*rho*np.abs(q)/(np.pi*mu*d)
    B=(37530/Re)**16
    A=(-2.457*np.log((7/Re)**0.9+0.27*rugosidade/d))**16
    fat=8*((8/Re)**12+1/(A+B)**1.5)**(1/12)

    #definindo perda de carga
    K=(8*fat*l)/(np.pi**2*g*d**5)
    return K

#definindo vetorK
vetork=np.array([0,0,0,0,0,0,0])

#definindo funcoes
def f1(q1,q2,q3,q4,q5,q6,q7):return 0.250-q1-q6
def f2(q1,q2,q3,q4,q5,q6,q7):return q1-q2-q7-0.070
def f3(q1,q2,q3,q4,q5,q6,q7):return q2+q3-0.050
def f4(q1,q2,q3,q4,q5,q6,q7):return q6-q5-0.030
def f5(q1,q2,q3,q4,q5,q6,q7):return q5+q7-q4-0.060
def f6(q1,q2,q3,q4,q5,q6,q7):return vetork[0]*q1**2+vetork[6]*q7**2-vetork[5]*q6**2-vetork[4]*q5**2
def f7(q1,q2,q3,q4,q5,q6,q7):return vetork[1]*q2**2-vetork[6]*q7**2-vetork[3]*q4**2-vetork[2]*q3**2


#definindo vetor reusltados fx
vetorfx=np.array([f1,f2,f3,f4,f5,f6,f7], dtype=object)


#declarando erro
erro=1
tolerancia=1e-12

#Utilizando metodo de newtown multidimensional
contador=0
while (erro>=tolerancia):

    #calculando vetorK
    i=0

    while i<7:
        vetork[i]=K(vetorx[i],dados[0,i],dados[1,i])
        i=i+1

    print(vetork)

    #definindo e calculando matriz J
    matrizJ=np.array([
        [-1,0,0,0,0,-1,0],
        [1,-1,0,0,0,0,-1],
        [0,1,1,0,0,0,0],
        [0,0,0,0,-1,1,0],
        [0,0,0,-1,1,0,1],
        [2*vetork[0]*vetorx[0],0,0,0,-2*vetork[4]*vetorx[4],-2*vetork[5]*vetorx[5],2*vetork[6]*vetorx[6]],
        [0,2*vetork[1]*vetorx[1],-2*vetork[2]*vetorx[2],-2*vetork[3]*vetorx[3],0,0,-2*vetork[6]*vetorx[6]]
    ], dtype=object)

    matrizJ_num=matrizJ.astype(float)

    print(matrizJ_num)

    resultados = np.array([f(*vetorx) for f in vetorfx])
    print(resultados)

    #Declarando matriz identidade = vetor b
    matrizI=np.array([
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1]
    ], dtype=float)

    #Declarando matrizJinv
    matrizJinv=np.array([
    [1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0]
    ], dtype=float)

    #Executando a inversao da matriz

    #declarando variaveis
    n=7
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
    while c<n:
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
    contador=contador+1
    print(vetorx)
    print(contador,erro)
print(f"Encontramos o vetor resultado {vetorx} após {contador} iterações com erro de {erro}")