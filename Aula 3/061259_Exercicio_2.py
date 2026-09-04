import numpy as np

## Declarando as constantes do do problema
a0=3.259
a1=1.356*10**(-3)
a2=1.502*10**(-5)
a3=-2.374*10**(-8)
a4=1.056*10**(-11)

P1=10
P2=1
T1=500

##Calculando a parte constante da formula
CTE=np.log(P2/P1)+a0*np.log(T1)+a1*T1+a2/2*T1**2+a3/3*T1**3+a4/4*T1**4


## Definindo a funcao a ser calculada
def Rankine(T2):
    resultado = a0*np.log(T2)+a1*T2+a2/2*T2**2+a3/3*T2**3+a4/4*T2**4-CTE

    return resultado

    ## Definindo a derivada da funcao a ser calculada
def Derivada(T2):
    resultado2 = a0/T2+a1+a2*T2+a3*T2**2+a4*T2**3

    return resultado2

    
## Metodo de newton-raphson

erro=1
xvelho=400
xnovo=0
i=0


while erro>=10**(-10):
    i=i+1
    xnovo=xvelho-Rankine(xvelho)/Derivada(xvelho)
    erro=abs(xnovo-xvelho)
    print(i, xvelho, xnovo, erro)
    xvelho=xnovo

cB=xnovo
iB=i


print(f"Resolvendo o Exercicio pelo Metodo da Newton-Raphson temos o valor de {cB} após {iB} interações.")