import numpy as np

#Declaracão de variaveis
Ti=298.15 #K
Tf=328.15 #K
Pi=1200000 #Pa

R =8.314 #Pa*m3/(K*mol)
Tc= 425.2 #K
Pc= 3.8*10**6 #Pa

a= 1.466*10**(-1) #27*R**2*Tc**2/(64*Pc)
b= 0.1226*10**(-3) #R*Tc/(8*Pc)
print(a,b)

## Definindo a funcao de volume da esfera
def volume(x):
    resultado = Pi*x**3-(Pi*b+R*Ti)*x**2+a*x-a*b

    return resultado

    ## Definindo a derivada da funcao a ser calculada
def Derivada(x):
    resultado = 3*Pi*x**2-2*(Pi*b+R*Ti)*x+a

    return resultado

## Metodo de newton-raphson

erro=1
xvelho=0.216
xnovo=0
i=0


while erro>=10**(-10):
    i=i+1
    xnovo=xvelho-volume(xvelho)/Derivada(xvelho)
    erro=abs(xnovo-xvelho)
    print(i, xvelho, xnovo, erro)
    xvelho=xnovo

# Calculando a pressao final

Pf=(R*Tf/(xnovo-b)-a/(xnovo**2))*10**(-5)

print(f"Resolvendo o Exercicio pelo Metodo da Newton-Raphson temos o valor de {Pf} após {i} interações.")