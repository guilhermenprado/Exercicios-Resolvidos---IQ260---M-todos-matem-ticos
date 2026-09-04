import numpy as np

## Declarando variaiveis
T=100
Ta=25
Tw=500
N=3.5


## Definindo a funcao de troca de calor
def calor(x):
    resultado = np.cosh(N*(1-x))/np.cosh(N)-(T-Ta)/(Tw-Ta)

    return resultado
    
## Aplicando metodo da secante

xk2 = 0
xk1 = 0.6
xk0 = 0.61
erro=1
i=0

while erro>=10**(-10):
    i=i+1
    xk2=(xk0*calor(xk1)-xk1*calor(xk0))/(calor(xk1)-calor(xk0))
    erro=abs(xk2-xk1)
    print(i,xk0,xk1,xk2)
    xk0=xk1
    xk1=xk2

print(f"Resolvendo o Exercicio pelo Metodo da Secante temos o valor de {xk2} após {i} interações.")