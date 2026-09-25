import numpy as np

#Declarando as variaveis
D=25.4*10**(-3)
e=0.0457*10**(-3)
Dp=-2*10**5
ro=1000
mi=0.001
L=50


## Definindo a funcao de fator de atrito
def fatorAtrito(x):
    resultado = 1/(-4*np.log10(0.27*e/D+(7/(ro*x*D/mi))**0.9))**2

    return resultado

## Definindo a funcao de velocidade
def velocidade(x):
    resultado = Dp/ro+2*fatorAtrito(x)*L*x**2/D

    return resultado

## Aplicando metodo da secante

vk2 = 0
vk1 = 3
vk0 = 3.01
erro=1
i=0

while erro>=10**(-10):
    i=i+1
    vk2=(vk0*velocidade(vk1)-vk1*velocidade(vk0))/(velocidade(vk1)-velocidade(vk0))
    erro=abs(vk2-vk1)
    vk0=vk1
    vk1=vk2

Vazao= np.pi*D**2/4*vk2*1000

print(f"Resolvendo o Exercicio pelo Metodo da Secante temos o valor de {Vazao} após {i} interações.")