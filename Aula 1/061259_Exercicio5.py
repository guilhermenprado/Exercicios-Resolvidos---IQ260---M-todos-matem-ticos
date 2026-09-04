import matplotlib.pyplot as plt

Termo0=4.568
Termo1=-8.975*10**(-3)
Termo2=3.631*10**(-5)
Termo3=-3.407*10**(-8)
Termo4=1.091*10**(-11)
R=8.314
Cp=[]
Tabela=[35.87,36.25,36.66,37.09,37.55,38.03,38.52,39.03,39.56,40.10,40.65]
Temperatura=[]

for T in range (300,410,10):
    Cp.append(R*(Termo0+Termo1*T+Termo2*T**2+Termo3*T**3+Termo4*T**4))
    Temperatura.append(T)

plt.plot(Temperatura,Tabela, label='Tabela')
plt.plot(Temperatura,Cp, label='Equação')

plt.title('Comparação entre Capacidade Calorífica')
plt.xlabel('Temperatura (K)')
plt.ylabel('Capacidade Calorífica (J/mol.K)')
plt.legend()
plt.grid(True)

plt.show()