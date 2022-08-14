#print('Estaba la pájara pinta sentada en el verde limón) falta comilla al finale del print
print('Estaba la pájara pinta sentada en el verde limón')

name = 'Daniel'
edad = 38
#print('Mi nombre es' name 'y tengo' edad, 'años') falta agregar el .format o f.string y las comillas solo en el print, no en cada texto
print(f'Mi nombre es {name} y tengo {edad}, años')
print('Mi nombre es {} y tengo {}, años'.format(name, edad))

#"Ornitorrinco" + 45 no se puede concatenar un str con un int

print("Ornitorrinco " + "45")