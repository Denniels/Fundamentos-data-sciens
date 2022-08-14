nombre = input('Ingrese su nombre: ')
apellido = input('Ingrese su apellido: ')
edad = int(input('Ingrese su edad: '))
actividad1 = str(input('Ingrese su primera actividad favorita: '))
actividad2 = str(input('Ingrese su segunda actividad favorita: '))
actividad3 = str(input('Ingrese su tercera actividad favorita: '))
mascota = input('¿Tiene mascota?: ').lower()
nombre_mascota = ''
actividades = []

actividades.append(actividad1)
actividades.append(actividad2)
actividades.append(actividad3)

print(actividades)
print(type(actividades))

if mascota == 'no':
    print(f'Su nombre es: {nombre.capitalize()} {apellido.capitalize()}, su edad es: {edad} años, mi actividad favorita es: {actividades[1]} y {mascota} tienes mascota')
else:
    nombre_mascota = input('Como se llama tu mascota: ')
    print(f'Su nombre es: {nombre.capitalize()} {apellido.capitalize()}, su edad es: {edad} años, mi actividad favorita es: {actividades[1]} y {mascota} tienes mascota y se llama {nombre_mascota.capitalize()}.')

if __name__ == '__main__':
    pass