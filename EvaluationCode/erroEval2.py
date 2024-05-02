import matplotlib.pyplot as plt


class_to_index = {'16qam':0, '8qam':1, 'am':2,'fsk':3,'qpsk':4,'64qam':5,'ask':6,'fm':7,'pm':8}
errors_by_class = {class_name: 0 for class_name in class_to_index.values()}

result_dict = {'16qam':17, '8qam':18, 'am':20,'fsk':17,'qpsk':16,'64qam':18,'ask':18,'fm':17,'pm':19}


plt.figure(figsize=(10, 6))
plt.bar(result_dict.keys(), result_dict.values(), color='skyblue')
plt.xlabel('Clase')
plt.ylabel('Valor')
plt.title('Valor por Clase')
plt.xticks(rotation=45)
plt.show()