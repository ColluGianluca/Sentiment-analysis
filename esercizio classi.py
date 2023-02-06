import pandas as pd


class Madre():
    def __init__(self,file):
        self.file=file
        self.text = self.pulizia()

    def load_info(self):
        text = list()
        with open(self.file) as infile:
            text = [line.strip() for line in infile]
        return text

    def get_info(self):
        text=self.load_info()
        numbers,n_address,n_owners=[],[],[]
        for k, line in enumerate(text):
            if '@' in line: numbers.append(k)
            if 'owner' in line: n_owners.append(k)
            if 'address' in line: n_address.append(k)
        return(numbers,n_owners,n_address)

    def pulizia(self):
        file=self.load_info()
        clean_text=[]
        for text in file:
         clean_text.append(text.replace('--', '').replace('@restaurant ', ''))
        return clean_text
    #     print(clean_text)
class Owner(Madre):
    def __init__(self,file):
        super().__init__(file)


    def owner_names(self):
         self.list_owner = [text.split(' ') for text in self.text[1:len(self.text):4]]
         lista= [ ' '.join(nome[1:3]) for nome in self.list_owner]
         Codice_Fiscale= [cf[-1] for cf in self.list_owner]
         return  lista, Codice_Fiscale

    def rest_numbers(self):
        owners=self.owner_names()[0]
        return owners

    def get_owner(self,prop):  # in ingresse pippone
        owner = {}
        info = self.get_info()
        owner[self.text[1]]=[]
        for own in info[1]:
            print(self.text[own])
            if self.text[own]==prop:
             owner[self.text[own]].append(self.text[own-1])
        return owner   # torna dizionario Ristorante:Owenr

    def get_owner_rest(self,owner,df):
        diz={}
        diz[owner]=[]
        prop, rist = df['Owner'].tolist(), df['Ristorante'].tolist()
        for i in range(0,len(prop)):
            if prop[i] == owner:
                diz[owner].append(rist[i])
        print(diz)


class ristoranti(Madre):

    def __init__(self,file):
        super().__init__(file)

    def restaurant_names(self):
        self.lista_rest=[text for text in self.text[0:len(self.text):4]]
        return self.lista_rest

    def get_owner(self,rist):
        owner = {}
        info = self.get_info()
        for own in info[0]:
            owner[self.text[own]] = self.text[own+1]
        return owner[rist]

    def get_address(self,rist):   # prende da rino
        address={}
        info=self.get_info()
        for rest in info[0]:
            address[self.text[rest]]=self.text[rest+2]
            print(address)
        return address[rist]

file='restaurants.lst'
nome=Owner(file)
ristoranti=ristoranti(file)
#print(nome.get_owner_rest('Beppe Murgia'))
data={'Ristorante': [ristorante for ristorante in ristoranti.restaurant_names()],
      'Owner': [owner for owner in nome.owner_names()[0]],
      'Codice Fiscale': [CF for CF in nome.owner_names()[1]]}
dataframe=pd.DataFrame(data)



print(nome.get_owner_rest('Beppe Murgia',dataframe))
#get_dict([ristorante for ristorante in ristoranti.restaurant_names()],[owner for owner in nome.owner_names()[0]])

# diz=dataframe.to_dict()
# prop=dataframe.Owner.values
# new=prop.tolist()


