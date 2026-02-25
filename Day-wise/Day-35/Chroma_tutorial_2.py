import chromadb 


client = chromadb.PersistentClient(path="d:\AppstoneLab-AI-intern\ChromaDB")
collection = client.get_collection('Anime')
print(collection)

#Output 
# Collection(name=Anime)