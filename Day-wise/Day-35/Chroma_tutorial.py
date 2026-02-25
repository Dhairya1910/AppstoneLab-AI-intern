import chromadb 

db_path = "d:\AppstoneLab-AI-intern\ChromaDB"
client = chromadb.PersistentClient(path = db_path)

# Created collection
collection = client.create_collection(name = "Anime")
print("Collection Created : ",collection.name)

# added documents and ids.
collection.add(
    documents=[
        "Naruto is my first anime.",
        "Naruto's second sequal is Shipudden.",
        "Deathnote has best mystery element in all the anime",
        "Levi is indeed a monster."
    ],
    ids = ["Naruto","Naruto Shippuden","Deathnote","Attack on titans"]
)

res = collection.query(
    query_texts= ['recommend me an action anime'],
    n_results=2
)

print(res)
#Output 
# Collection Created :  Anime
# {'ids': [['Naruto', 'Deathnote']], 'embeddings': None, 'documents': [['Naruto is my first anime.', 'Deathnote has best mystery element in all the anime']], 'uris': None, 'included': ['metadatas', 'documents', 'distances'], 'data': None, 'metadatas': [[None, None]], 'distances': [[0.9383842349052429, 1.1242341995239258]]}