from indexing.lsi_model import LSIRetriever, load_documents_from_json


def main() -> None:
    documents = load_documents_from_json("data/documents.json")

    lsi = LSIRetriever(
        n_components=2,
        max_features=5000,
        stop_words="english"
    )

    lsi.fit(documents)
    lsi.save("models")

    print("Modelo LSI entrenado y guardado en ./models")

    query = input("Escribe tu consulta: ").strip()
    results = lsi.search(query, top_k=3)

    print("\nResultados:")
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. [{result.doc_id}] {result.title}")
        print(f"Score: {result.score:.4f}")
        print(f"Contenido: {result.content}")


if __name__ == "__main__":
    main()