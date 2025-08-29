from athena.tokenizer import tokenizer

if __name__ == "__main__":
    inputs = [
        "Hello World",
        "HelloWorld",
        "fooBarBaz",
        "end-to-end",
        """Worked with a small team of professional developers to create a new chess engine: Torch, as of now the world's 2nd strongest chess engine. 
Wrote code to efficiently execute sparse neural networks by shuffling neurons and skipping the computation of connections that don’t affect the output. 
My contributions resulted in an immediate 20% increase in network execution speed.
Before my changes, a 100% increase in network size would result in a 100% increase in computational cost. After my changes, a 100% increase in network size only increased computational cost by 35%. This allowed us to greatly expand the size of our neural networks."""
    ]

    for text in inputs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(ids)

        print(f"Original : {text}")
        print(f"IDs      : {ids}")
        print(f"Tokens   : {tokens}")
        print(f"Decoded  : {decoded}")
        print(f"Match    : {decoded == text}")
        print("-" * 60)
