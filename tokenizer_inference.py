from athena.tokenizer import tokenizer

if __name__ == "__main__":
    inputs = [
        "Hello World",
        "HelloWorld",
        "fooBarBaz",
        "end-to-end",
        """And so they began. Harry, at least, felt extremely foolish, staring
blankly at the crystal ball, trying to keep his mind empty when thoughts
such as "this is stupid" kept drifting across it. It didn't help that
Ron kept breaking into silent giggles and Hermione kept tutting.

"Seen anything yet?" Harry asked them after a quarter of an hour's quiet
crystal gazing.

"Yeah, there's a burn on this table," said Ron, pointing. "Someone's
spilled their candle."

"This is such a waste of time," Hermione hissed. "I could be practicing
something useful. I could be catching up on Cheering Charms --"

Professor Trelawney rustled past."""
    ]

    for text in inputs:
        # 1) encode → list of token IDs (no special padding)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # 2) convert IDs → token strings
        tokens = tokenizer.convert_ids_to_tokens(ids)
        # 3) decode back to text
        decoded = tokenizer.decode(ids)

        print(f"Original : {text}")
        print(f"IDs      : {ids}")
        print(f"Tokens   : {tokens}")
        print(f"Decoded  : {decoded}")
        print(f"Match    : {decoded == text}")
        print("-" * 60)
