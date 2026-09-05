from shared.preprocessing import preprocess

sample = """This   is a test.

It has  irregular    spacing and
line breaks.  Does it work correctly?"""

result = preprocess(sample)
print(f"Normalized text: {repr(result['normalized_text'])}")
print(f"\nSentences ({len(result['sentences'])}):")
for i, s in enumerate(result['sentences']):
    print(f"  {i+1}. {s}")