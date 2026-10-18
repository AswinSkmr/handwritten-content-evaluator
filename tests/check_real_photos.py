"""
Checks handwriting recognition on real phone photos against known
text, with and without background flattening. Run as a module:
    python -m tests.check_real_photos
Not named test_*.py on purpose, so pytest doesn't load the model
every time the suite runs.
"""

from jiwer import cer

from pipeline_b_handwritten.extract_handwriting import extract_text_from_handwriting

TRUTH_PHOTO = "Hello my name is Aswin. I am 23 years old. I am currently pursuing MCA."
TRUTH_IAM = 'assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way .'

CASES = [
    ("data/test_samples/img1.jpg", TRUTH_PHOTO),
    ("data/test_samples/img2.jpg", TRUTH_PHOTO),
    ("data/test_samples/iam_sample.png", TRUTH_IAM),
]

for path, truth in CASES:
    print(f"=== {path} ===")
    for flatten in (False, True):
        predicted = extract_text_from_handwriting(path, flatten_background=flatten)
        error = cer(truth, " ".join(predicted.split()))
        label = "flattened background" if flatten else "plain grayscale"
        print(f"{label}:")
        print(f"  {predicted!r}")
        print(f"  CER: {error:.4f}\n")