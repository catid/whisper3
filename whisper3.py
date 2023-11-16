from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import re
import time


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model = model.to_bettertransformer()
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)




t0 = time.time()

result = pipe("the-gettysburg-address.mp3")

t1 = time.time()

result = pipe("gettysburg_johng_librivox.m4b")

t2 = time.time()

result = pipe("the-gettysburg-address.mp3")

t3 = time.time()

print(f"The operation took {t1 - t0} seconds the first time.")
print(f"The operation took {t2 - t1} seconds the second time.")
print(f"The operation took {t3 - t2} seconds the third time.")


result_text = result["text"]
print(result_text)

expected = """
The Gettysburg Address by Abraham Lincoln, delivered November 19, 1863.

Four score and seven years ago, our fathers brought forth
upon this continent a new nation:  conceived in liberty, and
dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war. . .testing whether
that nation, or any nation so conceived and so dedicated. . .
can long endure.  We are met on a great battlefield of that war.

We have come to dedicate a portion of it as a final resting place
for those who died here that the nation might live.
This we may in all propriety do.

But, in a larger sense, we cannot dedicate. . .we cannot consecrate. . .
we cannot hallow this ground.  The brave men, living and dead,
who struggled here have hallowed it, far above our poor power
to add or detract.  The world will little note, nor long remember,
what we say here, while it can never forget what they did here.

It is rather for us, the living, we here, be dedicated to the great task remaining
before us. . .that from these honored dead we take increased devotion
to that cause for which they here gave the last full measure of devotion. . .
that we here highly resolve that these dead shall not have died in vain. . .
that this nation shall have a new birth of freedom. . .
and that government of the people. . .by the people. . .for the people. . .
shall not perish from the earth."""

def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation and special characters (keep only alphanumeric and spaces)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

if normalize_string(result_text) == normalize_string(expected):
    print("Whisper3 result matches expected text")
else:
    print("Text does not match!")

    def text_until_difference(str1, str2):
        # Find the length of the shorter string
        min_length = min(len(str1), len(str2))

        # Initialize an index to track the position of the first difference
        first_diff_index = min_length  # Default to min_length in case no difference is found

        # Compare each character
        for i in range(min_length):
            if str1[i] != str2[i]:
                first_diff_index = i
                break

        # Extract the substring from the beginning to the first difference
        return str1[:first_diff_index]

    print("Text leading up to difference: ", text_until_difference(normalize_string(result_text), normalize_string(expected)))
