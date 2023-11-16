# Whisper 3 test

This is just a quick experiment to see how to use Whisper 3.


## Setup

Install conda and run:

```bash
git clone https://github.com/catid/whisper
cd whisper
conda create -n whisper python=3.10
conda activate whisper
pip install -U -r requirements.txt
```

## Test

```bash
cd whisper
conda activate whisper
python whisper3.py
```

Produces:

```bash
The operation took 1.5356569290161133 seconds the first time.
The operation took 1.203697681427002 seconds the second time.
The operation took 1.0399818420410156 seconds the third time.

 The Gettysburg Address by Abraham Lincoln, delivered November 19, 1863. Four score and seven years ago, our fathers brought forth upon this continent a new nation conceived in liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation, so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. We have come to dedicate a portion of it as a final resting place for those who died here, that the nation might live. This we may in all propriety do. But in a larger sense, we cannot dedicate, we cannot consecrate, we cannot hallow this ground. The brave men, living and dead, who struggled here have hallowed it, far above our poor power to add or detract. The world will little note nor long remember what we say here, while it can never forget what they did here. It is rather for us, the living, we here, be dedicated to the great task remaining before us, that from these honored dead we take increased devotion to that cause for which they here gave the last full measure of devotion, that we here highly resolve that these dead shall not have died in vain, that this nation shall have a new birth of freedom, and that government of the people, by the people, for the people, shall not perish from the earth.

Whisper3 result matches expected text
```

## Discussion

Flash Attention 2 seems fairly tricky to get set up for users, and it does not improve the runtime during steady state.  It seems to only help the first time we use the pipeline.  After that it takes the same amount of time each time.
So for an open-source project I wouldn't bother with it.

However using BetterTransformer model was easy to get set up and improved performance noticeably!

I tried speculative decoding as a speed optimization and it was not functional either on latest stable or main transformers.  Fortunately Whisper 3 provides 2x speed upgrade already over previous versions.  According to community notes this will be fixed when they release the distilled version of v3.
