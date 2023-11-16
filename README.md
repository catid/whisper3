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

## Discussion

Flash Attention 2 seems fairly tricky to get set up for users, and it does not improve the runtime during steady state.  It seems to only help the first time we use the pipeline.  After that it takes the same amount of time each time.
So for an open-source project I wouldn't bother with it.

However using BetterTransformer model was easy to get set up and improved performance noticeably!

I tried speculative decoding as a speed optimization and it was not functional either on latest stable or main transformers.  Fortunately Whisper 3 provides 2x speed upgrade already over previous versions.  According to community notes this will be fixed when they release the distilled version of v3.
