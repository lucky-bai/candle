from tokenizers import SentencePieceUnigramTokenizer
import sentencepiece as spm

# From https://huggingface.co/kyutai/stt-1b-en_fr/tree/main
spm_path = "tokenizer_en_fr_audio_8000.model"

# Load the SentencePiece model to extract vocabulary
sp_model = spm.SentencePieceProcessor()
sp_model.load(spm_path)

# Extract vocabulary
vocab = []
for i in range(sp_model.get_piece_size()):
    piece = sp_model.id_to_piece(i)
    score = sp_model.get_score(i)
    vocab.append((piece, score))

# Create tokenizer with extracted vocabulary
tok = SentencePieceUnigramTokenizer(vocab=vocab)
tok.save("tokenizer_en_fr_audio_8000.json")
